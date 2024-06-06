"""Flyte attendant workflow."""

import itertools
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

from flytekit import (
    task,
    workflow,
    current_context,
    wait_for_input,
    ImageSpec,
    Secret,
    Resources,
)
from flytekit.types.directory import FlyteDirectory

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from union_rag.document import get_links, FlyteDocument, HTML2MarkdownTransformer
from union_rag.utils import to_slack_mrkdown


image = ImageSpec(
    builder="unionai",
    apt_packages=["git"],
    requirements="requirements.txt",
    env={"GIT_PYTHON_REFRESH": "quiet"},
)


@task(
    container_image=image,
    cache=True,
    cache_version="2",
    requests=Resources(cpu="2", mem="8Gi")
)
def get_documents(
    root_url_tags_mapping: Optional[dict] = None,
    include_union: bool = False,
    limit: Optional[int] = None,
) -> list[FlyteDocument]:

    if root_url_tags_mapping is None:
        root_url_tags_mapping = {
            "https://docs.flyte.org/en/latest/": ("article", {"role": "main"}),
        }
    if include_union:
        root_url_tags_mapping.update({
            "https://docs.union.ai/": ("article", {"class": "bd-article"}),
        })

    page_transformer = HTML2MarkdownTransformer(root_url_tags_mapping)
    urls = list(
        itertools.chain(*(get_links(url, limit) for url in root_url_tags_mapping))
    )
    loader = AsyncHtmlLoader(urls)
    html = loader.lazy_load()

    md_transformed = page_transformer.transform_documents(
        html,
        unwanted_tags=[
            "script",
            "style",
            ("a", {"class": "headerlink"}),
            ("button", {"class": "CopyButton"}),
            ("div", {"class": "codeCopied"}),
            ("span", {"class": "lang"}),
        ],
        remove_lines=False,
    )

    root_path = Path("./docs")
    root_path.mkdir(exist_ok=True)
    documents = []
    for i, doc in enumerate(md_transformed):
        if doc.page_content == "":
            print(f"Skipping empty document {doc}")
            continue
        path = root_path / f"doc_{i}.md"
        print(f"Writing document {doc.metadata['source']} to {path}")
        with path.open("w") as f:
            f.write(doc.page_content)
        documents.append(FlyteDocument(page_filepath=path, metadata=doc.metadata))

    return documents


@task(
    container_image=image,
    cache=True,
    cache_version="5",
    requests=Resources(cpu="2", mem="8Gi"),
    secret_requests=[Secret(key="openai_api_key")],
)
def create_search_index(
    documents: list[FlyteDocument],
    chunk_size: int,
) -> FlyteDirectory:
    os.environ["OPENAI_API_KEY"] = current_context().secrets.get("openai_api_key")
    documents = [flyte_doc.to_document() for flyte_doc in documents]
    splitter = CharacterTextSplitter(
        separator=" ", chunk_size=chunk_size, chunk_overlap=0
    )
    index = FAISS.from_documents(
        [
            Document(page_content=chunk, metadata=doc.metadata)
            for doc in documents
            for chunk in splitter.split_text(doc.page_content)
        ],
        OpenAIEmbeddings(),
    )
    local_path = "./faiss_index"
    index.save_local(local_path)
    return FlyteDirectory(path=local_path)


@task(
    container_image=image,
    requests=Resources(cpu="2", mem="8Gi"),
    secret_requests=[Secret(key="openai_api_key")],
    enable_deck=True,
)
def answer_question(question: str, search_index: FlyteDirectory) -> str:
    os.environ["OPENAI_API_KEY"] = current_context().secrets.get("openai_api_key")
    search_index.download()
    index = FAISS.load_local(
        search_index.path,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    chain = load_qa_with_sources_chain(
        ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.9,
        )
    )
    answer = chain(
        {
            "input_documents": index.similarity_search(question, k=8),
            "question": question,
        },
    )
    output_text = answer["output_text"]

    output_text = to_slack_mrkdown(output_text)
    return output_text


@workflow
def ask(
    question: str,
    chunk_size: int = 1024,
    root_url_tags_mapping: Optional[dict] = None,
    include_union: bool = False,
    limit: Optional[int] = None,
) -> str:
    docs = get_documents(
        root_url_tags_mapping=root_url_tags_mapping,
        include_union=include_union,
        limit=limit,
    )
    search_index = create_search_index(documents=docs, chunk_size=chunk_size)
    answer = answer_question(question=question, search_index=search_index)
    return answer


@workflow
def ask_with_feedback(
    question: str,
    chunk_size: int = 1024,
    root_url_tags_mapping: Optional[dict] = None,
    include_union: bool = False,
    limit: Optional[int] = None,
) -> str:
    answer = ask(
        question=question,
        chunk_size=chunk_size,
        root_url_tags_mapping=root_url_tags_mapping,
        include_union=include_union,
        limit=limit,
    )
    feedback = wait_for_input("get-feedback", timeout=timedelta(hours=1), expected_type=str)
    answer >> feedback
    return feedback
