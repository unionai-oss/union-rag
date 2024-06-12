"""Flyte attendant workflow."""

import itertools
import os
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path
from subprocess import Popen, run
from typing import Annotated, Optional

from flytekit import (
    task,
    workflow,
    current_context,
    wait_for_input,
    Artifact,
    ImageSpec,
    Secret,
    Resources,
)
from flytekit.extras import accelerators
from flytekit.types.directory import FlyteDirectory

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from union_rag.document import get_links, CustomDocument, HTML2MarkdownTransformer


image = ImageSpec(
    builder="unionai",
    apt_packages=["git", "wget", "curl"],
    requirements="requirements.lock.txt",
    env={"GIT_PYTHON_REFRESH": "quiet"},
    cuda="11.8",
    source_root=".",
)

image_ollama = (
    image
    .with_apt_packages(["lsof"])
    .with_commands([
        "sh ./ollama_install.sh",
        "sh ./ollama_serve.sh phi3 /root/.ollama/models",
    ])
    .force_push()
)


KnowledgeBase = Artifact(name="knowledge-base")
VectorStore = Artifact(name="vector-store")


@task(
    container_image=image,
    cache=True,
    cache_version="3",
    requests=Resources(cpu="2", mem="8Gi"),
    enable_deck=True,
)
def get_documents(
    root_url_tags_mapping: Optional[dict] = None,
    include_union: bool = False,
    limit: Optional[int] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> Annotated[list[CustomDocument], KnowledgeBase]:
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
        itertools.chain(
            *(get_links(url, limit, exclude_patterns) for url in root_url_tags_mapping)
        )
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
        documents.append(CustomDocument(page_filepath=path, metadata=doc.metadata))

    return documents


@task(
    container_image=image,
    cache=True,
    cache_version="6",
    requests=Resources(cpu="2", mem="8Gi"),
    secret_requests=[Secret(key="openai_api_key")],
    enable_deck=True,
)
def create_search_index(
    documents: list[CustomDocument] = KnowledgeBase.query(),
    chunk_size: int | None = None,
    embedding_type: str = "openai",
) -> Annotated[FlyteDirectory, VectorStore]:

    chunk_size = chunk_size or 1024
    os.environ["OPENAI_API_KEY"] = current_context().secrets.get("openai_api_key")
    documents = [flyte_doc.to_document() for flyte_doc in documents]
    splitter = CharacterTextSplitter(
        separator=" ", chunk_size=chunk_size, chunk_overlap=0
    )

    if embedding_type == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
    elif embedding_type == "huggingface":
        from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    index = FAISS.from_documents(
        [
            Document(page_content=chunk, metadata=doc.metadata)
            for doc in documents
            for chunk in splitter.split_text(doc.page_content)
        ],
        embeddings,
    )
    local_path = "./faiss_index"
    index.save_local(local_path)
    return FlyteDirectory(path=local_path)


@workflow
def create_knowledge_base(
    chunk_size: int = 1024,
    root_url_tags_mapping: Optional[dict] = None,
    include_union: bool = False,
    limit: Optional[int] = None,
    exclude_patterns: Optional[list[str]] = None,
    embedding_type: str = "openai",
) -> FlyteDirectory:
    docs = get_documents(
        root_url_tags_mapping=root_url_tags_mapping,
        include_union=include_union,
        limit=limit,
        exclude_patterns=exclude_patterns,
    )
    search_index = create_search_index(
        documents=docs,
        chunk_size=chunk_size,
        embedding_type=embedding_type,
    )
    return search_index


@task(
    container_image=image,
    requests=Resources(cpu="2", mem="8Gi"),
    secret_requests=[Secret(key="openai_api_key")],
    enable_deck=True,
)
def answer_question(
    question: str,
    search_index: FlyteDirectory = VectorStore.query(),
) -> str:
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
    answer = chain.invoke(
        {
            "input_documents": index.similarity_search(question, k=8),
            "question": question,
        },
    )
    output_text = answer["output_text"]
    return output_text


def ollama_server(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print("Starting Ollama server")
        server = Popen(["ollama", "serve"], env=os.environ)
        time.sleep(6)
        print("Done sleeping")

        try:
            return fn(*args, **kwargs)
        finally:
            print("Terminating Ollama server")
            server.terminate()
    return wrapper


@task(
    container_image=image_ollama,
    accelerator=accelerators.T4,
    requests=Resources(cpu="4", mem="24Gi", gpu="1"),
    environment={"OLLAMA_MODELS": "/root/.ollama/models"},
)
@ollama_server
def answer_question_ollama(
    question: str,
    search_index: FlyteDirectory = VectorStore.query(),
) -> str:
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

    search_index.download()
    index = FAISS.load_local(
        search_index.path,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True,
    )
    chain = load_qa_with_sources_chain(
        ChatOllama(
            model="phi3",
            temperature=0.9,
        )
    )
    answer = chain.invoke(
        {
            "input_documents": index.similarity_search(question, k=8),
            "question": question,
        },
    )
    output_text = answer["output_text"]
    return output_text


@workflow
def ask(
    question: str,
    search_index: FlyteDirectory = VectorStore.query(),
) -> str:
    return answer_question(question=question, search_index=search_index)


@workflow
def ask_ollama(
    question: str,
    search_index: FlyteDirectory = VectorStore.query(),
) -> str:
    return answer_question_ollama(question=question, search_index=search_index)


@workflow
def ask_with_feedback(
    question: str,
    search_index: FlyteDirectory = VectorStore.query(),
) -> str:
    answer = ask(
        question=question,
        search_index=search_index,
    )
    feedback = wait_for_input("get-feedback", timeout=timedelta(hours=1), expected_type=str)
    answer >> feedback
    return feedback
