"""Flyte attendant workflow."""

import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mashumaro.mixins.json import DataClassJSONMixin

from flytekit import task, workflow, current_context, Secret
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from langchain.docstore.document import Document

from git import Repo
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.llms import OpenAI


@dataclass
class FlyteDocument(DataClassJSONMixin):
    page_filepath: FlyteFile
    metadata: dict

    def to_document(self) -> Document:
        with open(self.page_filepath) as f:
            page_content = f.read()
        return Document(page_content=page_content, metadata=self.metadata)
    

def set_openai_key():
    os.environ["OPENAI_API_KEY"] = current_context().secrets.get("openai_api_key")


@task(container_image="ghcr.io/unionai-oss/union-rag:latest")
def get_documents(
    urls: list[str],
    extensions: Optional[list[str]] = None,
    exclude_files: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> list[FlyteDocument]:
    """Fetch documents from a github url."""
    extensions = extensions or [".py", ".md", ".rst"]
    exclude_files = frozenset(exclude_files or ["__init__.py"])
    exclude_patterns = exclude_patterns or []

    output_dir = "."
    documents = []
    for url in urls:
        repo = Repo.clone_from(url, output_dir)
        git_sha = repo.head.commit.hexsha
        git_dir = Path(output_dir)

        exclude_from_patterns = [
            *itertools.chain(*(git_dir.glob(p) for p in exclude_patterns))
        ]

        for file in itertools.chain(
            *[git_dir.glob(f"**/*{ext}") for ext in extensions]
        ):
            if file.name in exclude_files or file in exclude_from_patterns:
                continue

            github_url = f"{url}/blob/{git_sha}/{file.relative_to(git_dir)}"
            documents.append(FlyteDocument(file, metadata={"source": github_url}))

    return documents


@task(
    container_image="ghcr.io/unionai-oss/union-rag:latest",
    cache=True,
    cache_version="0",
    secret_requests=[Secret(key="openai_api_key")],
)
def create_search_index(
    documents: list[FlyteDocument],
    chunk_size: int,
) -> FlyteDirectory:
    set_openai_key()
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
    container_image="ghcr.io/unionai-oss/union-rag:latest",
    cache=True,
    cache_version="0",
    secret_requests=[Secret(key="openai_api_key")],
)
def answer_question(question: str, search_index: FlyteDirectory) -> str:
    set_openai_key()
    index = FAISS.load_local(search_index.path, OpenAIEmbeddings())
    chain = load_qa_with_sources_chain(OpenAI(temperature=0.9))
    answer = chain(
        {
            "input_documents": index.similarity_search(question, k=3),
            "question": question,
        },
    )
    return answer["output_text"]


@workflow
def ask(question: str, chunk_size: int = 1024) -> str:
    docs = get_documents(urls=["https://github.com/flyteorg/flytesnacks"])
    search_index = create_search_index(documents=docs, chunk_size=chunk_size)
    answer = answer_question(question=question, search_index=search_index)
    return answer
