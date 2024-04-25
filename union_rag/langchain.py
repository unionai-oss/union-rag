"""Flyte attendant workflow.

TODO:
- ingest slack threads:
  - https://python.langchain.com/docs/integrations/document_loaders/slack/
  - https://python.langchain.com/docs/integrations/chat_loaders/slack/
"""

import itertools
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, List, Optional, Sequence
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from mashumaro.mixins.json import DataClassJSONMixin

from flytekit import task, workflow, current_context, wait_for_input, ImageSpec, Secret
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


image = ImageSpec(
    builder="unionai",
    apt_packages=["git"],
    requirements="requirements.txt",
    env={"GIT_PYTHON_REFRESH": "quiet"},
)


class HTML2MarkdownTransformer(BeautifulSoupTransformer):

    def __init__(self, root_url_tags_mapping: dict[str, tuple[str, dict]] = None):
        self.root_url_tags_mapping = root_url_tags_mapping

    def transform_documents(
        self,
        documents: Sequence[Document],
        unwanted_tags: list[str | tuple[str, dict]] = ["script", "style"],
        tags_to_extract: list[str] = ["p", "li", "div", "a"],
        remove_lines: bool = True,
        **kwargs: Any,
    ) -> Sequence[Document]:
        for doc in documents:
            cleaned_content = doc.page_content
            cleaned_content = self.remove_unwanted_tags(cleaned_content, unwanted_tags)
            cleaned_content = self.extract_tags(
                cleaned_content,
                self.get_root_tag(doc.metadata["source"]),
            )
            if remove_lines:
                cleaned_content = self.remove_unnecessary_lines(cleaned_content)
            doc.page_content = cleaned_content

        return documents
    
    def get_root_tag(self, source: str):
        for url, tag in self.root_url_tags_mapping.items():
            if source.startswith(url):
                return tag
        raise ValueError(f"Unknown source: {source}")
            
    @staticmethod
    def remove_unwanted_tags(html_content: str, unwanted_tags: List[str]) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in unwanted_tags:
            if isinstance(tag, str):
                tag = [tag]
            for element in soup.find_all(*tag):
                element.decompose()
        return str(soup)

    @staticmethod
    def extract_tags(
        html_content: str,
        root_tag: tuple[str, dict],
    ) -> str:
        """Custom content extraction."""
        soup = BeautifulSoup(html_content, "html.parser")
        # the <article role="main"> tag contains all the content of a page
        content = soup.find_all(*root_tag)
        if len(content) == 0:
            return ""
        content = content[0]
        return markdownify(str(content)).replace("\n\n\n\n", "\n\n").strip()


@dataclass
class FlyteDocument(DataClassJSONMixin):
    page_filepath: FlyteFile
    metadata: dict

    def to_document(self) -> Document:
        with open(self.page_filepath) as f:
            page_content = f.read()
        return Document(page_content=page_content, metadata=self.metadata)
    

def get_all_links(url, base_domain, visited: set, limit: Optional[int] = None):
    if url in visited or (limit is not None and len(visited) > limit):
        return visited

    visited.add(url)
    print("Adding", url)
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        for link in soup.find_all('a', href=True):
            full_link = urljoin(url, link['href'])
            full_link = full_link.split("#")[0]
            full_link = full_link.split("?")[0]
            if full_link.startswith(base_domain):
                visited = get_all_links(full_link, base_domain, visited, limit)
    except requests.exceptions.RequestException as e:
        print(f"Failed to access {url}: {str(e)}")
    return visited


def get_links(starting_url: str, limit: Optional[int] = None) -> List[str]:
    print(f"Collecting urls at {starting_url}")
    all_links = get_all_links(
        starting_url, starting_url, visited=set(), limit=limit
    )
    return list(all_links)



@task(
    container_image=image,
    cache=True,
    cache_version="1",
)
def get_documents(
    root_url_tags_mapping: Optional[dict] = None,
    include_union: bool = False,
    limit: Optional[int] = None,
) -> list[FlyteDocument]:

    if root_url_tags_mapping is None:
        root_url_tags_mapping = {
            "https://docs.flyte.org": ("article", {"role": "main"}),
            "https://flyte.org/blog": ("div", {"class": "blog-content"}),
        }
        if include_union:
            root_url_tags_mapping.update({
                "https://docs.union.ai": ("div", {"class": "content-container"}),
                "https://www.union.ai/blog-post": ("div", {"class": "blog-content"}),
            })

    page_transformer = HTML2MarkdownTransformer(root_url_tags_mapping)
    urls = list(
        itertools.chain(*(get_links(url, limit) for url in root_url_tags_mapping))
    )
    loader = AsyncHtmlLoader(urls)
    html = loader.load()

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
        with path.open("w") as f:
            f.write(doc.page_content)
        documents.append(FlyteDocument(page_filepath=path, metadata=doc.metadata))

    return documents


def set_openai_key():
    os.environ["OPENAI_API_KEY"] = current_context().secrets.get("openai_api_key")


@task(
    container_image=image,
    cache=True,
    cache_version="5",
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
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
)
def answer_question(question: str, search_index: FlyteDirectory) -> str:
    set_openai_key()
    search_index.download()
    index = FAISS.load_local(
        search_index.path,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    chain = load_qa_with_sources_chain(
        ChatOpenAI(
            model_name="gpt-4-0125-preview",
            temperature=0.9,
        )
    )
    answer = chain(
        {
            "input_documents": index.similarity_search(question, k=8),
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


@workflow
def ask_with_feedback(question: str, chunk_size: int = 1024) -> str:
    answer = ask(question=question, chunk_size=chunk_size)
    feedback = wait_for_input("get-feedback", timeout=timedelta(hours=1), expected_type=str)
    answer >> feedback
    return feedback
