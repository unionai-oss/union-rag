from dataclasses import dataclass
from langchain_core.documents import Document
from typing import Any, Iterator, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from mashumaro.mixins.json import DataClassJSONMixin

from flytekit.types.file import FlyteFile

from langchain_community.document_transformers import BeautifulSoupTransformer

class HTML2MarkdownTransformer(BeautifulSoupTransformer):

    def __init__(self, root_url_tags_mapping: dict[str, tuple[str, dict]] = None):
        self.root_url_tags_mapping = root_url_tags_mapping

    def transform_documents(
        self,
        documents: Iterator[Document],
        unwanted_tags: list[str | tuple[str, dict]] = ["script", "style"],
        tags_to_extract: list[str] = ["p", "li", "div", "a"],
        remove_lines: bool = True,
        **kwargs: Any,
    ) -> Iterator[Document]:
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
            yield doc
    
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