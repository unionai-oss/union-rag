"""Union and Flyte chat assistant workflow using Ollama server."""

import os
import time
from functools import wraps
from subprocess import Popen
from typing import Optional
from flytekit import (
    task,
    workflow,
    Resources,
)
from flytekit.extras import accelerators
from flytekit.types.directory import FlyteDirectory

from union_rag.simple_rag import image, VectorStore, DEFAULT_PROMPT_TEMPLATE


OLLAMA_MODEL_NAME = "llama3"

image_ollama = image.with_apt_packages(["lsof"]).with_commands(
    [
        "wget https://ollama.com/install.sh",
        "sh ./install.sh",
        f"sh ./ollama_serve.sh {OLLAMA_MODEL_NAME} /root/.ollama/models",
    ]
)
image_ollama.source_root = "."


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
    search_index: FlyteDirectory = VectorStore.query(embedding_type="huggingface"),
    prompt_template: Optional[str] = None,
) -> str:
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain_community.chat_models import ChatOllama
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate

    search_index.download()
    index = FAISS.load_local(
        search_index.path,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True,
    )

    chain = load_qa_with_sources_chain(
        ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.9),
        prompt=PromptTemplate.from_template(prompt_template or DEFAULT_PROMPT_TEMPLATE),
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
def ask_ollama(
    question: str,
    search_index: FlyteDirectory = VectorStore.query(embedding_type="huggingface"),
    prompt_template: Optional[str] = None,
) -> str:
    return answer_question_ollama(
        question=question,
        search_index=search_index,
        prompt_template=prompt_template,
    )
