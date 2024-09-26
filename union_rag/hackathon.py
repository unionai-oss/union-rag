import functools
import os
from dataclasses import dataclass
from typing import List, Annotated

import pandas as pd
from flytekit import task, workflow, Resources, Artifact, Secret, map_task, current_context

from union_rag.document import CustomDocument
from union_rag.simple_rag import image, KnowledgeBase

QuestionAndAnswerDataset = Artifact(name="question_and_answer_dataset")

@dataclass
class QuestionAndAnswers:
    question: str
    answers: List[str]

@task(
    container_image=image,
    requests=Resources(cpu="2", mem="8Gi"),
    secret_requests=[Secret(key="openai_api_key")],
    cache=True,
    cache_version="1",
)
def generate_qa_pair(flyte_doc: CustomDocument, n_answers: int) -> QuestionAndAnswers:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    os.environ["OPENAI_API_KEY"] = current_context().secrets.get(key="openai_api_key")

    document = flyte_doc.to_document()

    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)

    question_prompt = PromptTemplate(
        input_variables=["context"],
        template="Given the following context, generate a relevant question:\n\nContext: {context}\n\nQuestion:"
    )

    answer_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the following context and question, provide a detailed answer:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    question_chain = LLMChain(llm=llm, prompt=question_prompt)
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

    question = question_chain.run(context=document.page_content)
    answers = [answer_chain.run(context=document.page_content, question=question) for _ in range(n_answers)]

    return QuestionAndAnswers(question=question, answers=answers)


@task(
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
    cache=True,
    cache_version="1",
)
def create_dataset(questions_and_answers: List[QuestionAndAnswers], n_answers: int) -> pd.DataFrame:
    data = []
    for i, question_and_answers in enumerate(questions_and_answers):
        question = question_and_answers.question
        answers = question_and_answers.answers
        data_point = {
            'id': f'qa_{i}',
            'question': question
        }
        for j, answer in enumerate(answers, start=1):
            data_point[f'answer_{j}'] = answer
        data.append(data_point)

    return pd.DataFrame(data)


@workflow
def data_synthesis_workflow(documents: List[CustomDocument] = KnowledgeBase.query(), n_answers: int = 5) -> Annotated[pd.DataFrame, QuestionAndAnswerDataset]:
    partial_task = functools.partial(generate_qa_pair, n_answers=n_answers)
    questions_and_answers = map_task(partial_task)(flyte_doc=documents)
    dataset = create_dataset(questions_and_answers=questions_and_answers, n_answers=n_answers)

    return dataset

if __name__ == "__main__":
    data_synthesis_workflow()