import functools
import itertools
import json
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Annotated, Dict, Optional

from flytekit import task, workflow, Resources, Artifact, Secret, map_task, current_context, wait_for_input
from flytekit.core.artifact import Inputs
from flytekit.types.file import FlyteFile

from union_rag.document import CustomDocument
from union_rag.simple_rag import image, KnowledgeBase

DEFAULT_ANNOTATION_SET_NAME = "default"

QuestionAndAnswerDataset = Artifact(name="question_and_answer_dataset")
AnnotatedDataset = Artifact(name="annotated_dataset", partition_keys=["name"])

@dataclass
class QuestionAndAnswers:
    question: str
    answers: List[str]

@dataclass
class AnnotatedQuestions:
    id: int
    question: str
    answer_1: str
    answer_2: str
    preferred_answer: Optional[str]
    user_answer: Optional[str]

@task(
    container_image=image,
    requests=Resources(cpu="2", mem="8Gi"),
    secret_requests=[Secret(key="openai_api_key")],
    cache=True,
    cache_version="3",
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
    cache_version="7",
)
def create_dataset(questions_and_answers: List[QuestionAndAnswers]) -> FlyteFile:
    qa_triples_list = []
    id_counter = 1
    for i, qa in enumerate(questions_and_answers):
        question = qa.question
        answers = qa.answers
        answer_combinations = list(itertools.combinations(answers, 2))

        for j, combo in enumerate(answer_combinations):
            qa_triples_list.append(
                {
                    "id": id_counter,
                    "question": question,
                    "answers": list(combo),
                }
            )
            id_counter += 1

    file_path = 'qa_triples_list.json'
    with open(file_path, 'w') as f:
        json.dump(qa_triples_list, f, indent=4)

    return FlyteFile(path=file_path)


@workflow
def data_synthesis_workflow(documents: List[CustomDocument] = KnowledgeBase.query(), n_answers: int = 5) -> Annotated[FlyteFile, QuestionAndAnswerDataset]:
    partial_task = functools.partial(generate_qa_pair, n_answers=n_answers)
    questions_and_answers = map_task(partial_task)(flyte_doc=documents)
    dataset = create_dataset(questions_and_answers=questions_and_answers)
    return dataset

@task(
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
    cache=True,
    cache_version="1",
)
def sample_triple(random_seed: int, n_samples: int, dataset: FlyteFile) -> List[Dict]:
    dataset.download()
    with open(dataset.path, 'r') as f:
        qa_triples_list = json.load(f)

    assert len(
        qa_triples_list) >= n_samples, f"Not enough samples in the dataset. Required: {n_samples}, Available: {len(qa_triples_list)}"

    random.seed(random_seed)
    sampled_qa = random.sample(qa_triples_list, k=n_samples)

    return sampled_qa

@task(
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
)
def append_annotated_data(feedback: str, master_annotation_dataset: FlyteFile) -> FlyteFile:
    parsed_annotation_json = json.loads(feedback)

    master_annotation_dataset.download()
    with open(master_annotation_dataset.path, 'r') as file:
        master_annotation_dataset_list = json.load(file)

    master_annotation_dataset_list.extend(parsed_annotation_json)

    print(master_annotation_dataset_list)

    file_path = 'master_annotation_dataset.json'
    with open(file_path, 'w') as f:
        json.dump(master_annotation_dataset_list, f, indent=4)

    return FlyteFile(path=file_path)

@workflow
def create_annotation_set(random_seed: int, n_samples: int = 10, dataset: FlyteFile = QuestionAndAnswerDataset.query(), annotation_set_name: str = DEFAULT_ANNOTATION_SET_NAME, master_annotation_dataset: FlyteFile = AnnotatedDataset.query(name=Inputs.annotation_set_name)) -> Annotated[FlyteFile, AnnotatedDataset(name=Inputs.annotation_set_name)]:
    triple_sample = sample_triple(random_seed=random_seed, n_samples=n_samples, dataset=dataset)
    feedback = wait_for_input("feedback", timeout=timedelta(hours=1), expected_type=str)
    triple_sample >> feedback
    annotated_dataset = append_annotated_data(feedback=feedback, master_annotation_dataset=master_annotation_dataset)
    return annotated_dataset


@task(container_image=image)
def init_annotation_set() -> FlyteFile:
    file_path = 'master_annotation_dataset.json'
    with open(file_path, 'w') as f:
        json.dump([], f, indent=4)

    return FlyteFile(path=file_path)

@workflow
def init_annotation_set_wf(annotation_set_name: str = DEFAULT_ANNOTATION_SET_NAME) -> Annotated[FlyteFile, AnnotatedDataset(name=Inputs.annotation_set_name)]:
    return init_annotation_set()


if __name__ == "__main__":
    data_synthesis_workflow()