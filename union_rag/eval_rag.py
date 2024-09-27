"""Evaluate a RAG workflow."""

from dataclasses import dataclass, asdict
from typing import Annotated, Optional
from functools import partial

import pandas as pd
from flytekit import dynamic, task, workflow, Artifact, Secret
from flytekit.types.directory import FlyteDirectory

from union_rag.simple_rag import create_knowledge_base, answer_question, image, RAGConfig


EvalDatasetArtifact = Artifact(name="test-eval-dataset")


@dataclass
class Question:
    question: str
    id: int


@dataclass
class Answer:
    answer: str
    question_id: int


@task(
    container_image=image,
    cache=True,
    cache_version="1",
)
def prepare_questions(dataset: pd.DataFrame, n_answers: int) -> list[Question]:
    questions = (
        dataset.loc[dataset.index.repeat(n_answers)][["id", "question"]]
        .astype({"id": int})
        .to_dict(orient="records")
    )
    return [Question(**record) for record in questions]


@task(
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
)
def answer_questions(
    questions: list[Question],
    search_index: FlyteDirectory,
    prompt_template: str,
) -> list[Answer]:
    answers = []
    for question in questions:
        answer = answer_question.task_function(
            question=question.question,
            search_index=search_index,
            prompt_template=prompt_template,
        )
        answers.append(Answer(answer=answer, question_id=question.id))
    return answers


@workflow
def batch_rag_pipeline(
    questions: list[Question],
    config: RAGConfig,
) -> list[Answer]:
    search_index = create_knowledge_base(config) 
    answers = answer_questions(questions, search_index, config.prompt_template)
    return answers


@dynamic(container_image=image, cache=True, cache_version="1")
def run_evaluation(questions: list[Question], eval_configs: list[RAGConfig]) -> list[list[Answer]]:
    answers = []
    for config in eval_configs:
        answers.append(batch_rag_pipeline(questions, config))
    return answers


@task(container_image=image)
def combine_answers(
    answers: list[list[Answer]],
    eval_configs: list[RAGConfig],
    questions: list[Question],
) -> pd.DataFrame:
    # TODO: concatenate all answers into a single dataframe
    combined_answers = []
    for _answers, config in zip(answers, eval_configs):
        for answer, question in zip(_answers, questions):
            assert answer.question_id == question.id
            combined_answers.append({
                "question_id": question.id,
                "question": question.question,
                "answer": answer.answer,
                **asdict(config),
            })

    return pd.DataFrame(combined_answers)


@task(container_image=image)
def evaluate_answers(answers_dataset: pd.DataFrame) -> pd.DataFrame:
    # TODO:
    # - Run traditional NLP eval (e.g. BLEU, ROUGE, etc.) on all answers
    #   against the reference answers.
    # - Create LLM judge to select the best answer for each question:
    #   "Is the RAG answer equivalent or better than the reference answer in terms of correctness? Answer Y/N"
    # - Calculate percent of RAG answers that beat the reference answers.
    # - This should create a dataframe where each row contains eval configs and corresponding metrics.
    return pd.DataFrame()


@workflow
def evaluate_simple_rag(
    eval_configs: list[RAGConfig],
    eval_dataset: Annotated[pd.DataFrame, EvalDatasetArtifact] = EvalDatasetArtifact.query(),
    n_answers: int = 5,
) -> pd.DataFrame:
    questions = prepare_questions(eval_dataset, n_answers)
    answers = run_evaluation(questions, eval_configs)
    answers_dataset = combine_answers(answers, eval_configs, questions)
    answer_evals = evaluate_answers(answers_dataset)
    return answer_evals
