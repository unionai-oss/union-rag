"""Evaluate a RAG workflow."""

import os
from dataclasses import dataclass, asdict
from functools import partial
from typing import Annotated, Optional

import pandas as pd
from flytekit import (
    current_context,
    dynamic,
    map_task,
    task,
    workflow,
    Artifact,
    Deck,
    Secret,
    Resources,
)
from flytekit.deck import TopFrameRenderer
from flytekit.types.directory import FlyteDirectory

from union_rag.simple_rag import (
    create_knowledge_base,
    answer_question as _answer_question,
    image as rag_image,
    RAGConfig,
)


EvalDatasetArtifact = Artifact(name="test-eval-dataset")

image = rag_image.with_packages(["nltk", "rouge-score"])


@dataclass
class EvalRAGConfig(RAGConfig):
    condition_name: str = ""


@dataclass
class Question:
    question_id: int
    question: str
    reference_answer: str
    is_user_generated: bool


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
        dataset.loc[dataset.index.repeat(n_answers)][
            ["question_id", "question", "reference_answer", "is_user_generated"]
        ]
        .astype({"question_id": int})
        .to_dict(orient="records")
    )
    return [Question(**record) for record in questions]


@task(
    requests=Resources(cpu="2", mem="4Gi"),
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
    cache=True,
    cache_version="1",
)
def answer_question(
    question: Question,
    search_index: FlyteDirectory,
    prompt_template: str,
) -> Answer:
    return Answer(
        answer=_answer_question.task_function(
            question=question.question,
            search_index=search_index,
            prompt_template=prompt_template,
        ),
        question_id=question.question_id,
    )


@workflow
def batch_rag_pipeline(
    questions: list[Question],
    config: RAGConfig,
) -> list[Answer]:
    search_index = create_knowledge_base(config)
    partial_answer_questions = partial(
        answer_question,
        search_index=search_index,
        prompt_template=config.prompt_template,
    )
    answers = map_task(
        partial_answer_questions,
        concurrency=20,
    )(question=questions)
    return answers


@dynamic(container_image=image, cache=True, cache_version="1")
def run_evaluation(
    questions: list[Question], eval_configs: list[EvalRAGConfig]
) -> list[list[Answer]]:
    answers = []
    for config in eval_configs:
        answers.append(
            batch_rag_pipeline(
                questions=questions,
                config=RAGConfig(
                    prompt_template=config.prompt_template,
                    chunk_size=config.chunk_size,
                    include_union=config.include_union,
                    limit=config.limit,
                    embedding_type=config.embedding_type,
                ),
            )
        )
    return answers


@task(
    container_image=image,
    cache=True,
    cache_version="1",
    enable_deck=True,
)
def combine_answers(
    answers: list[list[Answer]],
    eval_configs: list[EvalRAGConfig],
    questions: list[Question],
) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    # TODO: concatenate all answers into a single dataframe
    combined_answers = []
    for _answers, config in zip(answers, eval_configs):
        for answer, question in zip(_answers, questions):
            assert answer.question_id == question.question_id
            combined_answers.append(
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "answer": answer.answer,
                    "reference_answer": question.reference_answer,
                    "is_user_generated": question.is_user_generated,
                    **asdict(config),
                }
            )

    return pd.DataFrame(combined_answers)


def traditional_nlp_eval(answers_dataset: pd.DataFrame) -> pd.DataFrame:
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer

    bleu_scores, rouge1_scores = [], []
    rouge_scorer = rouge_scorer.RougeScorer(["rouge1"])
    for row in answers_dataset.itertuples():
        bleu_scores.append(
            sentence_bleu([row.reference_answer.split()], row.answer.split())
        )
        _rouge_scores = rouge_scorer.score(row.reference_answer, row.answer)
        rouge1_scores.append(_rouge_scores["rouge1"].fmeasure)

    return answers_dataset.assign(
        bleu_score=bleu_scores,
        rouge1_f1=rouge1_scores,
    )


DEFAULT_EVAL_PROMPT_TEMPLATE = """### Task Description:
You are an expert judge of trivia questions and answers. Given a
question and a reference answer, determine if the candidate answer
is equivalent or better than the reference answer in terms of
correctness.

### Question:
{question}

### Reference Answer:
{reference_answer}

### Candidate Answer:
{candidate_answer}

### Judgement:
Is the candidate answer equivalent or better than the reference answer
in terms of correctness? You MUST answer "Yes" or "No".
"""


def llm_correctness_eval(
    answers_dataset: pd.DataFrame, eval_prompt_template: Optional[str] = None
) -> pd.DataFrame:
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.9)
    prompt = PromptTemplate.from_template(
        eval_prompt_template or DEFAULT_EVAL_PROMPT_TEMPLATE
    )

    llm_correctness_scores = []

    for _, row in answers_dataset.iterrows():
        chain = prompt | model
        result = chain.invoke(
            {
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "candidate_answer": row["answer"],
            }
        )

        result = result.content.lower().strip().strip(".").strip("'").strip('"')
        if result not in ["yes", "no"]:
            score = float("nan")
        elif result == "yes":
            score = 1.0
        else:
            score = 0.0
        llm_correctness_scores.append(score)

    return answers_dataset.assign(llm_correctness_score=llm_correctness_scores)


@task(
    container_image=image,
    enable_deck=True,
    secret_requests=[Secret(key="openai_api_key")],
    cache=True,
    cache_version="2",
)
def evaluate_answers(
    answers_dataset: pd.DataFrame,
    eval_prompt_template: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.environ["OPENAI_API_KEY"] = current_context().secrets.get(key="openai_api_key")

    evaluation = traditional_nlp_eval(answers_dataset)
    evaluation = llm_correctness_eval(evaluation, eval_prompt_template)

    evaluation_summary = (
        evaluation.groupby([*EvalRAGConfig.__dataclass_fields__])[
            ["bleu_score", "rouge1_f1", "llm_correctness_score"]
        ]
        .mean()
        .reset_index()
    )
    current_context().decks.insert(
        0, Deck("Evaluation", TopFrameRenderer(10).to_html(evaluation))
    )
    current_context().decks.insert(
        0, Deck("Evaluation Summary", TopFrameRenderer(10).to_html(evaluation_summary))
    )

    return evaluation, evaluation_summary


# @task(
#     container_image=image,
#     enable_deck=True,
#     cache=True,
#     cache_version="1",
# )
# def evaluation_report(eval_results: pd.DataFrame):
#     eval_results.set_index([*RAGConfig.__dataclass_fields__])
#     current_context().decks.insert(
#         0, Deck("Evaluation", TopFrameRenderer(10).to_html(eval_results))
#     )


@workflow
def evaluate_simple_rag(
    eval_configs: list[EvalRAGConfig],
    eval_dataset: Annotated[
        pd.DataFrame, EvalDatasetArtifact
    ] = EvalDatasetArtifact.query(),
    eval_prompt_template: Optional[str] = None,
    n_answers: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    questions = prepare_questions(eval_dataset, n_answers)
    answers = run_evaluation(questions, eval_configs)
    answers_dataset = combine_answers(answers, eval_configs, questions)
    eval_results = evaluate_answers(answers_dataset, eval_prompt_template)
    # evaluation_report(eval_results)
    return eval_results
