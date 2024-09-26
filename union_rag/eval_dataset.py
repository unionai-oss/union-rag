"""Create an evaluation dataset based on human annotations."""

import pandas as pd
from collections import defaultdict
from itertools import groupby

from flytekit import task, workflow, ImageSpec


image = ImageSpec(packages=["pandas", "pyarrow"])


MOCK_DATASET = [
    {
        "id": 1,
        "question": "Who wrote the Lord of the Rings?",
        "answer_1": "J.R.R. Tolkien",
        "answer_2": "C.S. Lewis",
        "preferred_answer": "J.R.R. Tolkien",
        "user_answer": None,
    },
    {
        "id": 1,
        "question": "Who wrote the Lord of the Rings?",
        "answer_1": "John Ronald Reuel Tolkien",
        "answer_2": "J.R.R. Tolkien",
        "preferred_answer": "J.R.R. Tolkien",
        "user_answer": None,
    },
    {
        "id": 1,
        "question": "Who wrote the Lord of the Rings?",
        "answer_1": "Roald Dahl",
        "answer_2": "J.R.R. Tolkien",
        "preferred_answer": "J.R.R. Tolkien",
        "user_answer": None,
    },
    {
        "id": 1,
        "question": "Who wrote the Lord of the Rings?",
        "answer_1": "John Ronald Reuel Tolkien",
        "answer_2": "C.S. Lewis",
        "preferred_answer": "John Ronald Reuel Tolkien",
        "user_answer": None,
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answer_1": "Egypt",
        "answer_2": "Russia",
        "preferred_answer": None,
        "user_answer": "Northeastern Africa, spanning Egypt to Kenya and Ethiopia",
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answer_1": "Egypt",
        "answer_2": "Russia",
        "preferred_answer": "Egypt",
        "user_answer": None,
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answer_1": "Egypt",
        "answer_2": "Northeastern Africa",
        "preferred_answer": "Northeastern Africa",
        "user_answer": None,
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answer_1": "Sudan",
        "answer_2": "Northeastern Africa",
        "preferred_answer": "Northeastern Africa",
        "user_answer": None,
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answer_1": "Kenya",
        "answer_2": "Northeastern Africa",
        "preferred_answer": "Northeastern Africa",
        "user_answer": None,
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answer_1": "Uganda",
        "answer_2": "Northeastern Africa",
        "preferred_answer": "Northeastern Africa",
        "user_answer": None,
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answer_1": "Uganda",
        "answer_2": "England",
        "preferred_answer": "Uganda",
        "user_answer": None,
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answer_1": "Sudan",
        "answer_2": "England",
        "preferred_answer": "Sudan",
        "user_answer": None,
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answer_1": "Kenya",
        "answer_2": "England",
        "preferred_answer": "Kenya",
        "user_answer": None,
    },
]


def create_comparison_data(annotations: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Create a list of comparisons between annotations.
    """
    comparisons = []
    user_reference_answers = []
    for annotation in annotations:
        if annotation["user_answer"] is not None:
            user_reference_answers.append({
                "id": annotation["id"],
                "question": annotation["question"],
                "reference_answer": annotation["user_answer"],
                "is_user_generated": True,
            })
            continue
        comparisons.append(annotation)
    return comparisons, user_reference_answers


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate the expected score of item A against item B.
    """
    expected_score_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    return expected_score_a


def update_ratings(rating_a: float, rating_b: float, score_a: float, K: int = 32) -> tuple[float, float]:
    """
    Update ratings for item A and item B after a comparison.
    """
    expected_score_a = calculate_expected_score(rating_a, rating_b)
    expected_score_b = 1 - expected_score_a  # Since expected scores sum to 1

    new_rating_a = rating_a + K * (score_a - expected_score_a)
    new_rating_b = rating_b + K * ((1 - score_a) - expected_score_b)

    return new_rating_a, new_rating_b


def calculate_elo_rankings(comparisons: list[dict], K: int = 32) -> tuple[defaultdict, dict]:
    """
    Calculate the ELO rankings for the comparisons.
    
    K-factor, can be adjusted based on the volatility desired
    """
    ratings = defaultdict(float)
    answer_to_question = {}

    # initialize ratings for all answers
    for item in comparisons:
        for answer in [item["answer_1"], item["answer_2"]]:
            if answer not in ratings:
                ratings[answer] = 1500.0
            if answer not in answer_to_question:
                answer_to_question[answer] = (item["id"], item["question"])

    # update the ratings based on the comparisons
    for item in comparisons:
        if item["answer_1"] == item["preferred_answer"]:
            score = 1.0
        elif item["answer_2"] == item["preferred_answer"]:
            score = 0.0
        else:
            score = 0.5

        rating_1, rating_2 = ratings[item["answer_1"]], ratings[item["answer_2"]]
        new_rating_1, new_rating_2 = update_ratings(rating_1, rating_2, score, K=K)
        ratings[item["answer_1"]] = new_rating_1
        ratings[item["answer_2"]] = new_rating_2

    return ratings, answer_to_question


def rank_annotations_per_question(annotations: list[dict]) -> tuple[list[dict], list[dict]]:
    comparisons, user_reference_answers = create_comparison_data(annotations)
    rankings, answer_to_question_id = calculate_elo_rankings(comparisons)

    annotated_reference_answers = []
    raw_rankings = []
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

    # collect raw rankings
    for answer, rating in sorted_rankings:
        question_id, question = answer_to_question_id[answer]
        raw_rankings.append({
            "id": question_id,
            "question": question,
            "answer": answer,
            "elo_rating": rating,
        })

    # collect best answer for this question
    best_answer = sorted_rankings[0][0]
    question_id, question = answer_to_question_id[best_answer]
    annotated_reference_answers.append({
        "id": question_id,
        "question": question,
        "reference_answer": best_answer,
        "is_user_generated": True,
    })

    return [*annotated_reference_answers, *user_reference_answers], raw_rankings
        

@task(container_image=image)
def collect_annotations() -> list[dict]:
    return MOCK_DATASET


@task(container_image=image)
def create_dataset(
    annotations: list[dict],
    min_annotations_per_question: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_annotations = sorted(annotations, key=lambda x: x["id"])

    all_reference_answers = []
    all_raw_rankings = []
    
    for _, annotations_by_question in groupby(sorted_annotations, key=lambda x: x["id"]):
        annotations_by_question = list(annotations_by_question)
        if len(annotations_by_question) <= min_annotations_per_question:
            print(f"Skipping question {annotations_by_question[0]['id']} because it has less than 10 annotations")
            continue
        reference_answers, raw_rankings = rank_annotations_per_question(annotations_by_question)
        all_reference_answers.extend(reference_answers)
        all_raw_rankings.extend(raw_rankings)

    return pd.DataFrame(all_reference_answers), pd.DataFrame(all_raw_rankings)


@workflow
def create_eval_dataset(min_annotations_per_question: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = collect_annotations()
    return create_dataset(annotations, min_annotations_per_question)
