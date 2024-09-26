"""
Eval Dataset Annotation App.

The purpose of this app is to annotate a dataset of questions and anwswers
about Flyte and Union. The task is to select the more factually correct
answer from a choice of two answers.
"""

import random
import streamlit as st

st.set_page_config(layout="wide")
st.title("Union Synthetic Evaluation Dataset")
st.write("Below is a question about Flyte or Union and two answers to the question.")

if "question_ids_answered" not in st.session_state:
    st.session_state.question_ids_answered = []

if "current_question_id" not in st.session_state:
    st.session_state.current_question_id = None


# TODO: get this dataset from the serverless execution via UnionRemote
SYNTHETIC_DATASET = [
    {
        "id": 1,
        "question": "What is the capital of the United States?",
        "answers": [
            "London",
            "Paris",
        ]
    },
    {
        "id": 2,
        "question": "Where is the Nile River located?",
        "answers": [
            "Egypt",
            "Russia",
        ]
    },
    {
        "id": 3,
        "question": "Who wrote the Lord of the Rings?",
        "answers": [
            "J.R.R. Tolkien",
            "C.S. Lewis",
        ]
    },
]

question_ids = [q["id"] for q in SYNTHETIC_DATASET]
question_ids_unanswered = [
    q["id"] for q in SYNTHETIC_DATASET
    if q["id"] not in st.session_state.question_ids_answered
]
if question_ids_unanswered == []:
    st.write("You've answered all the questions!")
    # TODO: start a new labeling session
    st.stop()

if st.session_state.current_question_id is None:
    st.session_state.current_question_id = random.choice(question_ids_unanswered)

data_point = [q for q in SYNTHETIC_DATASET if q["id"] == st.session_state.current_question_id]
assert len(data_point) == 1
data_point = data_point[0]

ANSWER_FORMAT = {
    "answer_1": "Answer 1",
    "answer_2": "Answer 2",
    "neither": "Neither are correct",
}

@st.cache_data
def format_func(answer: str) -> str:
    return ANSWER_FORMAT[answer]


question_container = st.container(border=True)
answer_1_column, answer_2_column = st.columns(2)


with question_container:
    st.write("**Question**")
    st.write(data_point["question"])

answers = data_point["answers"]

with answer_1_column:
    c = st.container(border=True)
    c.write("**Answer 1**")
    c.write(answers[0])

with answer_2_column:
    c = st.container(border=True)
    c.write("Answer 2")
    c.write(answers[1])


label = st.radio(
    "Select the more factually correct answer.",
    options=ANSWER_FORMAT.keys(),
    index=None,
    format_func=format_func,
    key=f"radio-{data_point['id']}",
)

correct_answer_text = None
if label == "neither":
    st.write("You said neither of the answers are correct.")

    text_area_col, text_preview_col = st.columns(2)

    with text_area_col:
        correct_answer_text = st.text_area(
            "If you're confident that you know the correct answer, enter it below. "
            "Be as specific and concise as possible.",
            key=f"text-area-{data_point['id']}",
            height=200,
        )

    with text_preview_col:
        st.write("**Preview**")
        st.markdown(correct_answer_text)

submitted = st.button("Submit", disabled=label is None)

if submitted:
    st.write("Thank you for your submission!")
    st.session_state.question_ids_answered.append(data_point["id"])
    st.write(f"Submission: {label, correct_answer_text}")
    unanswered = [q for q in question_ids if q not in st.session_state.question_ids_answered]
    if len(unanswered) > 0:
        st.session_state.current_question_id = random.choice(unanswered)
        st.write(st.session_state.question_ids_answered)
        st.write(st.session_state.current_question_id)
    # TODO: send feedback back to the serverless execution via UnionRemote
    st.rerun()
