"""
Eval Dataset Annotation App.

The purpose of this app is to annotate a dataset of questions and anwswers
about Flyte and Union. The task is to select the more factually correct
answer from a choice of two answers.
"""

import json
import random
import time

import streamlit as st
from flytekit import Labels
from flytekit.tools.translator import Options
from union.remote import UnionRemote


st.set_page_config(layout="wide")
st.title("🤖 Helpabot.")
st.write(
    "*Help a bot out by picking the more factually correct "
    "answer, i.e. the answer with fewer mistakes.* "
)


N_SAMPLES = 10


@st.cache_data(show_spinner=False)
def get_annotation_data() -> tuple[list[dict], str]:
    """
    Gets a new set of question and answer-pair triplets on every page load.
    """
    remote = UnionRemote()
    seed = random.randint(0, 1000000)
    workflow = remote.fetch_workflow(
        name="union_rag.annotate_data.create_annotation_set"
    )
    execution = remote.execute(
        workflow,
        inputs={"random_seed": seed, "n_samples": N_SAMPLES},
        options=Options(
            # Replace ':' with '_' since flyte does not allow ':' in the label value
            labels=Labels(values={"app": "union_annotator"}),
        ),
    )
    url = remote.generate_console_url(execution)
    print(f"🚀 Union Serverless execution url: {url}")

    n_retries = 240
    annotation_data = None
    for _ in range(n_retries):
        # gets the answer from the first node, which is the "ask" workflow.
        # the second part of the workflow is the feedback loop.
        if (
            "n0" in execution.node_executions
            and execution.node_executions["n0"].is_done
        ):
            annotation_data = execution.node_executions["n0"].outputs["o0"]
            break
        execution = remote.sync(execution, sync_nodes=True)
        time.sleep(1)

    assert annotation_data is not None
    random.shuffle(annotation_data)
    return annotation_data, execution.id.name, url


def submit_annotations(annotations: dict, execution_id: str):
    remote = UnionRemote()
    execution = remote.fetch_execution(name=execution_id)
    execution = remote.sync(execution)
    remote.set_signal("feedback", execution_id, json.dumps(annotations))
    st.session_state.execution_id = None
    print(f"🚀 Submitted annotations to Union Serverless execution: {execution_id}")


if "annotations" not in st.session_state:
    st.session_state.annotations = {}

if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0

if "email" not in st.session_state:
    st.session_state.email = ""

if "execution_id" not in st.session_state:
    st.session_state.execution_id = None


def refresh_session():
    get_annotation_data.clear()
    st.session_state.annotations = {}
    st.session_state.execution_id = None
    st.session_state.current_question_index = 0
    st.rerun()


##########
# Main app
##########

with st.sidebar:
    st.write("Enter your email to enter the leaderboard")
    email = st.text_input("Email", value=st.session_state.email)
    if email:
        st.session_state.email = email
    placeholder = st.empty()


with st.spinner("Starting a new annotation session..."):
    annotation_data, execution_id, url = get_annotation_data()
    st.session_state.execution_id = execution_id

# if st.session_state.execution_id is None:
#     with st.spinner("Starting a new annotation session..."):
#         annotation_data, execution_id, url = get_annotation_data(n_samples)
#         st.session_state.execution_id = execution_id
# else:
#     new_session = st.button("Start new session")
#     if new_session:
#         refresh_session()
#         annotation_data, execution_id, url = get_annotation_data(n_samples)
#         st.session_state.execution_id = execution_id

with placeholder.container():
    st.write(f"Annotation session: [{execution_id}]({url})")

# When all questions are answered, offer to start a new session
question_ids = [q["id"] for q in annotation_data]
st.write(f"Annotations: {len(st.session_state.annotations)}")
if len(st.session_state.annotations) == len(annotation_data):
    st.write("🎉 You've answered all the questions!")

    new_session = st.button("Start new session")
    if new_session:
        refresh_session()

    st.stop()

st.write("## Instructions:")
st.write("Below is a question about Flyte or Union and two answers to the question.")
st.warning("If you refresh the page, your progress will be lost.")


data_point = [annotation_data[st.session_state.current_question_index]]
data_point = data_point[0]

ANSWER_FORMAT = {
    "answer_1": "Answer 1",
    "answer_2": "Answer 2",
    "equivalent": "They are equivalent",
    "neither": "Neither are correct",
}


percent_complete = len(st.session_state.annotations) / len(annotation_data)
st.progress(percent_complete, f"Percent complete: {percent_complete * 100:.0f}%")


@st.cache_data
def format_func(answer: str) -> str:
    return ANSWER_FORMAT[answer]


######################
# QUESTION ANSWER FORM
######################

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
    c.write("**Answer 2**")
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
            "**(Optional)** *If you're confident that you know the correct answer, enter it below. "
            "Be as specific and concise as possible.*",
            key=f"text-area-{data_point['id']}",
            height=200,
        )

    with text_preview_col:
        st.write("**Preview**")
        st.markdown(correct_answer_text)

submitted = st.button("Submit", disabled=label is None)

if submitted:
    st.write("✅ Thank you for your submission!")
    preferred_answer = (
        answers[0]
        if label == "answer_1"
        else answers[1]
        if label == "answer_2"
        else None
    )
    st.session_state.annotations[data_point["id"]] = {
        "question_id": data_point["id"],
        "question": data_point["question"],
        "preferred_answer": preferred_answer,
        "label": label,
        "correct_answer_text": correct_answer_text,
    }
    if len(annotation_data) - len(st.session_state.annotations) > 0:
        st.session_state.current_question_index += 1
    else:
        with st.spinner("Submitting annotations..."):
            submit_annotations(st.session_state.annotations, execution_id)
    st.rerun()
