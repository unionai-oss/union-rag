"""
Eval Dataset Annotation App.

The purpose of this app is to annotate a dataset of questions and anwswers
about Flyte and Union. The task is to select the more factually correct
answer from a choice of two answers.
"""

import json
import random
import time
from typing import Optional

import redis
import streamlit as st
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from flytekit import Labels
from flytekit.tools.translator import Options
from union.remote import UnionRemote


redis_client = redis.Redis(
    host="eminent-moccasin-23904.upstash.io",
    port=6379,
    password=st.secrets["REDIS_KEY"],
    ssl=True,
)

N_SAMPLES = 5
ANSWER_FORMAT = {
    "answer_1": "Answer 1",
    "answer_2": "Answer 2",
    "tie": "It's a tie",
    "neither": "Neither are correct",
    "question_incoherent": "The question doesn't make sense",
}
APP_VERSION = "testing0"
LEADERBOARD_VERSION = st.secrets.get("LEADERBOARD_VERSION", "testing")


st.set_page_config(
    page_title="Helpabot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
session_id = st.runtime.scriptrunner.get_script_run_ctx().session_id


# Initialize session state
if "username" not in st.session_state:
    st.session_state.username = ""
if "passcode" not in st.session_state:
    st.session_state.passcode = ""
if "annotation_data" not in st.session_state:
    st.session_state.annotation_data = None
if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "execution_id" not in st.session_state:
    st.session_state.execution_id = None
if "execution_url" not in st.session_state:
    st.session_state.execution_url = None


@st.cache_data(show_spinner=False)
def get_annotation_data(username: str, session_id: str) -> tuple[list[dict], str]:
    """
    Gets a new set of question and answer-pair triplets on every page load.
    """
    st.write("✨ Creating serverless execution... ")
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
            labels=Labels(values={"union_annotator": "APP_VERSION"}),
        ),
    )
    url = remote.generate_console_url(execution)
    st.write(f"🚀 [Union Serverless execution]({url})")

    n_retries = 240
    annotation_data = None
    st.write("⏳ Waiting for annotation payload...")
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


def refresh_session():
    get_annotation_data.clear()
    st.session_state.annotations = {}
    st.session_state.execution_id = None
    st.session_state.current_question_index = 0
    st.rerun()


@st.cache_data
def format_func(answer: str) -> str:
    return ANSWER_FORMAT[answer]


def get_user_annotation_count(username):
    """Get the annotation count for a specific user."""
    return int(
        redis_client.get(f"{LEADERBOARD_VERSION}:user_annotations:{username}") or 0
    )  # Default to 0 if the user does not exist


def get_all_users():
    """Retrieve all users and their annotation counts."""
    # Use Redis keys pattern to find all user keys
    user_keys = redis_client.keys("{LEADERBOARD_VERSION}:user_annotations:*")
    user_data = {}

    for key in user_keys:
        username = key.decode("utf-8").split(":")[
            1
        ]  # Extract the username from the key
        count = get_user_annotation_count(username)
        user_data[username] = count

    return user_data


# Update user's annotation count
def update_user_annotations(username, count=1):
    # Create a unique key for each user based on their username
    user_key = f"{LEADERBOARD_VERSION}:user_annotations:{username}"

    # Increment the annotation count for the specified user
    new_count = redis_client.incrby(
        user_key, count
    )  # Use incrby to increment the value
    return new_count


# Get achievement based on annotation count
def get_achievement(count):
    if count >= 200:
        return "🏆", 4
    elif count >= 100:
        return "🥇", 3
    elif count >= 50:
        return "🥈", 2
    elif count >= 25:
        return "🥉", 1
    elif count >= 5:
        return "🌟", 0
    else:
        return "◼️", -1


# Send Slack notification
def send_slack_notification(username, level):
    slack_token = st.secrets["SLACK_API_TOKEN"]
    client = WebClient(token=slack_token)
    channel = st.secrets["SLACK_CHANNEL"]

    messages = [
        f"🌟 Whoa! {username} just started their annotation journey! They're officially a Novice Annotator. Every expert was once a beginner, right?",
        f"🥉 Bronze brilliance! {username} leveled up to Bronze Annotator! They're on fire... well, more like a warm glow, but still impressive!",
        f"🥈 Silver success! {username} just reached Silver Annotator status! They're shining brighter than a full moon on a clear night!",
        f"🥇 Gold glory! {username} has ascended to Gold Annotator! They're practically radiating awesomeness at this point!",
        f"🏆 Bow down to the new Annotation Master! {username} has reached the pinnacle of annotation greatness! We're not worthy! We're not worthy!",
    ]

    try:
        client.chat_postMessage(channel=channel, text=messages[level])
    except SlackApiError as e:
        st.error(f"Error sending message to Slack: {e}")


##########
# Main app
##########


def annotation_page(username: Optional[str], passcode_correct: bool):
    if not (username and passcode_correct):
        st.write(
            "Start a new session by entering a username and the secret passcode in the sidebar 👈."
        )
        return

    curr_user_annotation_count = get_user_annotation_count(username)
    _, curr_level = get_achievement(curr_user_annotation_count)

    if st.session_state.execution_id is None:
        with st.status("Starting a new annotation session...", expanded=True) as status:
            annotation_data, execution_id, execution_url = get_annotation_data(
                username, session_id
            )
            status.update(label="Session created", state="complete", expanded=False)
            st.session_state.execution_id = execution_id
            st.session_state.annotation_data = annotation_data
            st.session_state.execution_url = execution_url
            st.rerun()

    annotation_data = st.session_state.annotation_data
    execution_id = st.session_state.execution_id
    execution_url = st.session_state.execution_url

    st.write("#### Instructions:")
    st.write(
        "Below is a question about Flyte or Union and two answers to the question."
    )

    if len(st.session_state.annotations) == len(annotation_data):
        st.write("🎉 You've completed this annotation task!")

        new_session = st.button("Start new session")
        if new_session:
            refresh_session()

        return

    st.write(f"Annotation session: [{execution_id}]({execution_url})")
    data_point = annotation_data[st.session_state.current_question_index]

    percent_complete = len(st.session_state.annotations) / len(annotation_data)
    st.progress(percent_complete, f"Percent complete: {percent_complete * 100:.0f}%")

    answers = data_point["answers"]

    with st.container(border=True):
        question_column, answer_column = st.columns(2)
        with question_column:
            st.write("**Question**")
            st.write(data_point["question"])

            with answer_column:
                c = st.container(border=True)
                c.write("**Answer 1**")
                c.write(answers[0])

                c = st.container(border=True)
                c.write("**Answer 2**")
                c.write(answers[1])

    label = st.radio(
        "Select the better answer based on factual accuracy.",
        options=ANSWER_FORMAT.keys(),
        index=None,
        format_func=format_func,
        key=f"radio-{data_point['id']}",
    )

    correct_answer_text = None

    def submit_answer(
        correct_answer_text, submit_key, submit_with_dialog: bool = False
    ):
        submitted = st.button("Submit", disabled=label is None, key=submit_key)
        if submitted:
            st.session_state.annotations[data_point["id"]] = {
                "question_id": data_point["id"],
                "question": data_point["question"],
                "answers": answers,
                "label": label,
                "correct_answer_text": correct_answer_text or None,
            }

            if len(annotation_data) - len(st.session_state.annotations) > 0:
                st.session_state.current_question_index += 1
            else:

                def submitting(msg: str):
                    with st.spinner(msg):
                        submit_annotations(st.session_state.annotations, execution_id)
                        new_count = update_user_annotations(username, count=N_SAMPLES)
                        new_achievement, new_level = get_achievement(new_count)
                        if new_level > curr_level:
                            send_slack_notification(username, new_level)

                if submit_with_dialog:
                    st.dialog("Submitting annotations...")(submitting)("🗂️ ⬆️ ☁️")
                else:
                    submitting("Submitting annotations 🗂️ ⬆️ ☁️")

            st.rerun()

    if label == "neither":

        @st.dialog("(Optional) Submit your own answer.", width="large")
        def submit_correct_answer():
            st.write(f"**Question**: {data_point['question']}")
            correct_answer_text = st.text_area(
                "If you're confident that you know the correct answer, enter it below. "
                "If not leave it blank and click 'Submit' to continue.",
                key=f"text-area-{data_point['id']}",
                height=200,
            )
            submit_answer(correct_answer_text, submit_key="dialog-submit-answer")

        submit_answer(correct_answer_text, submit_key="submit-answer")
        submit_correct_answer()
    else:
        submit_answer(correct_answer_text, submit_key="submit-answer")


def leaderboard_page():
    st.title("Leaderboard")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_data = get_all_users()
        if len(user_data) == 0:
            st.write("No one has annotated any data yet.")

        else:
            # Sort users by annotation count
            sorted_users = sorted(user_data.items(), key=lambda x: x[1], reverse=True)

            # Display leaderboard
            for rank, (user, count) in enumerate(sorted_users, 1):
                achievement, curr_level = get_achievement(count)
                if curr_level >= 0:
                    st.markdown(
                        f"{rank}. **{achievement} {user}**: {count} annotations"
                    )

    with col2:
        with st.container(border=True):
            st.markdown("#### Achievement Legend")
            st.write("🌟 Novice Annotator: 5-24 annotations")
            st.write("🥉 Bronze Annotator: 25-49 annotations")
            st.write("🥈 Silver Annotator: 50-99 annotations")
            st.write("🥇 Gold Annotator: 100-199 annotations")
            st.write("🏆 Annotation Master: 200+ annotations")


def main():
    with st.sidebar:
        st.title("🤝🤖 Helpabot.")
        st.write("Help a bot out by selecting factually correct answers.")
        username = st.text_input(
            "Enter a username for the leaderboard:",
            value=st.session_state.username,
        )
        passcode = st.text_input(
            "Enter the secret passcode to start a new session:",
            value=st.session_state.passcode,
            type="password",
        )

        if username:
            st.session_state.username = username

        passcode_correct = False
        if passcode:
            if passcode == st.secrets["SECRET_PASSCODE"]:
                st.session_state.passcode = passcode
                passcode_correct = True

        if username and passcode_correct:
            st.warning(
                "Refreshing the page will start a new session and your progress will be lost."
            )
        else:
            if passcode and not passcode_correct:
                st.warning("Incorrect passcode. Please try again.")
            if not (username and passcode):
                st.info(
                    "Please enter a username and the secret passcode to start a session."
                )

    tab1, tab2 = st.tabs(["Annotation", "Leaderboard"])

    with tab1:
        annotation_page(username, passcode_correct)
    with tab2:
        leaderboard_page()


if __name__ == "__main__":
    main()
