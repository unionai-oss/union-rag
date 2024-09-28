"""
Eval Dataset Annotation App.

The purpose of this app is to annotate a dataset of questions and anwswers
about Flyte and Union. The task is to select the more factually correct
answer from a choice of two answers.
"""

import json
import random
import time

import redis
import streamlit as st
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from flytekit import Labels
from flytekit.tools.translator import Options
from union.remote import UnionRemote

redis_client = redis.Redis(
  host='eminent-moccasin-23904.upstash.io',
  port=6379,
  password=st.secrets['REDIS_KEY'],
  ssl=True
)

N_SAMPLES = 10
ANSWER_FORMAT = {
    "answer_1": "Answer 1",
    "answer_2": "Answer 2",
    "neither": "Neither are correct",
}


st.set_page_config(layout="wide")


# Initialize session state
if "current_question_id" not in st.session_state:
    st.session_state.current_question_id = None
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_level" not in st.session_state:
    st.session_state.user_level = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "email" not in st.session_state:
    st.session_state.email = ""
if "execution_id" not in st.session_state:
    st.session_state.execution_id = None



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
    return int(redis_client.get(f"user_annotations:{username}") or 0)  # Default to 0 if the user does not exist


def get_all_users():
    """Retrieve all users and their annotation counts."""
    # Use Redis keys pattern to find all user keys
    user_keys = redis_client.keys("user_annotations:*")
    user_data = {}

    for key in user_keys:
        username = key.decode("utf-8").split(":")[1]  # Extract the username from the key
        count = get_user_annotation_count(username)
        user_data[username] = count

    return user_data

# Update user's annotation count
def update_user_annotations(username, count=1):
    # Create a unique key for each user based on their username
    user_key = f"user_annotations:{username}"

    # Increment the annotation count for the specified user
    new_count = redis_client.incrby(user_key, count)  # Use incrby to increment the value
    return new_count

# Get achievement based on annotation count
def get_achievement(count):
    if count >= 100:
        return "🏆", 4
    elif count >= 50:
        return "🥇", 3
    elif count >= 20:
        return "🥈", 2
    elif count >= 5:
        return "🥉", 1
    else:
        return "🌟", 0

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
        f"🏆 Bow down to the new Annotation Master! {username} has reached the pinnacle of annotation greatness! We're not worthy! We're not worthy!"
    ]

    try:
        response = client.chat_postMessage(
            channel=channel,
            text=messages[level]
        )
    except SlackApiError as e:
        st.error(f"Error sending message to Slack: {e}")


##########
# Main app
##########

# with st.sidebar:
#     st.write("Enter your email to enter the leaderboard")
#     email = st.text_input("Email", value=st.session_state.email)
#     if email:
#         st.session_state.email = email
#     placeholder = st.empty()


def annotation_page():
    st.title("🤖 Helpabot.")
    st.write(
        "*Help a bot out by picking the more factually correct "
        "answer, i.e. the answer with fewer mistakes.* "
    )

    # Username input
    username = st.text_input("Enter your username:", value=st.session_state.username)
    if username:
        st.session_state.username = username

    if not username:
        st.warning("Please enter a username to start annotating.")
        return
    
    with st.spinner("Starting a new annotation session..."):
        annotation_data, execution_id, url = get_annotation_data()
        st.session_state.execution_id = execution_id

    st.write("Below is a question about Flyte or Union and two answers to the question.")

    question_ids = [q["id"] for q in annotation_data]
    if len(st.session_state.annotations) == len(annotation_data):
        st.write("You've answered all the questions!")
        
        new_session = st.button("Start new session")
        if new_session:
            refresh_session()

        st.stop()

    data_point = annotation_data[st.session_state.current_question_index]

    question_container = st.container(border=True)
    answer_columns = st.columns([4, 4, 1])

    answer_1_column, answer_2_column = st.columns(2)

    with question_container:
        st.write("**Question**")
        st.write(data_point["question"])

    answers = data_point["answers"]
    label = st.radio(
        "Select the more factually correct answer.",
        options=ANSWER_FORMAT.keys(),
        index=None,
        format_func=format_func,
        key=f"radio-{data_point['id']}",
    )

    with answer_1_column:
        c = st.container(border=True)
        c.write("**Answer 1**")
        c.write(answers[0])

    with answer_2_column:
        c = st.container(border=True)
        c.write("Answer 2")
        c.write(answers[1])

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
        new_count = update_user_annotations(username)
        new_achievement, new_level = get_achievement(new_count)
        if new_level > st.session_state.user_level:
            st.session_state.user_level = new_level
            send_slack_notification(username, new_level)
        unanswered = [q for q in question_ids if q not in st.session_state.question_ids_answered]
        if len(unanswered) > 0:
            st.session_state.current_question_id = random.choice(unanswered)
        # TODO: send feedback back to the serverless execution via UnionRemote
        st.rerun()

def leaderboard_page():
    st.title("Leaderboard")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_data = get_all_users()

        # Sort users by annotation count
        sorted_users = sorted(user_data.items(), key=lambda x: x[1], reverse=True)

        # Display leaderboard
        for rank, (user, count) in enumerate(sorted_users, 1):
            achievement, _ = get_achievement(count)
            st.markdown(f"{rank}. **{achievement} {user}**: {count} annotations")

    with col2:
        st.subheader("Achievement Legend")
        st.write("🌟 Novice Annotator: 0-4 annotations")
        st.write("🥉 Bronze Annotator: 5-19 annotations")
        st.write("🥈 Silver Annotator: 20-49 annotations")
        st.write("🥇 Gold Annotator: 50-99 annotations")
        st.write("🏆 Annotation Master: 100+ annotations")

def main():
    tab1, tab2 = st.tabs(["Annotation", "Leaderboard"])

    with tab1:
        annotation_page()
    with tab2:
        leaderboard_page()

if __name__ == "__main__":
    main()
