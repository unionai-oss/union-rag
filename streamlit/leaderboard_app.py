import random
import streamlit as st
import json
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# File to store user data
USER_DATA_FILE = "user_data.json"

# Initialize session state
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_level" not in st.session_state:
    st.session_state.user_level = 0

# Load user data from file
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}


# Save user data to file
def save_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f)


# Update user's annotation count
def update_user_annotations(username, count=1):
    user_data = load_user_data()
    if username in user_data:
        user_data[username] += count
    else:
        user_data[username] = count
    save_user_data(user_data)
    return user_data[username]


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


# Main annotation page
def annotation_page():
    st.title("Union Synthetic Evaluation Dataset")

    # Username input
    username = st.text_input("Enter your username:", value=st.session_state.username)
    if username:
        st.session_state.username = username

    if not username:
        st.warning("Please enter a username to start annotating.")
        return

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
                "London London London London London London London London London London London London London London London London London London London London London London London London London London London London London London London London London London London London ",
                "Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris ",
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

    question_container = st.container(border=True)
    answer_columns = st.columns([4, 4, 1])

    with question_container:
        st.write("**Question**")
        st.write(data_point["question"])

    answers = data_point["answers"]

    label = None

    def create_button(label, content, key):
        return st.button(
            f"**{label}**\n\n{content}",
            key=key,
            use_container_width=True
        )

    for i, (column, answer) in enumerate(zip(answer_columns[:2], answers), 1):
        with column:
            st.markdown(f"**Answer {i}**")
            if create_button("", answer, f"button-{i}"):
                label = f"answer_{i}"

    with answer_columns[2]:
        st.markdown("&nbsp;")  # Add space to align with answer boxes
        if create_button("Neither are correct", "", "button-neither"):
            label = "neither"

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
    # When submitting an annotation:
    if submitted:
        new_count = update_user_annotations(username)
        new_achievement, new_level = get_achievement(new_count)
        if new_level > st.session_state.user_level:
            st.session_state.user_level = new_level
            send_slack_notification(username, new_level)
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


# Leaderboard page
def leaderboard_page():
    st.title("Leaderboard")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_data = load_user_data()

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


# Main app
def main():
    st.set_page_config(layout="wide")

    # Tabs for navigation
    tab1, tab2 = st.tabs(["Annotation", "Leaderboard"])

    with tab1:
        annotation_page()
    with tab2:
        leaderboard_page()


if __name__ == "__main__":
    main()