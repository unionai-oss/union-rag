"""
Union RAG Chat Assistant
"""

import time

import streamlit as st

from unionai.remote import UnionRemote

st.title("Union RAG Chat Assistant")
st.write("Ask me anything about Flyte!")


if "feedback" not in st.session_state:
    st.session_state.feedback = {}


# -----------------
# Utility functions
# -----------------

@st.cache_resource
def get_remote():
    remote = UnionRemote()
    return remote


@st.cache_resource
def get_ask_workflow():
    remote = get_remote()
    workflow = remote.fetch_workflow(name="union_rag.langchain.ask_with_feedback")
    return workflow


def provide_feedback(execution_id, feedback):
    if execution_id in st.session_state.feedback:
        print("feedback already set")
        return
    print(f"feedback: {feedback}")
    remote = get_remote()
    execution = remote.fetch_execution(name=execution_id)
    execution = remote.sync(execution)
    
    st.session_state.feedback[execution_id] = feedback
    remote.set_signal("get-feedback", execution_id, feedback)


def ask(question: str):
    remote = get_remote()
    workflow = get_ask_workflow()
    execution = remote.execute(workflow, inputs={"question": question})
    url = remote.generate_console_url(execution)
    print(f"🚀 Union Serverless execution url: {url}")
    n_retries = 240

    answer = None
    for _ in range(n_retries):
        # gets the answer from the first node, which is the "ask" workflow.
        # the second part of the workflow is the feedback loop.
        if "n0" in execution.node_executions and execution.node_executions["n0"].is_done:
            answer = execution.node_executions["n0"].outputs["o0"]
            break
        execution = remote.sync(execution, sync_nodes=True)
        time.sleep(1)

    if answer is None:
        raise RuntimeError("Failed to get answer")
    
    return answer, execution.id.name

# --------
# Main App
# --------

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    button_disabled = message["execution_id"] in st.session_state.feedback
    button_selected = st.session_state.feedback.get(message["execution_id"], None)

    if message["role"] == "assistant":
        st.button(
            ":thumbsup:",
            key=f"thumbs-up-{i}",
            on_click=provide_feedback,
            args=[message["execution_id"], "thumbs-up"],
            # disabled=button_disabled,
            type="primary" if button_selected == "thumbs-up" else "secondary",
        )
        st.button(
            ":thumbsdown:",
            key=f"thumbs-down-{i}",
            on_click=provide_feedback,
            args=[message["execution_id"], "thumbs-down"],
            # disabled=button_disabled,
            type="primary" if button_selected == "thumbs-down" else "secondary",
        )

# Accept user input
if prompt := st.chat_input("How does Flyte work?"):
    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "execution_id": None}
    )
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner('🤖 Union RAG is thinking...'):
        output, execution_id = ask(prompt)

    with st.chat_message("assistant"):
        response = st.write(output)

    st.button(
        ":thumbsup:",
        key="thumbs-up-latest",
        on_click=provide_feedback,
        args=[execution_id, "thumbs-up"],
    )
    st.button(
        ":thumbsdown:",
        key="thumbs-down-latest",
        on_click=provide_feedback,
        args=[execution_id, "thumbs-down"],
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": output, "execution_id": execution_id}
    )
