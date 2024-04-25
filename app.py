import json
import logging
import re
import time

from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler


N_RETRIES = 180

# process_before_response must be True when running on FaaS
app = App(process_before_response=True)


def ack_answer_question(body, ack, say):
    ack()
    event = body["event"]
    thread_ts = event.get("thread_ts", None) or event["ts"]
    if event.get("text") is None:
        say(f"Please ask a question.", thread_ts=thread_ts)
    else:
        say(f"Processing your question, one moment...", thread_ts=thread_ts)


def ack_get_feedback(ack):
    ack()


def answer_question(body, say):
    from unionai.remote import UnionRemote

    event = body["event"]
    text = re.sub("<@.+>", "", event["text"]).strip()

    remote = UnionRemote()
    workflow = remote.fetch_workflow(name="union_rag.langchain.ask_with_feedback")
    execution = remote.execute(workflow, inputs={"question": text})
    execution = remote.sync(execution, sync_nodes=True)

    answer = None
    for _ in range(N_RETRIES):
        if "n0" in execution.node_executions and execution.node_executions["n0"].is_done:
            answer = execution.node_executions["n0"].outputs["o0"]
            break
        execution = remote.sync(execution, sync_nodes=True)
        time.sleep(1)

    if answer is None:
        raise RuntimeError("Failed to get answer")

    response = answer.replace("@flyte-attendant", "")
    thread_ts = event.get("thread_ts", None) or event["ts"]
    say(
        text=response,
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": response,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Was this answer helpful?",
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "emoji": True,
                            "text": "👍"
                        },
                        "value": f"thumbs_up:{execution.id.name}",
                        "action_id": "feedback_thumbs_up",
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "emoji": True,
                            "text": "👎"
                        },
                        "value": f"thumbs_down:{execution.id.name}",
                        "action_id": "feedback_thumbs_down",
                    }
                ]
            }
        ],
        thread_ts=thread_ts,
        unfurl_links=False,
        unfurl_media=False,
    )


def get_feedback(body, respond, ack):
    ack()

    from unionai.remote import UnionRemote

    remote = UnionRemote()
    action = body["actions"][0]
    label, execution_id = action["value"].split(":")

    execution = remote.fetch_execution(name=execution_id)
    execution = remote.sync(execution)
    if execution.is_done:
        print("feedback already set")
        return

    print("feedback_body", json.dumps(body, indent=4))
    text = body["message"]["text"]

    # update blocks so that the clicked button appears to be selected
    blocks = body["message"]["blocks"]
    for block in blocks:
        if block["type"] == "actions":
            for element in block["elements"]:
                if element["action_id"] == action["action_id"]:
                    element["style"] = "primary"
                elif "style" in element:
                    element.pop("style")

    remote.set_signal("get-feedback", execution_id, label)

    respond(
        replace_original=True,
        text=text,
        blocks=blocks,
    )


app.event("app_mention")(   
    ack=ack_answer_question,
    lazy=[answer_question],
)

app.action(re.compile("^feedback_thumbs_.+$"))(
    ack=ack_get_feedback,
    lazy=[get_feedback],
)


SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


def handler(event, context):
    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
