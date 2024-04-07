import logging
import re

from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

from unionai.remote import UnionRemote


# process_before_response must be True when running on FaaS
app = App(process_before_response=True)


def respond_to_slack_within_3_seconds(body, say):
    print(body)
    event = body["event"]
    thread_ts = event.get("thread_ts", None) or event["ts"]
    if event.get("text") is None:
        say(f"Please provide a query.", thread_ts=thread_ts)
    else:
        say(f"Processing your query, one moment...", thread_ts=thread_ts)


def answer_question(body, say):
    remote = UnionRemote()
    task = remote.fetch_workflow(name="union_rag.langchain.ask")
    event = body["event"]
    text = re.sub("<@.+>", "", event["text"]).strip()
    execution = remote.execute(task, inputs={"question": text}, wait=True)
    response = execution.outputs["o0"].replace("@flyte-attendant", "")
    thread_ts = event.get("thread_ts", None) or event["ts"]
    say(response, thread_ts=thread_ts)


app.event("app_mention")(   
    ack=respond_to_slack_within_3_seconds,
    lazy=[answer_question],
)

# TODO:
# - add thumbs up/thumbs down button to the response (https://slack.dev/bolt-python/concepts#message-sending)
# - create a workflow that takes the thumbs up/thumbs down signal and sends it back to Union
# - link the thumbs up/thumbs down button to send the signal to the gate node


app.command("/flyte-attendant")(
    ack=respond_to_slack_within_3_seconds,
    lazy=[answer_question],
)


SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


def handler(event, context):
    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
