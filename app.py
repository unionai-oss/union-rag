import logging
import re

from dataclasses import dataclass
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

from unionai.remote import UnionRemote


@dataclass
class Message:
    """
    message type: https://api.slack.com/events/message
    """
    type: str
    subtype: str
    text: str
    ts: str
    user: str


# process_before_response must be True when running on FaaS
app = App(process_before_response=True)


@app.event("app_mention")
def handle_app_mentions(body, say, logger):
    logger.info(body)
    say("What's up?")


@app.command("/flyte-attendant")
def flyte_attendant_command(ack):
    ack("Thanks!")


@app.message("@flyte-attendant")
def answer_question_union(message, say):
    message = Message(**message)
    remote = UnionRemote()
    task = remote.get_task("union_rag.langchain.ask")
    question = message.text.replace("@flyte-attendant", "")
    execution = remote.execute(task, inputs={"question": question}, wait=True)
    response, _ = [*execution.outputs.values()]
    say(response)


@app.message(re.compile("^\/flyte-attendant-lambda"))
def answer_question_lambda(message, say):
    message = Message(**message)
    remote = UnionRemote()
    # TODO: get latest execution of this task
    task = remote.get_task("union_rag.langchain.create_search_index")

    # call langchain.answer_question code here
    response = ...

    say(response)


SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


def handler(event, context):
    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
