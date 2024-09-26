import time
from argparse import ArgumentParser
from unionai.remote import UnionRemote


parser = ArgumentParser()
parser.add_argument("--question", type=str, required=True)
args = parser.parse_args()

remote = UnionRemote()
workflow = remote.fetch_workflow(name="union_rag.simple_rag.ask_with_feedback")
execution = remote.execute(workflow, inputs={"question": args.question})
execution = remote.sync(execution, sync_nodes=True)
url = remote.generate_console_url(execution)

print(f"execution: {url}")
print("waiting for answer...")
while True:
    if "n0" in execution.node_executions and execution.node_executions["n0"].is_done:
        value = execution.node_executions["n0"].outputs["o0"]
        break
    execution = remote.sync(execution, sync_nodes=True)

print(f"Answer: {value}")
time.sleep(3)

print("setting signal")
remote.set_signal("get-feedback", execution.id.name, "thumbs-up")
