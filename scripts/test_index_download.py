from unionai.remote import UnionRemote


remote = UnionRemote()
task = remote.fetch_task(name="union_rag.langchain.create_search_index")
execution = remote.fetch_execution(name="f59dba4aefe7344e1841")
execution = remote.sync_execution(execution, sync_nodes=True)

# get the search index
node_execution = execution.node_executions["n1"]
assert node_execution

search_index = node_execution.outputs["o0"]
with remote.remote_context():
    search_index.download()
