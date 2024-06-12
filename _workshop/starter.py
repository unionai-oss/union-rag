"""Simple workflow to make sure the connection to remote works."""
from flytekit import task, workflow


@task
def hello() -> str:
    return "world"


@workflow
def main() -> str:
    return hello()
