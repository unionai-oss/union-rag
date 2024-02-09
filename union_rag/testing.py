from flytekit import task

@task(container_image="ghcr.io/unionai-oss/union-rag:latest")
def func():
    print("hello")
