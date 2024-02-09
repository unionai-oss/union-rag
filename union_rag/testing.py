from flytekit import task

@task(container_image="ghcr.io/unionai-oss/union-rag:7162291")
def func():
    print("hello")
