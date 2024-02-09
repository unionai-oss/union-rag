from flytekit import task

@task(container_image="ghcr.io/unionai-oss/union-rag:e7fd0cb")
def func():
    print("hello")
