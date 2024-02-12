from flytekit import task

@task(container_image="ghcr.io/unionai-oss/union-rag:269d17c")
def func():
    print("hello")
