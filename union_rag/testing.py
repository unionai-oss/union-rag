from flytekit import task

@task(container_image="ghcr.io/unionai-oss/union-rag:6e031eb")
def func():
    print("hello")
