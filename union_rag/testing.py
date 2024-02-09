from flytekit import task

@task(container_image="ghcr.io/unionai-oss/union-rag:b86d6c5")
def func():
    print("hello")
