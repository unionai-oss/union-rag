from flytekit import task

@task(container_image="ghcr.io/unionai-oss/union-rag:96a822a")
def func():
    print("hello")
