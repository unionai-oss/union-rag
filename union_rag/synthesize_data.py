import json
import json
import random
from datetime import timedelta
from typing import List, Annotated, Dict

from flytekit import task, workflow, Resources, Artifact, wait_for_input
from flytekit.core.artifact import Inputs
from flytekit.types.file import FlyteFile

from union_rag.simple_rag import image

DEFAULT_ANNOTATION_SET_NAME = "default"

QuestionAndAnswerDataset = Artifact(name="question_and_answer_dataset")
AnnotatedDataset = Artifact(name="annotated_dataset", partition_keys=["name"])

@task(
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
    cache=True,
    cache_version="1",
)
def sample_triple(random_seed: int, n_samples: int, dataset: FlyteFile) -> List[Dict]:
    dataset.download()
    with open(dataset.path, 'r') as f:
        qa_triples_list = json.load(f)

    assert len(
        qa_triples_list) >= n_samples, f"Not enough samples in the dataset. Required: {n_samples}, Available: {len(qa_triples_list)}"

    random.seed(random_seed)
    sampled_qa = random.sample(qa_triples_list, k=n_samples)

    return sampled_qa

@task(
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
)
def append_annotated_data(feedback: str, master_annotation_dataset: FlyteFile) -> FlyteFile:
    parsed_annotation_json = json.loads(feedback)

    master_annotation_dataset.download()
    with open(master_annotation_dataset.path, 'r') as file:
        master_annotation_dataset_list = json.load(file)

    master_annotation_dataset_list.extend(parsed_annotation_json)

    print(master_annotation_dataset_list)

    file_path = 'master_annotation_dataset.json'
    with open(file_path, 'w') as f:
        json.dump(master_annotation_dataset_list, f, indent=4)

    return FlyteFile(path=file_path)

@workflow
def create_annotation_set(random_seed: int, n_samples: int = 10, dataset: FlyteFile = QuestionAndAnswerDataset.query(), annotation_set_name: str = DEFAULT_ANNOTATION_SET_NAME, master_annotation_dataset: FlyteFile = AnnotatedDataset.query(name=Inputs.annotation_set_name)) -> Annotated[FlyteFile, AnnotatedDataset(name=Inputs.annotation_set_name)]:
    triple_sample = sample_triple(random_seed=random_seed, n_samples=n_samples, dataset=dataset)
    feedback = wait_for_input("feedback", timeout=timedelta(hours=1), expected_type=str)
    triple_sample >> feedback
    annotated_dataset = append_annotated_data(feedback=feedback, master_annotation_dataset=master_annotation_dataset)
    return annotated_dataset


@task(container_image=image)
def init_annotation_set() -> FlyteFile:
    file_path = 'master_annotation_dataset.json'
    with open(file_path, 'w') as f:
        json.dump([], f, indent=4)

    return FlyteFile(path=file_path)

@workflow
def init_annotation_set_wf(annotation_set_name: str = DEFAULT_ANNOTATION_SET_NAME) -> Annotated[FlyteFile, AnnotatedDataset(name=Inputs.annotation_set_name)]:
    return init_annotation_set()