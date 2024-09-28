# RAG Evaluation Workflows

First create a raw knowledge base:

```bash
union run --remote union_rag/simple_rag.py get_documents --limit 100
```

Then synthesize a question and answer dataset:

```bash
union run --remote union_rag/synthesize_data.py data_synthesis_workflow \
   --n_questions 1 \
   --n_answers 5
```

Register the data annotation workflow:

```bash
union register union_rag/annotate_data.py
```

Run a single annotation session to test it out:

```bash
union run --remote union_rag/annotate_data.py create_annotation_set --random_seed 42 --n_samples 10
```

## Annotator App

Create a `secrets.txt` file to store these credentials. This file is ignored by
git and should look something like this:

```
UNIONAI_SERVERLESS_API_KEY=<UNIONAI_SERVERLESS_API_KEY>
```

Export the secrets to your environment:

```bash
export $(cat secrets.txt | xargs)
```

Run the app

```bash
streamlit run streamlit/annotation_app.py
```

Create the eval dataset:

```bash
union run --remote union_rag/eval_dataset.py create_eval_dataset --min_annotations_per_question 1
```

Evaluate a RAG experiment:

```bash
union run --remote union_rag/eval_rag.py evaluate_simple_rag --eval_configs eval_inputs.yaml
```
