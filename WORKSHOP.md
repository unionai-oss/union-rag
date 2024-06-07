# LLM RAG Workshop

## Setup

Sign up for a Union Serverless account here: https://signup.union.ai/?group=workshop

## Part 1: Running the RAG Workflow with Union Serverless

Create the knowledge base artifact:

```bash
unionai run --copy-all --remote union_rag/langchain.py create_knowledge_base \
    --exclude_patterns '["/api/", "/_tags/"]' \
    --limit 100
```

Then run the question to ask questions to the RAG model:

```bash
unionai run --copy-all --remote union_rag/langchain.py ask --question 'what is flytekit?'
```

We can even run a workflow where we can provide feedback:

```bash
unionai run --copy-all --remote union_rag/langchain.py ask_with_feedback --question 'what is flytekit?'
```

## Part 2: Creating a Chat UI with Streamlit

Now we're going to interact with this workflow through a chat interface through
streamlit.


## Bonus: Running a Local LLM with Ollama and Ngrok

Download Ollama: https://ollama.com/download

Then, sign up for an account on ngrok: https://ngrok.com/. Follow the setup
instructions in https://dashboard.ngrok.com/get-started/setup.

Start the ollama server:

```
ollama serve
```

Create the ngrok endpoint with the following command:

```bash
ngrok http 11434 --host-header="localhost:11434"
```

Finally, in the `union_rag/langchain.py`, update the `ChatOpenAI` arguments with
the following:

   ```python
   ChatOpenAI(
       model_name="llama3",
       temperature=0.9,
       base_url="https://<ngrok_endpoint>/v1",
       api_key="ollama",
   )
   ```

Rerun the workflow:

```bash
unionai run --copy-all --remote union_rag/langchain.py ask \
    --question 'what is flytekit?' \
    --exclude_patterns '["/api/", "/_tags/"]'
```

If you look at the terminal session that's running the ollama server, you should
see something like:

```
[GIN] 2024/06/07 - 10:47:54 | 200 |  5.198009084s |  35.193.158.216 | POST     "/v1/chat/completions"
```

This corresponds to the API call that the workflow made when generating its response!