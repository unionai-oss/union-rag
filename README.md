# Union RAG

UnionAI-native RAG applications.

> [!NOTE]
> If you're here for the LLM RAG workshop, go [here](./WORKSHOP.md).

## Setup

```bash
conda create -n union-rag python=3.11 -y
pip install -r requirements.txt
```

## Create the Knowledge Base

First create the knowledge base offline:

```bash
union run --remote union_rag/simple_rag.py create_knowledge_base \
   --config '{"exclude_patterns": ["/api/", "/_tags/"], "embedding_type": "openai", "limit": 100}'
```

## Run the Simple RAG Workflow

For quick iteration and development, you can run the workflow on Union with:

```bash
union run --remote union_rag/simple_rag.py ask --question 'what is flytekit?'
```

## Simple RAG Workflow Deployment

Deploy the workflow to Union with the following command:

```bash
union register union_rag/simple_rag.py
```

## Run the Agentic RAG Workflow

Run the agentic RAG workflow with a more complex question:

```bash
union run --remote union_rag/agentic_rag.py ask \
   --question 'Write a flytekit workflow that trains an sklearn model on the wine dataset.'
```

Run the same workflow with feedback:

```bash
union run --remote union_rag/agentic_rag.py ask_with_feedback \
   --question 'Write a flytekit workflow that trains an sklearn model on the wine dataset.'
```

## Slack App Deployment

<details>
<summary>Deployment instructions</summary>
<br>

Install [sam cli](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html).

We'll use [bolt](https://slack.dev/bolt-python) to create a slack bot and
`sam cli` to deploy a lambda function that will serve as one of the backend
layers for our question-answering slackbot.

1. Follow the [Bolt getting started](https://slack.dev/bolt-python/tutorial/getting-started)
   guide to create a slack app.
   - Follow the instructions to create a `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET`.
   - Create a `UNIONAI_SERVERLESS_API_KEY` using `unionai create app union-rag`
2. Create a `secrets.txt` file to store these credentials. This file is ignored by
   git and should look something like this:

   ```
   SLACK_BOT_TOKEN=<SLACK_BOT_TOKEN>
   SLACK_SIGNING_SECRET=<SLACK_SIGNING_SECRET>
   UNIONAI_SERVERLESS_API_KEY=<UNIONAI_SERVERLESS_API_KEY>
   ```

3. Export the secrets to your environment:

   ```bash
   export $(cat secrets.txt | xargs)
   ```

4. Create the `deploy.yaml` file:

   ```
   cat template.yaml | envsubst > deploy.yaml
   ```

5. Make sure your `~/.aws/credentials` file is properly configured with your
   `aws_access_key_id` and `aws_secret_access_key`.
6. Login to AWS ECR. First do `aws configure sso`, then:

   ```bash
   aws sso login --profile <PROFILE>
   ```

7. Run `sam build --template deploy.yaml` to build the app.
5. Run `sam deploy --guided` to deploy the app to AWS. This will ask you a
   series of questions on specific values you want to use for the deployment.
   In the end you should see an output like this:

   ```
   CloudFormation outputs from deployed stack
   -----------------------------------------------------------------------------------------------------------------
   Outputs
   -----------------------------------------------------------------------------------------------------------------
   Key                 UnionRagApi
   Description         API Gateway endpoint URL for Prod stage for union rag function
   Value               https://xyz.execute-api.us-east-2.amazonaws.com/Prod/

   Key                 UnionRagFunctionIamRole
   Description         Implicit IAM Role created for union rag function
   Value               arn:aws:iam::xyz:role/union-rag-UnionRagFunctionRole-xyz

   Key                 UnionRagFunction
   Description         union rag Lambda Function ARN
   Value               arn:aws:lambda:us-east-2:xyz:function:union-rag-UnionRagFunction-xyz
   -----------------------------------------------------------------------------------------------------------------

   Successfully created/updated stack - union-rag in us-east-2
   ```

6. Now test your slack app by installing it in your slack workspace and typing
   `@flyte-attendant what is flytekit?`. You should see an initial response
   from the bot, followed by the answer to your question.

</details>
