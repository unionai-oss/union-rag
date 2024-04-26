# Union RAG

UnionAI-native RAG applications.

## Setup

```bash
conda create -n union-rag python=3.11 -y
pip install -r requirements.txt
```

Then install [sam cli](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html).

## Run the Union RAG Workflow

For quick iteration and development, you can run the workflow on Union with:

```bash
unionai run --remote union_rag/langchain.py ask --question 'what is flytekit?'
```

## Workflow Deployment

Deploy the workflow to Union with the following command:

```bash
unionai register union_rag/langchain.py
```

## App Deployment

We're using [bolt](https://slack.dev/bolt-python) to create a slack bot and
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
