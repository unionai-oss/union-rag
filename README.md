# Union RAG

UnionAI-native RAG applications.

## Setup

```bash
conda create -n union-rag python=3.11 -y
pip install -r requirements.txt
```

Then install [sam cli](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html).

## App Deployment

We're using [bolt](https://slack.dev/bolt-python) to create a slack bot and
`sam cli` to deploy a lambda function that will serve as one of the backend
layers for our question-answering slackbot.

1. Follow the [Bolt getting started](https://slack.dev/bolt-python/tutorial/getting-started)
   guide to create a slack app.
   - Follow the instructions to create a `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET`.
2. Edit the `template.yaml` with the correct values:
   - Update the `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` environment variables.
3. Make sure your `~/.aws/credentials` file is properly configured with your
   `aws_access_key_id` and `aws_secret_access_key`.
4. Login to AWS ECR:

   ```bash
   aws ecr-public get-login-password --profile <PROFILE> --region <REGION> | \
      docker login --username AWS --password-stdin public.ecr.aws/unionai
   ```

4. Run `sam build` to build the app.
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

6. In the slack app you created at `api.slack.com`, go to **slash commands**
   and create a new command.
   - Under Command, enter `/flyte-attendant`
   - Under Request URL, enter the Rest API URL from the previous step with `/slack/events`
     appended to the end. In this example, it would be `https://xyz.execute-api.us-east-2.amazonaws.com/Prod/slack/events`
7. Now test your slack app by installing it in your slack workspace and typing
   `/flyte-attendant`. You should see `Thanks!` as the response to your command.
