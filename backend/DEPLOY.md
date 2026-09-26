# Deploying the Strata API on AWS Lambda

The API's Docker image runs on **AWS Lambda** behind a free **Function URL**, using the
[Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) — no code changes.

**Cost:** Lambda's always-free tier (1M requests + 400,000 GB-seconds a month) covers a
portfolio demo. The only other charge is ECR image storage, about $0.05/month (only the
latest 2 images are kept).

**Trade-off:** the first request after a quiet spell takes a few seconds (cold start).

## Deploy

Needs Docker running and the [AWS CLI v2](https://aws.amazon.com/cli/) (keep it up to date).

```bash
aws configure                  # once: access key, secret, region (e.g. eu-west-2)
cd backend
./aws/deploy.sh
```

The first run creates the ECR repo, IAM role, Lambda function and public URL, then prints:

```
API URL: https://<id>.lambda-url.<region>.on.aws
```

Open `<API URL>/health` — you should see `"ok": true`.

Run `./aws/deploy.sh` again whenever the backend changes.

Optional variables for a run:

| Variable | Effect |
| --- | --- |
| `AWS_REGION` | Region to deploy to (default `eu-west-2`) |
| `OPENAI_API_KEY` | Enables the LLM clinician explanation |
| `CORS_ORIGINS` | Comma-separated allowed sites (default `https://strata.samintech.dev`) |

```bash
OPENAI_API_KEY=sk-... ./aws/deploy.sh
```

## Connect the frontend

Cloudflare Pages project → **Settings → Variables and Secrets** → `VITE_API_BASE_URL` =
the API URL (no trailing slash), then redeploy the frontend.

## Budget alarm (recommended)

AWS Console → **Billing → Budgets → Create budget → Zero spend budget** emails you if
anything ever costs money.

## Remove everything

```bash
aws lambda delete-function --function-name strata-api
aws ecr delete-repository --repository-name strata-api --force
aws iam detach-role-policy --role-name strata-api-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name strata-api-role
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
or from the repo root: `docker compose up --build`.
