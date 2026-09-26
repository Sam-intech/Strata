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

Optional variables for a run (anything you don't pass keeps its current value):

| Variable | Effect |
| --- | --- |
| `AWS_REGION` | Region to deploy to (default `eu-west-2`) |
| `CORS_ORIGINS` | Comma-separated allowed sites (default `https://strata.samintech.dev`) |
| `LLM_API_KEY` | Key for the LLM that writes the clinician explanation |
| `LLM_BASE_URL` | Any OpenAI-compatible endpoint (blank = OpenAI) |
| `LLM_MODEL` | Model name (default `gpt-4.1-mini`) |
| `OPENAI_API_KEY` | Older name for `LLM_API_KEY` when using OpenAI |

Without any LLM key the API still returns risk, triage and diagnosis — just no explanation.
If the LLM call fails (no credit, rate limit) the same happens instead of an error.

## Clinician explanations with Cloudflare Workers AI (free tier)

1. Cloudflare dashboard → **Account home** → copy your **Account ID** (right sidebar,
   or from any dashboard URL: `dash.cloudflare.com/<ACCOUNT_ID>/...`).
2. **My Profile → API Tokens → Create Token → "Workers AI" template → Create** → copy the token.
3. Deploy with it:

```bash
LLM_API_KEY=<cloudflare-token> \
LLM_BASE_URL=https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1 \
LLM_MODEL=@cf/meta/llama-3.3-70b-instruct-fp8-fast \
./aws/deploy.sh
```

The free allowance (10,000 neurons/day) covers roughly 70 explanations a day with this
model, or several hundred with `@cf/meta/llama-3.1-8b-instruct-fast`.

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
