#!/usr/bin/env bash
# Build the API image, push it to ECR and create/update the Lambda function.
# First run creates everything and prints the public URL; later runs just update the code.
#
#   cd backend && ./aws/deploy.sh
#
# Needs: Docker running, AWS CLI v2 logged in (`aws configure`).
# Optional env: AWS_REGION (default eu-west-2), CORS_ORIGINS, and the LLM used for
# the clinician explanation: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL (or OPENAI_API_KEY).
# Settings you don't pass keep their current value on the function.
set -euo pipefail

REGION="${AWS_REGION:-eu-west-2}"
NAME="strata-api"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REGISTRY/$NAME:latest"
export AWS_REGION="$REGION" AWS_PAGER=""

cd "$(dirname "$0")/.."

echo "==> ECR repository"
aws ecr describe-repositories --repository-names "$NAME" >/dev/null 2>&1 ||
  aws ecr create-repository --repository-name "$NAME" >/dev/null
# Keep only the latest 2 images so storage stays tiny.
aws ecr put-lifecycle-policy --repository-name "$NAME" --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":2},"action":{"type":"expire"}}]}' >/dev/null

echo "==> Build + push image (arm64)"
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
# --provenance=false: Lambda only accepts a single-platform image manifest.
docker buildx build --platform linux/arm64 --provenance=false -t "$IMAGE" --push .

ENV_KEYS=(CORS_ORIGINS LLM_API_KEY LLM_BASE_URL LLM_MODEL OPENAI_API_KEY)
FUNCTION_EXISTS=false
aws lambda get-function --function-name "$NAME" >/dev/null 2>&1 && FUNCTION_EXISTS=true

# Build the environment: value passed now > value already on the function > default.
build_env() {
  local json="" key val
  for key in "${ENV_KEYS[@]}"; do
    val="${!key:-}"
    if [[ -z "$val" && "$FUNCTION_EXISTS" == true ]]; then
      val="$(aws lambda get-function-configuration --function-name "$NAME" \
        --query "Environment.Variables.$key" --output text 2>/dev/null || true)"
      [[ "$val" == "None" ]] && val=""
    fi
    [[ -z "$val" && "$key" == CORS_ORIGINS ]] && val="https://strata.samintech.dev"
    [[ -n "$val" ]] && json+="${json:+,}\"$key\":\"$val\""
  done
  printf '{"Variables":{%s}}' "$json"
}
ENV_VARS="$(build_env)"

if [[ "$FUNCTION_EXISTS" == true ]]; then
  echo "==> Update function"
  aws lambda update-function-code --function-name "$NAME" --image-uri "$IMAGE" >/dev/null
  aws lambda wait function-updated --function-name "$NAME"
  aws lambda update-function-configuration --function-name "$NAME" --environment "$ENV_VARS" >/dev/null
  aws lambda wait function-updated --function-name "$NAME"
else
  echo "==> Execution role"
  ROLE="$NAME-role"
  if ! ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text 2>/dev/null)"; then
    ROLE_ARN="$(aws iam create-role --role-name "$ROLE" --query Role.Arn --output text \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}')"
    aws iam attach-role-policy --role-name "$ROLE" \
      --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    sleep 10 # let IAM propagate
  fi

  echo "==> Create function"
  aws lambda create-function --function-name "$NAME" --package-type Image \
    --code ImageUri="$IMAGE" --role "$ROLE_ARN" --architectures arm64 \
    --memory-size 2048 --timeout 60 --environment "$ENV_VARS" >/dev/null
  aws lambda wait function-active-v2 --function-name "$NAME"

  echo "==> Public URL"
  # CORS is handled by FastAPI, so the URL itself has none configured.
  aws lambda create-function-url-config --function-name "$NAME" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$NAME" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$NAME" --statement-id public-invoke \
    --action lambda:InvokeFunction --principal '*' --invoked-via-function-url >/dev/null 2>&1 || true
fi

URL="$(aws lambda get-function-url-config --function-name "$NAME" --query FunctionUrl --output text)"
echo
echo "API URL: ${URL%/}"
echo "Check:   ${URL%/}/health"
