# Deploying the Strata API (free)

The API runs as a free **Hugging Face Docker Space** (2 vCPU, 16 GB RAM, no card needed).
A GitHub Action pushes `backend/` to the Space whenever it changes on `main`.

> Free Spaces sleep after ~48 h without traffic; the next request wakes it (takes ~1 min).

## One-time setup

1. Sign up at <https://huggingface.co> (free).
2. **New Space** (<https://huggingface.co/new-space>):
   - Name: `strata-api` · SDK: **Docker** → *Blank* · Hardware: **CPU basic (free)** · Public.
3. **Access token**: <https://huggingface.co/settings/tokens> → *Create new token* → type **Write** → copy it.
4. In GitHub → this repo → **Settings → Secrets and variables → Actions**:
   - *Secrets* tab → `HF_TOKEN` = the token from step 3.
   - *Variables* tab → `HF_SPACE` = `<your-hf-username>/strata-api`.
5. Optional — LLM clinician explanations: in the Space → **Settings → Variables and secrets**,
   add secret `OPENAI_API_KEY`. Without it the API still returns risk, triage and diagnosis.
6. GitHub → **Actions → Deploy backend to Hugging Face → Run workflow**.

The Space builds in a few minutes. Your API URL is
`https://<your-hf-username>-strata-api.hf.space` — check `<that URL>/health` shows `"ok": true`.

## Connect the frontend

Cloudflare Pages project → **Settings → Variables and Secrets** → `VITE_API_BASE_URL` =
the API URL above (no trailing slash), then redeploy the frontend.

## Allowing more sites

Set `CORS_ORIGINS` (comma-separated) as a Space variable. Default: `https://strata.samintech.dev`.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
or from the repo root: `docker compose up --build`.
