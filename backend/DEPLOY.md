# Deploying the Strata API on Cloudflare

The API runs as a **Cloudflare Container** (your `Dockerfile`), fronted by a tiny
Worker (`worker/index.ts`) that forwards requests to it.

> Cloudflare Containers require the **Workers Paid** plan ($5/month).

## One-time setup

```bash
cd backend
npm install
npx wrangler login
# Optional — enables the LLM clinician explanation. Without it the API still
# returns the risk score, triage and diagnosis.
npx wrangler secret put OPENAI_API_KEY
```

## Deploy

Docker must be running locally (wrangler builds the image and pushes it to Cloudflare).

```bash
npx wrangler deploy
```

Wrangler prints the URL, e.g. `https://strata-api.<your-subdomain>.workers.dev`.
Check it: open `<that URL>/health` — you should see `"ok": true`.
The first request after a quiet period takes a little longer while the container starts.

## Connect the frontend

In the Cloudflare Pages project for the frontend → **Settings → Variables and Secrets**,
add `VITE_API_BASE_URL` = the Worker URL above (no trailing slash), then redeploy
the frontend.

## Allowing more sites

Edit `CORS_ORIGINS` in `wrangler.jsonc` (comma-separated), then `npx wrangler deploy`.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
