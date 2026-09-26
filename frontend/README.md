# Strata frontend

React 19 + TypeScript + Vite + Tailwind CSS 4 web app for the Strata clinical
decision-support API. See the [root README](../README.md) for the full project.

## Run locally

Requires Node.js 22 (see `.nvmrc`).

```bash
npm ci
npm run dev        # http://localhost:5173
```

The app sends the patient form to `POST {VITE_API_BASE_URL}/infer`. If
`VITE_API_BASE_URL` is unset, it uses `http://127.0.0.1:8000`, the local backend.
To override it, create `.env.local`:

```dotenv
VITE_API_BASE_URL=https://your-api.example.com
```

## Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Vite dev server with hot reload |
| `npm run build` | Type-check (`tsc -b`) and build to `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | ESLint |

## Structure

```
src/
├── App.tsx                    # Layout, API call, loading/error state
├── main.tsx                   # React entry point
├── index.css                  # Tailwind + shared component classes
└── components/
    ├── header.tsx
    ├── patientforms.tsx       # Patient + lab input form
    └── resultspanel.tsx       # Risk, triage, lab plan, diagnosis, clinician report
```

## Deployment

Cloudflare Pages builds this folder with `npm run build` and serves `dist/`. Set
`VITE_API_BASE_URL` in the Pages project under **Settings → Variables and
Secrets**, then redeploy.
