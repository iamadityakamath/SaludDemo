# Salud app

One app for both phases: **Agent Call Reviews** (Phase 1: transcripts, sentiment, AI reviews) and **Payment
Propensity** (Phase 2). A new build: it does not use the old dashboard or the research code in `src/`.

> **Local only.** There is no login and the app shows Salud's real calls, so it runs on your own machine: the backend
> listens on `127.0.0.1:8000`, the frontend on `127.0.0.1:5173`, and the backend only accepts requests from that
> frontend. Every database connection is **read-only** (enforced by the database, and tested). Don't deploy it
> publicly until access protection is decided.

## Structure

```
app/
├── backend/                    FastAPI (Python 3.11+)
│   ├── pyproject.toml
│   ├── salud_app/
│   │   ├── main.py             app, CORS (local frontend only), routes under /api
│   │   ├── config.py           settings: DATABASE_URL (env or repo-root .env), Phase 2 file locations
│   │   ├── db.py               read-only database connections
│   │   ├── schemas.py          response models (the API contract)
│   │   ├── transcripts.py      transcript text -> speaker turns
│   │   ├── repositories/       data access: calls.py (Neon + local sentiment/QA files), phase2.py (master file)
│   │   └── routers/            health.py, calls.py, phase2.py
│   └── tests/                  API tests (no database needed) + a live read-only check (needs DATABASE_URL)
└── frontend/                   React + Vite + TypeScript
    ├── vite.config.ts          dev server on 127.0.0.1:5173, /api proxied to the backend
    └── src/
        ├── api/                client.ts (fetch), types.ts (mirrors schemas.py), useApi.ts
        ├── components/         Layout, badges (source, sentiment)
        └── pages/              Home, CallReviews, CallDetail, Phase2, NotFound
```

## Run it

You need Python 3.11+ with [uv](https://docs.astral.sh/uv/), Node 20+, and the team `DATABASE_URL` in the repository
root `.env` (the same file the rest of the project uses).

```bash
# backend (terminal 1)
cd app/backend
uv venv && uv pip install -e '.[dev]'
.venv/bin/uvicorn salud_app.main:app --reload --host 127.0.0.1 --port 8000

# frontend (terminal 2)
cd app/frontend
npm install
npm run dev            # open http://127.0.0.1:5173
```

Checks: `cd app/backend && .venv/bin/pytest` · `cd app/frontend && npm run typecheck && npm run build`.
API docs while the backend runs: http://127.0.0.1:8000/api/docs.

## API

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | database reachable, Phase 2 master file present |
| GET | `/api/calls?source=real\|synthetic\|pasted&search=&limit=&offset=` | call list: source, date, outcome, caller sentiment at the end, QA score, AI review yes/no, turns |
| GET | `/api/calls/{key}` | one call: transcript turns (with turn sentiment where stored), call-level sentiment, AI review (summary, checklist, coaching), Salud's call record |
| GET | `/api/phase2/summary` | master file overview (aggregates only) and the saved feature sets |

## Where the data comes from

| Data | Source |
|---|---|
| Salud's 84 real calls (records + transcripts) | database schema `final` |
| Phase 1 calls: 10 earlier real, 50 synthetic, 8 pasted (transcripts, sentiment, AI reviews) | database schema `phase1` |
| Sentiment and QA of the 84 real calls | local `Phase 2/reports/sentiment_cache/real.csv`, `Phase 2/reports/qa_real.csv` |
| Phase 2 overview | local `Phase 2/reports/pre_training_data.xlsx`, `Phase 2/reports/experiments/feature_sets_master.json` |

144 calls in total: 8 of the 10 earlier real Phase 1 calls are also in Salud's package and appear once, as the real
call. Turn-level sentiment is shown for the 60 calls whose stored scores belong to the transcript shown (scores are
attached only when speakers and turn counts line up exactly); the 84 real calls have call-level sentiment.

## Not done yet

- The UI design (current pages use placeholder styling).
- Turn-level sentiment for Salud's 84 real calls (needs the sentiment model in the backend).
- Phase 2 model results (they come back from the six task files in `Phase 2/Task/`).
- Access protection, if the app is ever deployed.
- Call details make several database round trips (~2 s); can be batched if it matters.
