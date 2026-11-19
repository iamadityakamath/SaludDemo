# Salud app

One app for both phases:
- **Agent Call Reviews** (Phase 1): the team dashboard (https://agent-call-analysis.vercel.app), rebuilt here with the
  same layout, behaviour and data, plus **Salud's final-package calls** (76 more real calls, their own list group). Pick a call to see the caller's sentiment (KPI tiles and a chart over the call), the
  AI summary, the transcript with a score on every turn, the QA checklist with quoted evidence (click a quote to jump to
  it) and coaching tips. **Upload calls** saves transcripts and analyzes them straight away; **Re-analyze** redoes a call.
- **Payment Propensity** (Phase 2): the training data overview and the selected feature sets; model results to come.

The app is self-contained: it doesn't import the research code in `src/`. The Phase 1 logic (Groq review prompt,
sentiment model and scoring, database writes) was ported from it and checked to give identical results (see below).

> **Local only.** There is no login, so the app runs on your own machine: backend on `127.0.0.1:8000`, frontend on
> `127.0.0.1:5173`; the backend only accepts requests from that frontend. Don't deploy it until access protection is decided.
>
> **It writes to the shared team database**, exactly like the team dashboard: uploaded calls, their sentiment and their
> AI review go into the `phase1` tables, so they also appear on the website. Uploading sends a transcript to Groq
> (redacted text only).

## How a call is analyzed

| Part | How | Where it's saved |
|---|---|---|
| Sentiment | Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest`, int8 ONNX, run locally (downloaded once, 126 MB, to `~/.cache/salud-sentiment`); every caller and agent turn gets a score from −1 to 1; call numbers: start, end, shift, lowest and highest turn | `phase1.turn_sentiment`, `phase1.call_sentiment` |
| AI review | Groq (`GROQ_MODEL`, default `openai/gpt-oss-120b`), prompt version `assist-v2`: 7-item QA checklist with quotes (each quote verified against the transcript), outcome, summary, coaching tips | `phase1.ai_analyses`, `ai_checklist`, `ai_coaching` |

A call is reviewed once; opening it later shows the stored results. Calls missing a part are filled in when opened, and
a background sweep (every 15 s) catches calls added any other way. Re-analyze replaces both parts for that one call.
Reviews are cached on disk in `Phase 1/reports/llm_cache` (the same cache as the team dashboard).

## Structure

```
app/
├── backend/                     FastAPI (Python 3.11+)
│   └── salud_app/
│       ├── main.py              app, CORS (local frontend only), background sweep, routes under /api
│       ├── config.py            settings from the repo-root .env: DATABASE_URL, GROQ_API_KEY, GROQ_MODEL, …
│       ├── db.py                connection pool
│       ├── phase1/              checklist.py, llm.py (Groq client), prompt.py (review prompt + evidence checks),
│       │                        store.py (phase1 tables), analysis.py (fill in / re-analyze / sweep),
│       │                        import_package.py (adds Salud's final-package calls)
│       ├── sentiment/           model.py (Hugging Face ONNX), scoring.py (turn roles, weights, call summary)
│       ├── repositories/        phase2.py (master file overview)
│       └── routers/             health.py, phase1.py, phase2.py
│   └── tests/                   prompt, scoring, analysis and API tests (no database or Groq needed)
└── frontend/                    React + Vite + TypeScript
    └── src/
        ├── styles.css           the team dashboard's stylesheet, unchanged, plus the home / Phase 2 pages
    └── public/salud-logo.png    the Salud logo (header and browser tab; the old dashboard reads it from here too)
        ├── api/                 client.ts, types.ts, useApi.ts
        ├── components/          AppHeader, SentimentCard
        ├── lib/format.ts        score formatting and colours (same rules as the dashboard)
        └── pages/               Home, AgentDashboard, Phase2, NotFound
```

## Run it

You need Python 3.11+ with [uv](https://docs.astral.sh/uv/), Node 20+, and in the repository root `.env`:
`DATABASE_URL` (team database) and, for the AI review, `GROQ_API_KEY` (optional `GROQ_MODEL`). Without a Groq key the
app still works: new calls get sentiment only, and the header says so.

```bash
# backend (terminal 1)
cd app/backend
uv venv && uv pip install -e '.[dev]'
.venv/bin/uvicorn salud_app.main:app --reload --reload-dir salud_app --host 127.0.0.1 --port 8000

# frontend (terminal 2)
cd app/frontend
npm install
npm run dev -- --strictPort      # open http://127.0.0.1:5173
```

Optional settings (`.env`): `SALUD_APP_USER` (your name, shown in the header and saved on uploads; default `aditya`),
`SALUD_AUTO_ANALYZE=false` (turn the background sweep off).

Checks: `cd app/backend && .venv/bin/pytest` · `cd app/frontend && npm run typecheck && npm run build`.
API docs while the backend runs: http://127.0.0.1:8000/api/docs.

## API

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | database reachable, Phase 2 master file present |
| GET | `/api/analyze/list` | every call with its state (done / processing / incomplete), the checklist items and their definitions, whether the AI review is set up |
| GET | `/api/call?id=` | a call's transcript, split into numbered turns |
| GET | `/api/result?id=` | what is stored for a call: sentiment, AI review |
| POST | `/api/analyze/run` `{id, redo}` | fill in what's missing (`redo: false`) or redo both parts (`redo: true`) |
| POST | `/api/analyze/new` `{transcripts: [...]}` | save 1–20 transcripts as new calls and analyze them in the background |
| GET | `/api/phase2/summary` | master file overview (aggregates only) and the saved feature sets |

## How it was checked against the team dashboard

- The old dashboard and the new backend side by side on all 68 calls: transcripts and stored results identical
  (68/68 each); the call list identical except the timestamp format (`…T…Z` instead of `… +00:00`, same instant).
- Ported code on 17 stored calls: turn splitting, the Groq prompt and the sentiment numbers byte-identical.
- Upload → sentiment → Groq review → Re-analyze tested end to end against the Neon **test** branch with an invented
  call (then removed).
- In the browser: same layout as the website; evidence and chart clicks jump to the turn; dialogs; search.

## Salud's final-package calls

Salud's package has 84 transcripts; 8 are the earlier real calls, the other **76** are added as Phase 1 calls with
source `salud` (`salud/<file name>`), shown in their own group, **Salud package calls**, with the call date.
- Migration `db/migrations/010_salud_package_calls.sql` lets `phase1.calls` accept source `salud` (applied to the team
  database and the test branch on 2026-11-19).
- `cd app/backend && .venv/bin/python -m salud_app.phase1.import_package [--dry-run]` adds them, scores their sentiment
  locally and stores their AI reviews **from the local Groq cache only** (the Stage 1 QA run reviewed all 84 with the
  same prompt and model), so it never calls Groq. Safe to run again. Done on the team database: 76 calls, all complete.
- They aren't tagged by raters, so the Phase 1 AI-vs-tags agreement is unchanged. The website doesn't list them (its
  page only knows three groups).

## Differences from the website

- No sign-in (local only); the user name comes from `SALUD_APP_USER`.
- A **Home** link in the header and a home page with the two parts of the project.
- The **Salud package calls** group (above).

## Next: trained models

Planned, not built: ONNX models in Neon object storage, a registry table, and a backend module `salud_app/ml/` that
loads them for the Payment Propensity page. See `Phase 2/progress/model-storage-plan.md`.
