# Instructions for AI coding assistants (Claude Code, ChatGPT / Codex, Cursor, …)

This repo is UIUC BIG Team B's Salud project. A teammate has just cloned it and wants to **tag calls on the
Phase 1 dashboard**. Your job: get the dashboard running and open it for them. Follow these steps in order
and keep your messages short; the user is a teammate, not necessarily a developer.

**First, check whether they need local setup at all:** the dashboard is also a website,
https://agent-call-analysis.vercel.app (team passcode, then their name). If they only want to tag calls,
point them there and stop. Continue below only if they want it running locally.

## 1. Ask the user for two things (before running anything)

1. **Their first name**, lower case. It must be one of: `avni`, `sharayu`, `aditya`, `ibrahim`, `ananya`,
   `rudrakshi`. Their calls are already assigned under that name.
2. **The team database connection string** (`DATABASE_URL`), which the team lead shared privately.
   It starts with `postgresql://` and its host contains `-pooler`.

**Secret handling:** write the connection string only into `.env` (gitignored). Never print it back, never
put it in any other file, a commit, a command line argument, or a URL. Never commit `.env`. The repo is public.

## 2. Install

Requires Python 3.11 or newer. Check with `python3 --version`.

macOS / Linux, with [uv](https://docs.astral.sh/uv/) (preferred):

```bash
uv venv && uv pip install -e '.[dev]'
```

Without uv:

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

Windows (PowerShell): `py -3.11 -m venv .venv` then `.venv\Scripts\pip install -e ".[dev]"`, and use
`.venv\Scripts\salud-phase1` wherever this file says `.venv/bin/salud-phase1`.

## 3. Create `.env`

```bash
cp .env.example .env
```

Then set the `DATABASE_URL=` line in `.env` to the string the user gave you. Leave every other line as it
is. Teammates don't need `GROQ_API_KEY`, `DATABASE_URL_UNPOOLED` or `DATABASE_URL_TEST`.

## 4. Check the connection, then start the dashboard

```bash
.venv/bin/salud-phase1 assign
```

This only reads. It should print a table of raters with their assigned and tagged counts. If it does, start
the dashboard **in the background** (it keeps running until stopped), replacing `<name>` with the user's
first name:

```bash
.venv/bin/salud-phase1 dashboard --rater <name>
```

It prints `Open http://127.0.0.1:8765/` and opens the browser. If no browser opens (for example you run in a
sandbox), tell the user to open http://127.0.0.1:8765/ themselves. If your tool has a built-in browser, you
may open that URL in it.

## 5. Tell the user how to use it (keep it to these points)

- The list on the left opens on **"My calls"**: the calls assigned to them (11–12 each).
- Each call is **pre-filled with the AI's answers**. Check each item against the transcript, change any
  answer that's wrong, pick sentiment and outcome, then **Save & next** (⌘S / Ctrl+S).
- **Red** section = the AI gave no supporting quote; decide it yourself. It turns green once answered.
- **Yes** = the agent did this item correctly, **No** = they didn't, **N/A** = the item doesn't apply
  to this call (e.g. no balance discussed).
- Saving goes straight to the shared database. They can stop any time and continue later; to stop the
  dashboard, press Ctrl+C in its terminal (or stop the background process).

## Troubleshooting

| Message | Fix |
|---|---|
| `No database configured` | `DATABASE_URL` is missing from `.env`, or `.env` isn't in the repo root. |
| Connection or password errors | The string was pasted incompletely (often cut at `?sslmode=require`). Ask the user to paste it again. |
| `Port 8765 is already in use` | An earlier dashboard is still running. Stop it, or add `--port 8766` and use that port in the URL. |
| `no calls are assigned to '<name>'` | Name typo. Use one of the six names above. |
| `command not found: salud-phase1` | Install step didn't finish, or you're not in the repo root. Rerun step 2. |
| Python older than 3.11 | Install 3.11+ (`uv python install 3.11`, or from python.org), then redo step 2. |

## Don't, unless the user explicitly asks

- **Don't run `salud-db setup`, `salud-phase1 assign --raters …` or `salud-phase1 analyze`.** They change
  the shared team database (the first two are for the team lead) or send real calls to an external LLM.
- **Don't write to the database** with SQL or scripts, and don't delete or edit anyone's tags.
- **Don't read or print transcripts, tags or AI analyses** into the chat. The database holds Salud's real
  (redacted) calls; only the dashboard should show them.
- Don't change code, commit or push. This setup needs no code changes.

## For context only

- `README.md` explains the whole project: Phase 1 (LLM call analysis vs manual tags) and Phase 2
  (synthetic data for payment propensity).
- Tests: `.venv/bin/pytest`. Database tests need `DATABASE_URL_TEST`, which teammates don't have; those
  tests are skipped without it.
