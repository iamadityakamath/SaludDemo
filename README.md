# Salud Call QA & Payment Propensity Pipeline

UIUC BIG Team B · Fall 2026 · Client: Salud Revenue Partners

One connected pipeline: **call transcript → QA / sentiment scores → payment propensity score (0–1) → ranked "call first" list**. Project scope, issue tree, benchmarking and roadmap live in `docs/`.

| Phase | Question | Method | Data |
|---|---|---|---|
| **1. Call analysis** | Did the agent handle each checklist item? What was the caller's sentiment, and how did the call end? | LLM (Groq `openai/gpt-oss-120b`), checked against manual tags | 10 real calls + 50 LLM-written synthetic calls |
| **2. Propensity** | Which accounts are most likely to pay? | Traditional ML on account features + Phase 1's call-quality score | Synthetic accounts with payment histories |

Phase 1's per-call export (`reports/phase1/comparison/calls.csv`) becomes Phase 2's call-quality input. The brief requires the two stages to connect rather than be separate projects.

## Setup

```bash
uv venv && uv pip install -e '.[dev]'
cp .env.example .env        # then set GROQ_API_KEY=...
.venv/bin/pytest
```

Client files keep the names Salud sent them under and are **gitignored**: `docs/`, `data/Redacted Transcripts …/` and `data/BIG_Schema_Only_Handoff/`. So are generated data (`data/phase1/`, `data/phase2/`) and outputs (`reports/`).

---

## Phase 1: call analysis

The same LLM prompt ([`phase1/prompt.py`](src/salud_pipeline/phase1/prompt.py)) powers both routes.

### Route 1: dashboard (assists manual tagging)

```bash
.venv/bin/salud-phase1 dashboard --rater AK            # Analyze button can pre-fill the checklist
.venv/bin/salud-phase1 dashboard --rater AK --no-llm   # tag with the AI switched off
```

It opens http://127.0.0.1:8765 and listens only on this machine. It lists the 10 real calls and 50 synthetic calls.

- **Filters:** by source, by status, and by call-ID search.
- **Analyze** (top right, or ⌘/Ctrl+Enter) runs the LLM and returns:
  - checklist answers, with reasons and verified transcript quotes
  - caller sentiment and call outcome
  - a summary and coaching tips
- **Flags:** a section is **red** when the AI's answer has no supporting quote, and turns **green** once you choose the answer yourself.
- **Where tags are saved:** `data/phase1/manual_tags/real_calls.csv` and `synthetic_calls.csv`, one row per call per rater.
  - `llm_model` is blank when you tagged without pressing Analyze. These are the fairest benchmark, because pre-filled tags tend to follow the AI.
  - `overrides` lists the items you changed from the AI's answer.

### Route 2: script (AI analysis, then comparison)

```bash
.venv/bin/salud-phase1 analyze             # LLM on all 60 calls -> reports/phase1/ai_analyses.json
.venv/bin/salud-phase1 analyze --limit 20  # cap new LLM requests this run
.venv/bin/salud-phase1 compare             # AI vs manual tags -> reports/phase1/comparison/
```

- **`analyze`** saves every call's analysis in **one JSON file keyed by call ID** (e.g. `real/0fd3661b-….txt`), with the model and prompt version.
  - It resumes: calls already in the file are skipped.
  - It stops cleanly when Groq's daily cap is hit; run it again later to continue.
  - Calls already analyzed on the dashboard come from the shared LLM cache (`reports/phase1/llm_cache/`) at no cost.
- **`compare`** reads that JSON and the manual tags, and writes `phase1_results.xlsx` plus CSVs and `summary.md`:

| Sheet | Contents |
|---|---|
| Summary | counts and average AI–human agreement: all tagged calls, real, synthetic, tagged with AI off |
| Calls | one row per call: AI answers, reasons, evidence, needs-review flags, human answers, agree/disagree, AI and human call-quality scores, summary |
| Coaching | one row per AI coaching tip |
| AI vs human | per item: agreement with 95% CI, Cohen's kappa, missing AI answers |
| Rater agreement | per rater pair and item |
| Synthetic check | human vs planted labels and AI vs planted labels |
| Definitions | checklist items, outcomes, metrics |

**Comparison rules:**
- **N/A is an answer.** A human N/A vs. an AI Yes is a disagreement.
- **A missing AI answer counts as a disagreement.**
- **Two raters who split on an item give no consensus,** and that item is left out of the AI comparison for that call.
- **Call-quality score:** Yes / (Yes + No) over the 7 transcript-visible items, with N/A excluded. It's provisional until Salud sets weights.

**Scope:** Salud's checklist has 9 items. "Correct call statusing" and "Grammar in notes" need the CRM disposition and agent note, so Phase 1 scores the other 7.

### The Phase 1 calls

- **Real:** the 10 redacted Salud calls.
- **Synthetic:** 50 calls written by Qwen (a different model family from the scorer) from planted labels. They're listed in [`configs/phase1_synthetic_calls.txt`](configs/phase1_synthetic_calls.txt), which is tracked so every rater tags the same calls.
  - They were picked from 168 to cover every outcome and scenario, failures of every checklist item, and some all-pass calls.
  - Transcripts and planted labels are in `data/phase1/synthetic_calls/`.

### LLM, Groq and real calls

- **Model:** `GROQ_MODEL` in `.env` (default `openai/gpt-oss-120b`), or any OpenAI-compatible server through `SALUD_LLM_BASE_URL` / `SALUD_LLM_MODEL` / `SALUD_LLM_API_KEY`.
- **Editing the prompt:** change [`prompt.py`](src/salud_pipeline/phase1/prompt.py) and the definitions in [`checklist.py`](src/salud_pipeline/phase1/checklist.py), then bump `PROMPT_VERSION`. `analyze` will then redo every call.
- **Groq free-tier limits on this key:**
  - **200,000 tokens per day** for gpt-oss-120b. One analysis is ≈3,200 tokens, so **about 60 analyses a day**, shared by the dashboard and the script.
  - 8,000 tokens per minute per model.
- **Real calls:** Groq is a third-party cloud service, and the brief requires real call data to stay in a HIPAA-compliant environment. The calls are redacted, but confirm with Salud before sending them. The dashboard asks once per browser session; `analyze` says so when it starts.

### Why the LLM and not ML

On 60 synthetic test calls, the LLM agreed with the planted labels at checklist kappa **0.66**, against **0.15–0.38** for TF-IDF + logistic regression. Salud has about 10 real calls, far too few to train a classifier. See [`results/phase1_llm_vs_ml_2026-10-18/`](results/phase1_llm_vs_ml_2026-10-18/README.md). The ML code remains in git history (commit `487a6a7`).

---

## Phase 2: synthetic data for propensity

```bash
.venv/bin/salud-synth --config configs/synthetic_world.json   # -> data/phase2/synthetic_world
```

A 2,000-guarantor run takes about 3 seconds and produces roughly 3.2k accounts, 11k contacts, 4.1k connected calls and 1.4k QA reviews, all validated.

### Output

```
data/phase2/synthetic_world/
  crm/            6 tables shaped exactly like BIG_Synthetic_Schema.sql (the only valid model inputs)
  truth/          planted latents and outcomes (for evaluation only, never use as features)
  transcripts/    one template transcript per connected call
  synthetic.sqlite    every table, for SQL access
  manifest.json       config, counts, summary stats, validation results
```

### How the planted truth works

1. **Agents** have a skill level that sets their chance of passing each of the 9 QA checklist items.
2. **Guarantors** have hidden traits: ability to pay, willingness, reachability and hardship.
3. Each **connected call** draws true QA item results from its agent. Call quality, together with the patient's traits, drives sentiment and the call outcome.
   - If Correct Call Statusing is No, the CRM records the wrong result; if Grammar is No, the agent note is garbled.
4. **History before the cutoff:** statement-driven payments, promise-to-pay follow-through, payment plans, financial-assistance adjustments and insurance payments.
5. **Outcome window** (cutoff 2026-04-01, 90 days): each open account pays with probability `sigmoid(OutcomeModel · features)`.
   - `configs/synthetic_world_null_call_quality.json` sets the call-quality effect to 0, to check that the pipeline doesn't find signal where none was planted. Indirect effects remain, as in real data.

### Traps that are deliberate (and realistic)

- **The worklist is current state** as of the day after the outcome window, so balance and payment fields reveal the outcome. Build features only from dated rows on or before the cutoff.
- **Contacts and accounts are many-to-many** through `salud_Account_X_ContactHistory`. Aggregate each to account level before joining.
- **Some receipts aren't self-pay cash.** Use `phase2.outcome_labels.self_pay_payments`.

### Conventions (open items in the handoff)

- Timestamps are naive America/Chicago. Money is USD; receipt amounts are positive.
- `ActionsTaken` is a `;`-joined list of option codes.
- The payment horizon (90 days) and the minimum amount ($0.01) are working assumptions until Salud confirms them.

---

## Layout

```
src/salud_pipeline/
  paths.py              every project location
  llm_client.py         OpenAI-compatible client (.env, Groq defaults, retries, max_tokens)
  phase1/
    calls.py            the 60 Phase 1 calls and where their tags are saved
    checklist.py        checklist items, outcomes, definitions
    prompt.py           the LLM prompt, response checking, LLM cache
    dashboard.py        route 1: dashboard server   (static/dashboard.html: the page)
    analyze.py          route 2, step 1: LLM on every call -> ai_analyses.json
    compare.py          route 2, step 2: AI vs manual tags -> Excel / CSV
    metrics.py          agreement, kappa, rater consensus
    cli.py              salud-phase1 dashboard | analyze | compare
  phase2/
    schema.py           tables and option codes from Salud's schema handoff
    outcome_labels.py   self-pay outcome labels at account grain
    synthetic/          synthetic world generator (salud-synth); also wrote the Phase 1 calls
configs/                phase1_synthetic_calls.txt, synthetic_world*.json
results/                tracked results (Phase 1 LLM vs ML decision)
branding/               dashboard logo
tests/
data/                   client folders + phase1/ and phase2/ generated data (gitignored)
reports/phase1/         llm_cache/, ai_analyses.json, comparison/ (gitignored)
```
