# Salud Call QA & Payment Propensity Pipeline

UIUC BIG Team B · Fall 2026 · Client: Salud Revenue Partners

One connected pipeline: **call transcript → QA / sentiment scores → payment propensity score (0–1) → ranked "call first" list**. Project scope, issue tree, benchmarking and roadmap live in `docs/`.

| Phase | Question | Method | Data |
|---|---|---|---|
| **1. Call analysis** | Did the agent handle each checklist item? What was the caller's sentiment, and how did the call end? | LLM (Groq `openai/gpt-oss-120b`), checked against manual tags | 10 real calls + 50 LLM-written synthetic calls |
| **2. Propensity** | Which accounts are most likely to pay? | Traditional ML on account features + Phase 1's call-quality score | Synthetic accounts with payment histories |

Phase 1's AI analyses (`phase1.ai_analyses`, joined to `big_synthetic` accounts) become Phase 2's call-quality input. The brief requires the two stages to connect rather than be separate projects.

## Team setup

Everything lives in the team's **shared Neon Postgres database** (project `salud-pipeline`, database `salud`): calls and transcripts, AI analyses and every rater's tags, plus the synthetic world for Phase 2. Teammates don't need any local data files.

```bash
git clone https://github.com/iamadityakamath/SaludDemo && cd SaludDemo
uv venv && uv pip install -e '.[dev]'
cp .env.example .env     # paste DATABASE_URL (from the team lead); GROQ_API_KEY only if you run `analyze`
.venv/bin/salud-phase1 dashboard --rater <your first name>   # e.g. avni
```

- **Connection string.** It's shared privately by the team lead. Never commit it; this repository is public.
- **Real call data.** The database holds Salud's 10 redacted real calls. This Neon project isn't HIPAA-enabled, so keep access to the team, and confirm with Salud before sharing it further.
- **Team lead only:** `salud-db setup` applies schema changes and imports local files. It's safe to rerun; it only adds what's missing.

---

## Phase 1: call analysis

### Route 1: dashboard (review the AI's answers and tag)

```bash
.venv/bin/salud-phase1 assign --raters avni,sharayu,aditya,ibrahim,ananya,rudrakshi   # team lead, once
.venv/bin/salud-phase1 dashboard --rater avni
```

Each teammate runs it on their own laptop (http://127.0.0.1:8765). They all read and write the same database.

- **Who tags what:** `assign` splits the calls evenly. Each real call gets **2 raters**, so rater agreement can be measured on Salud's own calls, and each synthetic call gets 1. That's 70 tags, 11–12 per person. Running it again only fills calls that are short of raters and never moves existing assignments. `salud-phase1 assign` with no options shows who has what and how many each person has tagged.
- **The list opens on "My calls".** The menu at the top of the list switches to "All raters" or to another rater's calls; badges like `Av · Sh` show who's assigned. Use the same first name you were assigned under (case doesn't matter).

- **Every call opens pre-filled** with its AI analysis:
  - checklist answers, with reasons and highlighted transcript quotes
  - caller sentiment and call outcome
  - a summary and coaching tips
- **The dashboard never calls the LLM and never changes AI analyses.**
- **You check, correct and save.** A section is **red** when the AI's answer has no supporting quote, and turns **green** once you choose the answer yourself.
- **Saving writes your row** in `phase1.reviewed_tags` (one row per call per rater):
  - `llm_model` records which AI answers pre-filled it.
  - `overrides` lists the items you changed.
  - Raters only see their own tags.

### Route 2: script (AI analysis, then comparison)

```bash
.venv/bin/salud-phase1 analyze             # LLM on calls without an analysis -> phase1.ai_analyses
.venv/bin/salud-phase1 analyze --limit 20  # cap new LLM requests this run
.venv/bin/salud-phase1 compare             # AI vs every rater's tags -> reports/phase1/comparison/
```

- **`analyze`** only adds analyses for calls that have none.
  - It never changes an existing analysis.
  - It refuses to run if the model or prompt version differs from the stored analyses.
  - It stops cleanly at Groq's daily cap.
- **`compare`** reads everything from the database and writes `phase1_results.xlsx` (plus CSVs and `summary.md`) locally. With pre-filled tags, **agreement = the share of AI answers the reviewer kept**, and each disagreement is a correction.

| Sheet | Contents |
|---|---|
| Summary | counts and average AI–human agreement: all tagged calls, real, synthetic |
| Calls | one row per call: AI answers, reasons, evidence, needs-review flags, human answers, agree/disagree, call-quality scores, summary |
| Coaching | one row per AI coaching tip |
| AI vs human | per item: agreement with 95% CI, Cohen's kappa, missing AI answers |
| Rater agreement | per rater pair and item |
| Synthetic check | human vs planted labels and AI vs planted labels |
| Salud format | every saved tag in Salud's `salud_gphpvqualityreview` columns and option codes (also `salud_quality_review.csv`) |
| Definitions | checklist items, outcomes, metrics |

**Comparison rules:**
- **N/A is an answer.** A human N/A vs. an AI Yes is a disagreement.
- **A missing AI answer counts as a disagreement.**
- **Two raters who split on an item give no consensus,** and that item is left out of the AI comparison for that call.
- **Call-quality score:** Yes / (Yes + No) over the 7 transcript-visible items, with N/A excluded. It's provisional until Salud sets weights.

**Differences from Salud's QA form** (the handoff README and `BIG_Schema.json`). A completed Salud review needs a Yes/No score and a comment for each of 9 items. Phase 1 deliberately differs:

| Salud's form | Phase 1 | In the Salud format export |
|---|---|---|
| 9 items | 7 items. "Correct call statusing" and "Grammar in notes" are judged from the CRM disposition and agent note, which the real transcripts don't have. | Those 2 scores are NULL. |
| Yes / No only (N/A excluded on purpose) | Yes / No / N/A, kept for tagging and agreement | N/A is NULL, and the item is listed in `na_items`. |
| A comment per item | One note per call | Comment columns are NULL. |

Yes = `785570000` and No = `785570001`. NULL means "not scored" (Salud's rule for an incomplete review), so every exported row is marked `review_complete = False`. `salud_QualityReviewId` is the reviewed contact: the synthetic call's `contact_id`, and NULL for real calls, which have no CRM record here.

### The Phase 1 calls

- **Real:** the 10 redacted Salud calls.
- **Synthetic:** 50 calls written by Qwen (a different model family from the scorer) from planted labels, picked from 168 to cover every outcome, scenario and checklist failure. They're listed in [`configs/phase1_synthetic_calls.txt`](configs/phase1_synthetic_calls.txt).
- **Each synthetic call is a real row** in the synthetic CRM's contact history, so its quality score joins to the account and payment data Phase 2 uses.

### LLM and Groq

- **Model:** `GROQ_MODEL` in `.env` (default `openai/gpt-oss-120b`), or any OpenAI-compatible server through `SALUD_LLM_BASE_URL` / `SALUD_LLM_MODEL` / `SALUD_LLM_API_KEY`. Only `analyze` uses it.
- **Editing the prompt:** change [`prompt.py`](src/salud_pipeline/phase1/prompt.py) and the definitions in [`checklist.py`](src/salud_pipeline/phase1/checklist.py), then bump `PROMPT_VERSION`. `analyze` won't mix versions in the database.
- **Groq free-tier limits on this key:** 200,000 tokens a day for gpt-oss-120b. One analysis is ≈3,200 tokens, so about 60 analyses a day.
- **Real calls go to Groq,** a third-party cloud service, when analyzed. Confirm with Salud.

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

---

## Database (Neon Postgres)

Salud's own data lives in **Microsoft SQL Server**: a Dynamics CRM database, `saludrevenue_MSCRM.dbo`, per the schema handoff. The team database mirrors that structure in Postgres and adds Phase 1.

```bash
.venv/bin/salud-db status   # row counts per table
.venv/bin/salud-db setup    # team lead: apply new migrations, import local files (add-only, safe to rerun)
```

| Schema | Tables |
|---|---|
| `big_synthetic` | Salud's 6 CRM tables (accounts, contact history, account↔contact bridge, receipts, QA reviews, guarantors), same names as their SQL Server script, filled with synthetic data |
| `phase1` | `calls` (real + synthetic, with transcripts and planted labels), `ai_analyses`, `ai_checklist`, `ai_coaching`, `reviewed_tags`, `legacy_manual_tags` (tags made before AI pre-fill; not used in comparisons) |

- **Connections:** `DATABASE_URL` (pooled) is used by the dashboard and scripts. `DATABASE_URL_UNPOOLED` (direct) is used by `salud-db` for schema changes.
- **SQL Server → Postgres:** `uniqueidentifier`→`uuid`, `nvarchar`→`varchar`/`text`, `money`→`numeric(19,4)`, `datetime`→`timestamp`, `bit`→`boolean`. Names are unchanged, so moving to Salud's SQL Server is a type swap.
- **Schema changes:** add a new numbered file in `db/migrations/` and run `salud-db setup`. Applied migrations are recorded in `public.schema_migrations`.
- **Tests** run against the Neon branch `tests` (`DATABASE_URL_TEST` in `.env`), a copy-on-write clone, never the team's live data. Each test removes what it created.
- **Backups:** Neon keeps 6 hours of history on this plan for point-in-time restore.

## Layout

```
src/salud_pipeline/
  paths.py              every project location
  llm_client.py         OpenAI-compatible client (.env, Groq defaults, retries, max_tokens)
  db.py                 salud-db setup | status (migrations, one-time import)
  phase1/
    store.py            the team database for Phase 1 (calls, transcripts, AI analyses, tags)
    checklist.py        checklist items, outcomes, definitions
    prompt.py           the LLM prompt, response checking, local LLM cache
    dashboard.py        route 1: dashboard server   (static/dashboard.html: the page)
    analyze.py          route 2, step 1: LLM on every call -> database
    compare.py          route 2, step 2: AI vs tags -> Excel / CSV
    metrics.py          agreement, kappa, rater consensus
    calls.py            locating the Phase 1 call files (for the one-time import)
    cli.py              salud-phase1 dashboard | analyze | compare
  phase2/
    schema.py           tables and option codes from Salud's schema handoff
    outcome_labels.py   self-pay outcome labels at account grain
    synthetic/          synthetic world generator (salud-synth); also wrote the Phase 1 calls
db/migrations/          Postgres schema as code (001 big_synthetic, 002 phase1, 003 shared team db, 004 planted labels)
configs/                phase1_synthetic_calls.txt, synthetic_world*.json
results/                tracked results (Phase 1 LLM vs ML decision)
branding/               dashboard logo
tests/                  database tests use the Neon 'tests' branch
data/, reports/         local source files and outputs (gitignored)
```
