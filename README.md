# Salud Call QA & Payment Propensity Pipeline

UIUC BIG Team B · Fall 2026 · Client: Salud Revenue Partners

One connected pipeline: **call transcript → QA / sentiment scores → payment propensity score (0–1) → ranked "call first" list**. Project scope, issue tree, benchmarking and roadmap live in `docs/`.

| Phase | Question | Method | Data |
|---|---|---|---|
| **1. Call analysis** | Did the agent handle each checklist item? What was the caller's sentiment, and how did the call end? | LLM (Groq `openai/gpt-oss-120b`), checked against manual tags | 10 real calls + 50 LLM-written synthetic calls |
| **2. Propensity** | Which patients are most likely to pay? | Traditional ML (86 variants compared; monotone LightGBM chosen) on account and payment history, plus call outcome, sentiment and QA after a call | Trained on a synthetic world tuned to Salud's data; tested on Salud's 84-call final package |

Phase 1's AI analyses (`phase1.ai_analyses`, joined to `big_synthetic` accounts) become Phase 2's call-quality input. The brief requires the two stages to connect rather than be separate projects.

## Team setup

**Just analyzing calls?** Use the team website, **https://agent-call-analysis.vercel.app**: sign in with the team passcode (shared privately by the team lead) and pick your name. There's nothing to install. The setup below is for running things locally.

**The website on Vercel** (project `agent-call-analysis`) runs the same dashboard from [`api/index.py`](api/index.py). Every push to `main` redeploys it.
- **Sign-in:** data only loads after the team passcode and a name from the assignments. The session is an HttpOnly cookie, signed with a key derived from the passcode and valid for 30 days.
- **Settings:** `DATABASE_URL` and `TEAM_PASSCODE` are stored as Vercel environment variables, never in the repo.
- **Changing the passcode** in the Vercel settings (then redeploying) signs everyone out.
- **What gets uploaded:** [`.vercelignore`](.vercelignore) uploads only `api/`, `src/`, `branding/` and config, so no client data or secrets are ever uploaded.

Everything lives in the team's **shared Neon Postgres database** (project `salud-pipeline`, database `salud`): calls and transcripts, AI analyses and every rater's tags, plus the synthetic world for Phase 2. Teammates don't need any local data files.

```bash
git clone https://github.com/iamadityakamath/SaludDemo && cd SaludDemo
uv venv && uv pip install -e '.[dev]'
cp .env.example .env     # paste DATABASE_URL (from the team lead); GROQ_API_KEY only if you run `analyze`
.venv/bin/salud-phase1 dashboard --rater <your first name>   # e.g. avni
```

- **Using Claude Code, ChatGPT/Codex or Cursor?** Open the cloned folder and ask it to *"set up and open the agent dashboard"*. [`AGENTS.md`](AGENTS.md) (also loaded by `CLAUDE.md`) walks the assistant through the setup and tells it what not to touch.
- **Connection string.** It's shared privately by the team lead. Never commit it; this repository is public.
- **Real call data.** The database holds Salud's 10 redacted real calls. This Neon project isn't HIPAA-enabled, so keep access to the team, and confirm with Salud before sharing it further.
- **Team lead only:** `salud-db setup` applies schema changes and imports local files. It's safe to rerun; it only adds what's missing.
- **Salud's final package** (84 real calls: CRM tabs + transcripts, `data/BIG_Final_Package_84`, gitignored) is in schema **`final`**, one table per workbook tab plus `final.transcripts` and `final.option_codes`, stored as received. Team lead loads it with `salud-db load-final` (`--test` for the Neon tests branch, `--replace` to reload); it checks the package first and verifies every row after.

---

## Phase 1: call analysis

### Route 1: the agent dashboard

```bash
.venv/bin/salud-phase1 dashboard --rater avni     # local: http://127.0.0.1:8765, no sign-in
```

Or use the team website (passcode, then your name). Pick a call and its stored results show straight away:
- customer sentiment: how the call ended, the shift from the start, the curve, and a score on every turn
- the QA score and checklist answers, with transcript quotes that jump to their turn
- the AI summary, the call outcome and coaching tips for the agent

**Upload calls** takes one or more `.txt` transcripts (or a pasted one), saves each as source `new` and starts its analysis at once: sentiment scored in the server and the LLM review (when `GROQ_API_KEY` is set), both saved to Neon. The call shows a spinner in the list, for everyone, until its results are stored, and the page refreshes on its own. On a laptop the server runs the analysis in the background; on the website, where nothing may run after a response, the uploader's page starts it.

**Transcripts added any other way** (a script, SQL) are picked up too:
- the laptop dashboard checks the database every 15 seconds (`--no-auto` turns this and the upload analysis off);
- the website runs once a day through a Vercel cron (`/api/cron/process`, which needs the `CRON_SECRET` environment variable; Vercel's Hobby plan allows one run a day);
- `salud-phase1 process` runs it once from the command line (`--limit N` caps the calls).

These stop at the first failed AI review (for example Groq's daily cap) and pick up again on the next run. A call that is still missing a part after 10 minutes shows a **!** in the list; opening it finishes the analysis. Stored calls load instantly at no cost. **Re-analyze** redoes both and replaces the stored sentiment and AI review (for a Phase 1 call it asks first, since that call's comparison with raters' tags will then use the new review).

Manual tagging is finished, and its page has been removed. The tags saved during Phase 1 stay in `phase1.reviewed_tags` (assigned with `salud-phase1 assign`, two raters per real call), and `salud-phase1 compare` still uses them.

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

### Turn-level sentiment

Like Google CCAI Insights: every caller and agent turn gets a **score** from -1 to 1 (P(positive) − P(negative)) and a **magnitude** from 0 to 1 (1 − P(neutral)), from [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), run as a pinned int8 ONNX copy ([Xenova's conversion](https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest), 126 MB). The same model runs on laptops and on the website, so scores don't depend on where a call was scored; it matches the full PyTorch model on 97% of turn labels. Each call gets caller and agent summaries: overall, **start** (first 30% of turns), **end** (last 30%), **shift** (end − start), lowest and highest turn, and a label (negative / mixed / neutral / positive). Results live in the team database (`phase1.turn_sentiment`, `phase1.call_sentiment`).

```bash
salud-sentiment run                # first run downloads the model (~130 MB); scores calls not scored yet -> team database (--redo to rescore)
salud-sentiment eval               # call labels vs planted labels, the LLM's label and raters' tags
```

On the 60 calls, the **end** label agrees with raters' sentiment tags about 75–80% of the time against about 60% for the whole-call average, so the dashboard leads with how the call ended.

---

## Phase 2: synthetic data and payment propensity

Salud's final package has 84 real calls, only 58 of them usable for a 30-day payment label, which is far too few to train on.
So models are trained on a **synthetic world** (a simulated call centre in Salud's own table format) and tested once on
the real calls. Results, methods and known issues: [`eval-model.md`](eval-model.md) and the watch list at the top of
[`reflections.md`](reflections.md).

### Two synthetic worlds

| World | Config | Folder | Use |
|---|---|---|---|
| **`salud`** (default) | `configs/synthetic_world_salud.json` (realism mode) | `data/phase2/synthetic_world_salud` | Tuned to Salud's final package; all modelling uses it |
| `original` | `configs/synthetic_world.json` | `data/phase2/synthetic_world` | First world (Oct 2026); loaded into Neon `big_synthetic`; its key effects pointed the wrong way, so its model failed on real data |

```bash
.venv/bin/salud-synth --config configs/synthetic_world_salud.json --out data/phase2/synthetic_world_salud
.venv/bin/salud-synth-calibrate --world data/phase2/synthetic_world_salud      # compare with Salud's data (needs DATABASE_URL)
.venv/bin/salud-propensity build                                               # modelling files (real + synthetic)
```

The Salud world holds 2,000 guarantors, 10,780 accounts, 13,288 contacts, 15,069 receipts, 2,831 QA reviews and 8,062
connected calls, each with a transcript. All 59 validation checks pass.

### Output

```
data/phase2/synthetic_world_salud/
  crm/            6 tables shaped exactly like Salud's schema (the only valid model inputs)
  truth/          hidden traits and true outcomes (for evaluation only, never use as features)
  transcripts/    one template transcript per connected call
  synthetic.sqlite    every table, for SQL access
  manifest.json       config, counts, summary stats, validation results
```

### The criteria the synthetic data must meet

1. **Same structure as Salud's tables.**
   - The world has the same 6 tables, field names and option codes as the schema handoff.
   - 59 automatic checks run on every build:
     - unique keys and no orphan links;
     - valid codes and field lengths;
     - dates in order (no payment before placement, no QA review before its call);
     - balances that reconcile to receipts, with none negative;
     - labels read from receipts that equal the planted outcomes 100% of the time.
2. **Statistically like Salud's real calls.** `salud-synth-calibrate` computes 51 statistics on both, and each synthetic value
   must fall inside the real 95% bootstrap interval (500 resamples of the 84 calls). The statistics cover:
   - **call mix (25):** direction, disposition, contact type, result, actions, talk time;
   - **patient profile (13):** accounts, hospital share, balances, debt age, payment history, insurance, adjustments;
   - **outcome (2):** paid within 30 days, and paid after a call with no payment on it;
   - **relationships (11):** each key feature's single-feature AUC, both direction and strength.

   Matching the relationships matters as much as matching the averages. The current world matches **50 of 51** on 4 seeds; the miss is
   the share of patients with a recent adjustment (0.250 against an upper bound of 0.242).
3. **A believable cause-and-effect story** (the `realism` block of the config):
   - **accounts:** a skewed count with median 2, later visits spread over about 300 days, 48% hospital, lognormal balances;
   - **insurance processing:** insurance payments and adjustments after placement;
   - **a monthly payment habit:** patients start paying with ability, willingness and debt age, then tend to continue, as on a plan;
   - **calls:** inbound calls, outbound campaigns (most reach the phone menu), transfers, and campaigns that skip patients who are paying;
   - **call results** tuned to Salud's mix, with promises often not kept.
4. **Known hidden answers.** Guarantor traits (ability, willingness, hardship), agent skill and every call's true QA items
   and outcome are in `truth/`. They are used to check what the models recover, and never as features.
5. **Realistic pitfalls, built in on purpose:**
   - **The worklist is current state.** Balance and payment-status fields reveal the outcome, so build features only from dated rows on
     or before the scoring date.
   - **Contacts and accounts are many-to-many** through `salud_Account_X_ContactHistory`. Aggregate each to account level before joining.
   - **Some receipts aren't self-pay cash.** Use `phase2.outcome_labels.self_pay_payments`.
6. **Reproducible and safe:**
   - a fixed seed gives identical data, and the original world still regenerates byte for byte;
   - no real record is copied (only Salud's aggregate statistics are targets);
   - transcripts come from templates that match the CRM outcome.

**Known gaps** (tracked in the watch list):
- New accounts with no payment history pay 19% in the synthetic world but 48% in Salud's data. The rate isn't one of the 51 checks, and the model
  under-scores new accounts.
- Real balances and adjustments have much heavier tails.
- Salud's extract has no contact history, so contact features are excluded.
- The targets come from only 84 calls.

### The propensity models

**Master file (next training data):** `salud-propensity master` writes `reports/phase2/pre_training_data.xlsx`, one row per
call for both models. Synthetic rows come from the master world (`configs/synthetic_world_master.json`: Salud's time frame,
calls up to 2026-11-18, 55/55 statistics; generate it with
`salud-synth --config configs/synthetic_world_master.json --out data/phase2/synthetic_world_master --transcripts none`);
real rows are Salud's 84 calls. Columns `data_type` (synthetic / real), `split` (train / test by patient, real_test),
`label_usable` and `post_call_row` select the rows; the pre-call model uses the 11 history columns, the post-call model
32. Synthetic sentiment and QA are sampled from the real calls with the same outcome. The notebooks still use the
earlier files until they are re-run.
Feature selection: [`feature-selection.md`](feature-selection.md) (notebook 13: which features matter, how it was checked, the saved sets).
Checks: `salud-propensity master` runs 26 integrity checks before writing; `salud-propensity verify` rebuilds every model
field and target from the raw tables with separate code and reports mismatches.

- **Pre-call:** one row per synthetic patient with an open balance on 2026-04-01. The label is any payment within 30 days; the features
  are 11 that exist in Salud's data. 86 variants are compared in `notebooks/00`–`11` with repeated grouped 5×5 cross-validation, and the
  chosen model (monotone LightGBM) is tested once on the real rows. The scores for Salud are in `reports/phase2/propensity_scores.xlsx`.
- **Post-call:** one row per call with no payment on the call. It adds how the call ended, sentiment and the QA score, and is a comparison
  only so far.
- **Real QA:** `salud-propensity qa` runs Stage 1 QA on the real calls with the dashboard's prompt. It sends the redacted transcripts to Groq,
  and results stay local in `reports/phase2/qa_real.csv`.

### Conventions (open items in the handoff)

- Timestamps are naive America/Chicago. Money is USD; receipt amounts are positive.
- `ActionsTaken` is a `;`-joined list of option codes.
- The payment horizon and the minimum amount ($0.01) are working assumptions until Salud confirms them. The models use 30 days:
  about one statement or payment-plan cycle, short enough to credit the call, and the most predictable window tested
  (30 / 60 / 90 days: 0.861 / 0.831 / 0.801). Most real payments after a call arrive within 2 weeks. Salud's data has no
  due dates, plan amounts or promised dates, so "paid on time" can't be the target yet.

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
| `phase1` | `calls` (real + synthetic, with transcripts and planted labels), `ai_analyses`, `ai_checklist`, `ai_coaching`, `ai_analysis_versions` (prompt trials), `call_sentiment`, `turn_sentiment`, `reviewed_tags`, `assignments` (also the login's list of names) |

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
    dashboard.py        route 1: the agent dashboard server   (static/analyze.html: the page)
    live.py             Analyze / Re-analyze: sentiment + LLM review, saved to the database
    analyze.py          route 2, step 1: LLM on every call -> database
    compare.py          route 2, step 2: AI vs tags -> Excel / CSV
    metrics.py          agreement, kappa, rater consensus
    calls.py            locating the Phase 1 call files (for the one-time import)
    cli.py              salud-phase1 dashboard | analyze | compare
  sentiment/            turn-level sentiment (salud-sentiment): int8 ONNX model, call summaries, eval
  final_package.py      loads Salud's final package into Neon schema 'final' (salud-db load-final)
  phase2/
    schema.py           tables and option codes from Salud's schema handoff
    outcome_labels.py   self-pay outcome labels at account grain
    synthetic/          synthetic world generator (salud-synth), calibration against Salud (salud-synth-calibrate)
    propensity/         snapshots, feature dictionary, modelling code, real QA (salud-propensity)
notebooks/              00-11: data review, model families, ensembles, final model and real test
db/migrations/          Postgres schema as code (001 big_synthetic … 007 sentiment, 009 final package)
configs/                phase1_synthetic_calls.txt, synthetic_world.json (original), synthetic_world_salud.json (default)
results/                tracked results (Phase 1 LLM vs ML decision)
branding/               dashboard logo
tests/                  database tests use the Neon 'tests' branch
data/, reports/         local source files and outputs (gitignored)
```
