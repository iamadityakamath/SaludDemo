# Salud Call QA & Payment Propensity Pipeline

UIUC BIG Team B · Fall 2026 · Client: Salud Revenue Partners

One connected pipeline: **call transcript → QA / sentiment scores → payment propensity score (0–1) → ranked "call first" list**. Project scope, issue tree, benchmarking and roadmap live in `docs/`.

The work runs in two phases:

| Phase | Question | Method | Data |
|---|---|---|---|
| **1. Call analysis** | Did the agent handle each checklist item? What was the caller's sentiment, and how did the call end? | LLM (Groq `openai/gpt-oss-120b`), checked against manual tags | 10 real calls + 168 LLM-written synthetic calls (tag 10 real + 50 synthetic) |
| **2. Propensity** | Which accounts are most likely to pay? | Traditional ML (logistic regression / gradient boosting) on account features + Phase 1's call-quality score | Thousands of synthetic accounts with payment histories |

Phase 1's per-call output (`reports/phase1/calls.csv`) is Phase 2's call-quality input. The brief requires the two stages to connect rather than be separate projects.

## Status

| Piece | State |
|---|---|
| Synthetic data generator (`salud_pipeline.synth`) | working (Phase 2 data + Phase 1 synthetic calls) |
| Review page with AI analysis (`salud-qa tag`) | working |
| Phase 1 results and export (`salud-qa phase1`) | working; waiting on manual tags |
| Manual tags: 10 real + 50 synthetic calls | in progress |
| LLM vs ML decision for Phase 1 | done: LLM only (evidence in `results/llm_vs_ml_2026-10-18/`) |
| Phase 2 propensity models, thresholds, tiers | not started |

## Setup

```bash
uv venv && uv pip install -e '.[dev]'
cp .env.example .env        # then set GROQ_API_KEY=...
.venv/bin/salud-qa ping     # confirms the key works and lists available models
```

Client files are **gitignored**: `docs/`, the redacted transcripts, the schema handoff, `data/ground_truth/` and `reports/`. Each teammate keeps their own copy.

---

## Phase 1: call analysis

### 1. Tag calls on the review page

```bash
.venv/bin/salud-qa tag --rater AK
```

It opens http://127.0.0.1:8765, listening only on this machine. The page lists the **10 real calls** and the **168 LLM-written synthetic calls** (`data/synthetic/run_llm`).

- **Filters:** by source (All, Real, Synthetic), by status (not tagged, in progress, tagged), and by call-ID search.
- **Analyze button (top right).** The LLM runs only when you press it (or ⌘/Ctrl+Enter). One prompt returns:
  - an answer for each checklist item, with a reason and exact transcript quotes
  - caller sentiment and call outcome
  - a one-line summary
  - 1–4 coaching tips
- **Evidence and flags:**
  - Quotes are checked against the transcript; unverifiable ones are hidden.
  - A section is **red** when the AI's answer has no supporting quote or no answer at all. It turns **green** once you pick the answer yourself.
- **Saving:**
  - Real-call tags go to `data/ground_truth/real_calls_labels.csv`, synthetic ones to `synthetic_calls_labels.csv`.
  - Each row records `llm_model` (blank means tagged with the AI off) and `overrides` (the items you changed).
- **AI cache:** analyses are saved in `reports/qa/assist_cache/` and reused, with no second Groq call. Changing the prompt version (`assist.PROMPT_VERSION`) means calls need to be analyzed again.

**Tagging plan:**
- 10 real + 50 synthetic calls.
- Two raters on the real calls if possible, for rater-agreement numbers.
- Tag at least some calls **before** pressing Analyze: AI-pre-filled tags tend to follow the AI, so "tagged with AI off" is the fairest benchmark.

### 2. Compare AI vs human and export

```bash
.venv/bin/salud-qa phase1                    # uses saved AI analyses only (no Groq calls)
.venv/bin/salud-qa phase1 --analyze-missing  # also analyzes tagged calls that have no AI analysis yet
```

Writes `reports/phase1/phase1_results.xlsx` (with CSV copies and `summary.md`):

| Sheet | Contents |
|---|---|
| Summary | counts and average AI–human agreement for: all tagged calls, real, synthetic, tagged with AI off |
| Calls | one row per call: AI answers, reasons, evidence quotes, needs-review flags, human answers, agree/disagree, AI and human call-quality scores, summary |
| Coaching | one row per AI coaching tip |
| AI vs human | per item: agreement with 95% CI, Cohen's kappa, missing AI answers |
| Rater agreement | per rater pair and item, when two people tagged the same calls |
| Synthetic check | human vs planted labels and AI vs planted labels (do the synthetic transcripts say what they were meant to?) |
| Definitions | checklist items, outcome categories, metrics |

**Rules used in the comparison:**
- **N/A is an answer.** A human N/A vs. an AI Yes is a disagreement.
- **A missing AI answer counts as a disagreement.**
- **Two raters who split on an item give no consensus,** and that item is left out of the AI comparison for that call.
- **Call-quality score:** Yes / (Yes + No) over the 7 transcript-visible items, with N/A excluded. It's provisional until Salud sets weights.

**Scope:** Salud's checklist has 9 items. "Correct call statusing" and "Grammar in notes" need the CRM disposition and agent note, not the transcript, so Phase 1 scores the other 7.

### LLM, Groq and real calls

- **Model:** `GROQ_MODEL` in `.env` (default `openai/gpt-oss-120b`). Any OpenAI-compatible server works through `SALUD_LLM_BASE_URL` / `SALUD_LLM_MODEL` / `SALUD_LLM_API_KEY`.
- **Groq free-tier limits on this key:**
  - **200,000 tokens per day** for gpt-oss-120b. One analysis is ≈3,200 tokens, so **about 60 analyses a day**, shared by the review page and `phase1 --analyze-missing`.
  - 8,000 tokens per minute and 1,000 requests a day per model.
  - Requests always set `max_tokens`; without it, Groq reserves a large default against the limits.
- **Real calls:**
  - Groq is a third-party cloud service, and the brief requires real call data to stay in a HIPAA-compliant environment (the docs assume a self-hosted model).
  - The real calls are redacted, but confirm with Salud before sending them.
  - The review page asks once per browser session before analyzing a real call.

### Why the LLM and not ML for Phase 1

On 60 held-out synthetic calls, the LLM agreed with the planted labels at checklist kappa **0.66**, against **0.15** for TF-IDF + logistic regression trained on realistic calls (0.38 when trained on 4,092 template calls). Salud has about 10 real calls, far too few to train a classifier. Full results and caveats are in [`results/llm_vs_ml_2026-10-18/`](results/llm_vs_ml_2026-10-18/README.md). The ML code was removed afterwards and remains in git history (commit `487a6a7`).

---

## Phase 2: synthetic data for propensity

```bash
.venv/bin/salud-synth --config configs/default.json --out data/synthetic/run_default
.venv/bin/pytest
```

A 2,000-guarantor run takes about 3 seconds and produces roughly 3.2k accounts, 11k contacts, 4.1k connected calls and 1.4k QA reviews. Every run validates itself, and the CLI exits non-zero if any check fails. Phase 1 doesn't use `run_default`'s template transcripts; Phase 2 uses its accounts, payments and contact history.

### Output

```
data/synthetic/<run>/
  crm/            6 tables shaped exactly like BIG_Synthetic_Schema.sql (the only valid model inputs)
  truth/          planted latents and outcomes (for evaluation only, never use as features)
  transcripts/    <contact id>.txt per connected call
  llm_prompts.jsonl   one generation prompt per call, with its planted labels
  synthetic.sqlite    every table, for SQL access
  manifest.json       config, counts, summary stats, validation results
```

### How the planted truth works

1. **Agents** have a skill level. Skill sets each agent's chance of passing each of the 9 QA checklist items.
2. **Guarantors** have hidden traits: ability to pay, willingness, reachability and hardship.
3. Each **connected call** draws true QA item results from its agent. The call's quality, together with the patient's traits, drives sentiment and the call outcome.
   - If Correct Call Statusing is No, the CRM records the *wrong* result.
   - If Grammar is No, the agent note is garbled.
   - These two items can only be judged from CRM fields and notes. They are not visible in the transcript.
4. **History before the cutoff:** statement-driven payments, promise-to-pay follow-through, payment plans, financial-assistance adjustments and insurance payments.
5. **Outcome window** (cutoff 2026-04-01, 90 days): each open account pays with probability `sigmoid(OutcomeModel · features)`. The coefficients are in `configs/` and `synth/config.py`.
   - `configs/null_call_quality.json` sets the call-quality effect to 0, to check that the pipeline doesn't find signal where none was planted.
   - That config removes only the *direct* effect. Better calls still produce more on-call payments and plans before the cutoff, so some raw correlation between quality and payment remains, just as in real data.

### Traps that are deliberate (and realistic)

- **The worklist is current state**, as of the day after the outcome window. `SelfPayBalance`, `SelfPayReceipts` and `LastPatientPaymentFromTxnsDate` reveal the outcome. Build features only from dated rows on or before the cutoff.
- **Contacts and accounts are many-to-many** through `salud_Account_X_ContactHistory`. Aggregate contacts and receipts to account level separately before joining them.
- **Some receipts must not count as self-pay cash:** excluded rows, NULL payment types, adjustments and insurance payments. Use `labels.self_pay_payments`.

### Synthetic conventions (open items in the handoff)

- Timestamps are naive America/Chicago. Money is USD; receipt amounts are positive.
- `ActionsTaken` is a `;`-joined list of option codes.
- The payment horizon (90 days) and the minimum qualifying amount ($0.01) are working assumptions until Salud confirms them.

### Transcript modes

- `--transcripts template` (default): rule-based, reproducible, formulaic wording. Fine for Phase 2 data; not realistic enough to test call analysis.
- `--transcripts llm --llm-model qwen/qwen3.8-27b --llm-limit N`: LLM-written calls from the planted labels. It's resumable and records its order in `llm_order.txt`. Use a **different model family** from the scorer. On Groq's free tier, Qwen is capped at 1,000 output tokens a minute (about 1 call per minute). Qwen doesn't always follow every planted label; the "Synthetic check" sheet measures this.

---

## Layout

```
src/salud_pipeline/
  schema.py        table/column/option-code constants from the handoff
  labels.py        self-pay outcome labels at account grain (Phase 2)
  llm_client.py    OpenAI-compatible client (.env, Groq defaults, retries, max_tokens)
  synth/           synthetic world, transcripts (template and LLM), validation, CLI
  qa/              Phase 1: checklist targets, one-prompt call analysis (assist), review page,
                   AI-vs-human agreement (evaluate, phase1), CLI
configs/           generator configs (default, null call-quality effect)
results/           tracked results (LLM vs ML decision)
branding/          logo for the review page
tests/
```
