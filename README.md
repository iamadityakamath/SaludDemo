# Salud Call QA & Payment Propensity Pipeline

UIUC BIG Team B · Fall 2026 · Client: Salud Revenue Partners

One connected pipeline: **call transcript → QA / sentiment scores → payment propensity score (0–1) → ranked "call first" list**. Project scope, issue tree, benchmarking and roadmap live in `docs/`.

## Status

| Piece | State |
|---|---|
| Synthetic data generator (`salud_pipeline.synth`) | v0.1, working |
| Outcome labels from receipts (`salud_pipeline.labels`) | v0.1 |
| QA scoring harness (`salud_pipeline.qa`): LLM scorer, TF-IDF + LR scorer, metrics | v0.1, LLM not yet run (waiting on team endpoint) |
| Hand-tagged ground truth for the 10 real calls | template ready, not filled |
| Propensity models, thresholds, tiers | not started (Phase 5) |

## Setup

```bash
uv venv && uv pip install -e '.[dev]'
```

Client files are **gitignored** (`docs/`, the redacted transcripts and the schema handoff). Each teammate keeps their own copy under `data/` and `docs/`. The generator runs without them. The validator also cross-checks option codes against `data/BIG_Schema_Only_Handoff/BIG_Schema.json` when that file is present.

## Generate synthetic data

```bash
.venv/bin/salud-synth --config configs/default.json --out data/synthetic/run_default
.venv/bin/pytest
```

A 2,000-guarantor run takes about 3 seconds and produces roughly 3.2k accounts, 11k contacts, 4.1k connected calls with transcripts, and 1.4k QA reviews. Every run validates itself, and the CLI exits non-zero if any check fails.

### Output

```
data/synthetic/<run>/
  crm/            6 tables shaped exactly like BIG_Synthetic_Schema.sql (the only valid model inputs)
  truth/          planted latents and outcomes (for evaluation only, never use as features)
  transcripts/    <contact id>.txt per connected call, same format as the redacted samples
  llm_prompts.jsonl   one generation prompt per call, with its planted labels
  synthetic.sqlite    every table, for SQL access
  manifest.json       config, counts, summary stats, validation results
```

### How the planted truth works

1. **Agents** have a skill level. Skill sets each agent's chance of passing each of the 9 QA checklist items.
2. **Guarantors** have hidden traits: ability to pay, willingness, reachability and hardship.
3. Each **connected call** draws true QA item results from its agent. The call's quality, together with the patient's traits, drives sentiment and the call outcome (paid, promise to pay, plan, dispute, and so on).
   - If Correct Call Statusing is No, the CRM records the *wrong* result.
   - If Grammar is No, the agent note is garbled.
   - These two items can only be judged from CRM fields and notes. They are not visible in the transcript.
4. **History before the cutoff**: statement-driven payments, promise-to-pay follow-through, payment plans, financial-assistance adjustments and insurance payments.
5. **Outcome window** (cutoff 2026-04-01, 90 days): each open account pays with probability `sigmoid(OutcomeModel · features)`. The coefficients are in `configs/` and `synth/config.py`.
   - `configs/null_call_quality.json` sets the call-quality effect to 0. Use it to check that the pipeline doesn't find signal where none was planted.
   - That config removes only the *direct* effect. Better calls still produce more on-call payments and plans before the cutoff, which raise `prior_payer` and `on_plan` and lower the balance. So some raw correlation between quality and payment remains, just as it would in real data.

### Traps that are deliberate (and realistic)

- **The worklist is current state**, as of the day after the outcome window. `SelfPayBalance`, `SelfPayReceipts` and `LastPatientPaymentFromTxnsDate` reveal the outcome. Build features only from dated rows on or before the cutoff.
- **Contacts and accounts are many-to-many** through `salud_Account_X_ContactHistory`. Aggregate contacts and receipts to account level separately before joining them.
- **Some receipts must not count as self-pay cash**: excluded rows, NULL payment types, adjustments and insurance payments. Use `labels.self_pay_payments`.

### Synthetic conventions (open items in the handoff)

- Timestamps are naive America/Chicago.
- Money is USD; receipt amounts are positive.
- `ActionsTaken` is a `;`-joined list of option codes.
- The payment horizon (90 days) and the minimum qualifying amount ($0.01) are working assumptions until Salud confirms them.

### Transcript modes

- `--transcripts template` (default) is rule-based and reproducible. Its phrasing is formulaic, so TF-IDF models will look better on it than on real calls. **Never report template-only accuracy as a result.**
- `--transcripts llm` writes each call through the team LLM endpoint (see below; `--llm-model` overrides the model).
  - Use a **different model family** for generation than for scoring, so the scorer isn't graded on its own writing style.
  - Always report accuracy on the 10 real hand-tagged calls separately.

## LLM endpoint

The LLM scorer and LLM transcript mode both call the team-hosted, OpenAI-compatible server (vLLM, Ollama or llama.cpp). It is configured through environment variables, so URLs and keys never go into git:

```bash
export SALUD_LLM_BASE_URL="https://<team-server>/v1"
export SALUD_LLM_MODEL="<model id from salud-qa ping>"
export SALUD_LLM_API_KEY="<token, if the server needs one>"
```

Synthetic calls contain no PHI and can go to any team server. The 10 **real** redacted calls are sent to a non-local server only when you pass `--include-real`. Pass it only once the server is confirmed inside the HIPAA-approved environment, which is still pending Salud's sign-off.

## QA scoring (Phase 4)

```bash
.venv/bin/salud-qa template      # writes data/ground_truth/real_calls_labels.csv (gitignored)
.venv/bin/salud-qa ml            # learning curve on synthetic calls, plus real calls once tagged
.venv/bin/salud-qa ping         # check the team endpoint and list its models
.venv/bin/salud-qa llm --n 150   # add --include-real once the server is HIPAA-approved
.venv/bin/salud-qa summary
```

Every scorer predicts the same targets: the 7 checklist items that are visible in a transcript, plus caller sentiment and call outcome. Call Statusing and Grammar come from the CRM record and the agent note, not the transcript.

**Ground truth.** Each rater fills one row per call, entering their initials in `rater`. Use Yes/No/N/A for the checklist items, negative/neutral/positive for sentiment, and one outcome from `qa/targets.py`. With two or more raters, inter-rater kappa is reported automatically.

**Findings so far (Oct 16).** Trained on template calls, TF-IDF + LR reaches ~100% on synthetic test calls by around 500 training calls. That figure is an upper bound caused by formulaic phrasing, not a result. On the real calls it mislabels most greetings as failures and calls a transfer a hang-up. Real accuracy can't be measured until the calls are hand-tagged.

## Layout

```
src/salud_pipeline/
  schema.py        table/column/option-code constants from the handoff
  labels.py        self-pay outcome labels at account grain
  synth/           config, world simulator, transcripts, LLM prompts, validation, CLI
  qa/              QA targets, LLM scorer, ML scorer, evaluation, CLI
configs/           generator configs (default, null call-quality effect)
tests/             validation, reproducibility, planted-effect recovery
```
