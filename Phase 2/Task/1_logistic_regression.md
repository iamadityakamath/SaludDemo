# Task 1 of 6 · Logistic regression (L1 / L2 / elastic net)

> **Who this is for.** One teammate (and their AI assistant) builds **logistic regression (L1 / L2 / elastic net)** models for Salud's payment-propensity
> project. Five other people do the same for other model families with the same data, the same rules and the same
> evaluation code, so all results can be compared and combined at the end. Work **only** on your model family.
>
> **If you are an AI assistant:** this file is your full instruction. Read all of it before starting. It overrides the
> repository's `AGENTS.md`, which is about setting up the call-analysis dashboard. Follow the data rules in section 6
> strictly: **never print rows of real data into the chat**, only aggregates. Do not change files outside the ones
> listed in section 11. Ask the person when something here is unclear rather than guessing.

## 1 · The project in two minutes

Salud is a medical-billing call centre. It calls patients who owe money on hospital or physician accounts. We predict
**whether a patient makes a payment of their own within 30 days**, so the team can call the most likely payers first and
learn what calls lead to payment. There are **two models**, and you build both for your family:

| | Pre-call model | Post-call model |
|---|---|---|
| Question | Before calling: will this patient pay within 30 days? | After a call: will they pay in the 30 days after it? |
| Use | rank who to call first | follow-up priority; what about a call predicts payment |
| Knows | the patient's history up to the call day | history + what happened on the call |
| Target column | `paid_30d` (1 = paid) | `paid_after_call_30d` (1 = paid) |
| Rows | every usable call | only calls **without** a payment taken on the call (`post_call_row = 1`) |

Definitions: one row = one recorded call, described as of the call's date and time. "Paid" = a transaction of type
*Payment / Patient Payment*, any amount, any of the patient's accounts; `paid_30d` counts the call day through 30 days
later, `paid_after_call_30d` counts day 1 through day 30. Insurance payments and adjustments do not count.

The training data is **synthetic**: a simulated call centre in Salud's table format, tuned to match 55 statistics of
Salud's real data (accounts from 2023, calls up to 18 Nov 2026). It also contains **Salud's 84 real calls**, of which 58
have a usable label. Background, if you want it: `README.md`, `Phase 2/progress/feature-selection.md` (which features matter and why),
`Phase 2/notebooks/01_master_data_eda.ipynb` and `Phase 2/notebooks/02_feature_selection.ipynb`.

**Primary metric: AUC** (how well the model ranks payers above non-payers; 0.5 = random, 1 = perfect). Also tracked:
PR-AUC, Brier score and log loss (are the probabilities right), calibration slope (ideal 1.0), share of payers caught in
the top 20% of scores.

**The number to beat:** the rule *"patients who paid before first, most recent payers first"* (no model). It scores
**0.716 (pre-call) / 0.720 (post-call)** on the real rows and **0.677** on the synthetic test split.

## 2 · Your job in one paragraph

Train logistic regression (L1 / L2 / elastic net) for both models on six feature sets, tune them, use the real rows fairly to improve them, pick one final
model per target, score it once on the synthetic test split, and hand back a notebook, a results file in the shared
format, prediction files and a short report on a git branch. Section 8 lists exactly what to try for your family.

## 3 · Setup (once)

Requires Python 3.11+ and git.

```bash
git clone <the repo URL your lead gives you> && cd <repo folder>
git checkout final && git pull
git checkout -b task/logistic_regression                     # your own branch; never commit to final or main
uv venv && uv pip install -e '.[dev,model]'      # or: python3 -m venv .venv && .venv/bin/pip install -e '.[dev,model]'
.venv/bin/nbstripout --install                   # REQUIRED: strips notebook outputs on commit (they can show real data)
.venv/bin/python -m ipykernel install --user --name salud --display-name "Salud (project venv)"
```

**On Windows** (PowerShell): `py -3.11 -m venv .venv`, then `.venv\Scripts\pip install -e ".[dev,model]"`,
`.venv\Scripts
bstripout --install` and `.venv\Scripts\python -m ipykernel install --user --name salud`. Wherever this file
says `.venv/bin/`, use `.venv\Scripts\`. Paths with a space (`Phase 2/...`) need quotes in the terminal.

In Jupyter, VS Code or Cursor, pick the kernel **Salud (project venv)** for your notebook.

**The data file.** Your lead sends you `pre_training_data.xlsx` through a private channel. Put it at
`Phase 2/reports/pre_training_data.xlsx` (that folder is gitignored, so it is never committed). Do not upload
it anywhere else, and do not pass it to any online service.

**Check you have the right file** (run once; every number must match):

```python
import hashlib, pandas as pd
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())   # works from any folder
m = pd.read_excel(ROOT / "Phase 2" / "reports" / "pre_training_data.xlsx")
key = m.sort_values("call_id")[["call_id", "split", "paid_30d", "paid_after_call_30d"]].astype(str).agg("|".join, axis=1)
print(m.shape, hashlib.sha256("\n".join(key).encode()).hexdigest()[:16])
print(m.groupby(["data_type", "split"]).agg(rows=("call_id", "size"), usable=("label_usable", "sum")))
```

Expected: shape `(8833, 76)`, fingerprint `acd75c39fe5bfa98`, and

| data_type | split | rows | usable |
|---|---|---|---|
| real | real_test | 84 | 58 |
| synthetic | test | 1,778 | 1,702 |
| synthetic | train | 6,971 | 6,647 |

If anything differs, stop and ask your lead for the current file.

**Your folder.** Everything you produce goes in **`Phase 2/notebooks/logistic_regression/`** (it already exists in the repo):
create your notebook there as `Phase 2/notebooks/logistic_regression/logistic_regression.ipynb`. The shared code writes your results file and
prediction files into the same folder, and you write your report there too. Open the notebook from that folder (Jupyter, VS Code or Cursor);
the shared code finds the data from anywhere in the repo. Don't put files anywhere else in the repo.

## 4 · The data: which rows are for what

Always keep only rows with `label_usable = True` (the patient has account data and the full 30 days are inside the data).

| Part | Selection | Pre-call rows | Post-call rows | Use it for |
|---|---|---|---|---|
| **Synthetic train** | `data_type = synthetic`, `split = train` | 6,647 (1,477 patients) | 5,368 | training, tuning, comparing (grouped 5-fold cross-validation) |
| **Synthetic test** | `data_type = synthetic`, `split = test` | 1,702 | 1,365 | **once**, at the end, for your final model per target |
| **Real** | `data_type = real` | 58 | 47 | extra training rows and fair checks, under the rules in section 6 |

- Pay rates: synthetic 0.57 (pre-call) / 0.46 (post-call); real 0.50 / 0.36.
- Each synthetic patient has about 4–5 calls. **Rows of one patient are not independent**, so every split must keep a
  patient's calls together. The `split` column already does (no patient is in both train and test), and the shared
  code's folds are grouped by `guarantor_id`.
- The synthetic test split is the honest check on unseen patients. Don't look at it until your final step; every look
  makes it less honest.

## 5 · Features

### Candidates

- **Pre-call (21):** `n_accounts`, `n_hospital_accounts`, `n_physician_accounts`, `placement_balance`, `balance_at_t`,
  `days_since_first_placement`, `days_since_last_placement`, `days_since_last_discharge`, `n_payments_90d`,
  `amount_paid_90d`, `n_payments_365d`, `amount_paid_365d`, `n_payments_ever`, `amount_paid_ever`,
  `days_since_last_payment` (empty = never paid), `n_insurance_payments_365d`, `amount_insurance_365d`,
  `n_adjustments_365d`, `amount_adjustments_365d`, `paid_share_so_far`, `last_payment_amount`.
- **Post-call adds (21):** `inbound`, `spoke_with_agent`, `collection_attempted`, `promised_to_pay`, `non_committal`,
  `contested_or_refused`, `pursued_insurance`, `transferred`, `action_shared_balance`, `action_payment_plan`,
  `action_financial_assistance`, `action_statement_or_bill`, `action_insurance`, `n_actions`,
  `asked_financial_assistance`, `talk_seconds`, `caller_sentiment_overall`, `caller_sentiment_end`,
  `caller_sentiment_shift`, `agent_sentiment_overall`, `qa_score`.
- **Never use as inputs** (they contain the answer or describe the row): `paid_30d`, `paid_after_call_30d`,
  `paid_amount_30d`, `days_to_first_payment`, `paid_fraction_30d`, `paid_on_call`, `payment_on_call`, `label_usable`,
  `label_reliable`, `window_complete`, `data_type`, `split`, `post_call_row`, `sentiment_source`, `qa_source`, any id,
  `as_of`, `agent`, `ever_paid` (exact copy of "never paid"), `n_prior_contacts_90d`, `n_prior_calls`,
  `days_since_last_contact` (Salud's data has no contact history). The shared code checks this.

### The six feature sets (from `Phase 2/progress/feature-selection.md`; defined in the shared code as `SETS`)

| Set | Pre-call | Post-call | Idea |
|---|---|---|---|
| `stable` | 2 | 5 | the features every method agreed on |
| `top5` | 5 | 5 | a small, readable model |
| `l1_one_se` | 7 | 14 | what L1 regularization kept |
| `tiers12` | 7 | 10 | tier 1 + tier 2 of the tier list |
| `current` | 11 | 32 | the fields the current models use (reference) |
| `all` | 21 | 42 | every candidate |

What is known already (`Phase 2/progress/feature-selection.md`): payment history dominates (`n_payments_90d`, `days_since_last_payment`;
post-call also `action_payment_plan`, `promised_to_pay`). Payment-history fields are near-duplicates of each other
(ρ 0.84–1.00). Synthetic sentiment and QA were generated from the call outcome, so they can't show value beyond it.
Honest (nested) cross-validation gave about 0.707 / 0.797 with all features and about 0.692 / 0.782 with the top 5,
for logistic regression and LightGBM.

## 6 · Rules for the real data (read carefully)

**Privacy.** The real rows are Salud's patients (redacted IDs, but real billing data).
- Never print, paste or screenshot real rows (individual values, IDs, or labels) into a chat, an AI prompt, a report or
  a commit. Aggregates only (counts, rates, AUCs).
- Never commit the data file or anything that contains real rows or real-row predictions. The shared code writes those
  to `Phase 2/reports/tasks/logistic_regression/` (gitignored). Send that folder to your lead privately at the end.
- Don't send the data to online services or external LLMs.
- Notebook outputs are stripped on commit only if you ran `nbstripout --install` (section 3). Check before your first commit.

**Fair use.** You may use the real rows to improve your model: as extra training rows (with a weight), for calibration,
and to compare versions. But **a real row may never be scored by a model that was trained on it.** Always measure real
performance with `real_eval()` from the shared code. It splits the real patients into 5 groups, scores each group with
a model trained on synthetic data plus the other 4 groups, repeats this 10 times, and averages the AUC within groups.
- Every call to `real_eval()` is counted (`real_looks_so_far`) and logged. Report the total in your report.
- With 58 / 47 rows, real AUCs are uncertain by about **±0.10 to ±0.15**. Don't chase differences smaller than that.
  Synthetic cross-validation (thousands of rows, SE ≈ 0.01) is the precise measure; use real data as a second opinion.
- Don't fit anything on the real rows outside the shared functions (no "fit, then score the same rows").

## 7 · Shared code (copy into the first cell of your notebook; `FAMILY` is already set to `"logistic_regression"`)

Everyone uses this block unchanged, so results are comparable. It is self-contained (needs only pandas, numpy,
scikit-learn; LightGBM for task 5). It provides:

| Name | What it does |
|---|---|
| `SYN_TRAIN`, `SYN_TEST`, `REAL` | the three parts of the data (usable rows only) |
| `rows(df, target)` | the rows for `"pre"` or `"post"` |
| `SETS[target][set_name]` | the six feature sets |
| `Prep(log=True)` | fills "never paid" (+ a `never_paid` flag) and other gaps; optional log1p for skewed fields. Put it inside your pipeline |
| `cv_eval(make_model, feats, target)` | grouped 5-fold CV on synthetic train → mean metrics + out-of-fold predictions |
| `real_eval(make_model, feats, target, real_weight)` | fair real-data score (section 6); `real_weight` = 0, 1, 5, 20 … |
| `rule_real_auc(target)`, `rule_scores(df)` | the baseline rule |
| `log_result(variant, target, set_name, feats, params, cv, real)` | appends one row to `Phase 2/notebooks/logistic_regression/results.csv` |
| `final_model(make_model, feats, target, real_weight, variant, set_name, params)` | the final step (section 9) |

`make_model` is a function with no arguments that returns a **new, unfitted** scikit-learn `Pipeline` whose last step
is named `"model"` (needed for sample weights), for example
`lambda: Pipeline([("prep", Prep()), ("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, max_iter=5000))])`.
Estimators that don't accept sample weights still work: real rows then count once each.

```python
# ===================== SHARED SETUP: copy this whole block into the first cell, unchanged =====================
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

FAMILY = "logistic_regression"          # your model family (already filled in in your task file)
SEED = 42
# The repository root, found from wherever the notebook runs (Jupyter starts in the notebook's own folder).
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists() and (p / "Phase 2").exists())
DATA = ROOT / "Phase 2" / "reports" / "pre_training_data.xlsx"
OUT = ROOT / "Phase 2" / "notebooks" / FAMILY                    # your folder: notebook + committed, synthetic-only outputs
PRIVATE = ROOT / "Phase 2" / "reports" / "tasks" / FAMILY       # NOT committed (gitignored): anything derived from real rows
OUT.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)

# ---------- data ----------
m = pd.read_excel(DATA)
BOOLS = ["label_usable", "post_call_row", "paid_30d", "paid_after_call_30d", "inbound", "spoke_with_agent",
         "collection_attempted", "promised_to_pay", "non_committal", "contested_or_refused", "pursued_insurance",
         "transferred", "action_shared_balance", "action_payment_plan", "action_financial_assistance",
         "action_statement_or_bill", "action_insurance", "asked_financial_assistance"]
for c in BOOLS:
    m[c] = m[c].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}).astype(float)
u = m[m.label_usable == 1]
SYN_TRAIN = u[(u.data_type == "synthetic") & (u.split == "train")]
SYN_TEST = u[(u.data_type == "synthetic") & (u.split == "test")]
REAL = u[u.data_type == "real"]
TARGET = {"pre": "paid_30d", "post": "paid_after_call_30d"}

def rows(df, target):
    """Pre-call model: every usable row. Post-call model: only calls without a payment on the call."""
    return df[df.post_call_row == 1] if target == "post" else df

# ---------- features ----------
PRE_ALL = ["n_accounts", "n_hospital_accounts", "n_physician_accounts", "placement_balance", "balance_at_t",
           "days_since_first_placement", "days_since_last_placement", "days_since_last_discharge", "n_payments_90d",
           "amount_paid_90d", "n_payments_365d", "amount_paid_365d", "n_payments_ever", "amount_paid_ever",
           "days_since_last_payment", "n_insurance_payments_365d", "amount_insurance_365d", "n_adjustments_365d",
           "amount_adjustments_365d", "paid_share_so_far", "last_payment_amount"]
CALL = ["inbound", "spoke_with_agent", "collection_attempted", "promised_to_pay", "non_committal", "contested_or_refused",
        "pursued_insurance", "transferred", "action_shared_balance", "action_payment_plan", "action_financial_assistance",
        "action_statement_or_bill", "action_insurance", "n_actions", "asked_financial_assistance", "talk_seconds",
        "caller_sentiment_overall", "caller_sentiment_end", "caller_sentiment_shift", "agent_sentiment_overall", "qa_score"]
CURRENT_PRE = ["n_accounts", "n_hospital_accounts", "n_physician_accounts", "balance_at_t", "days_since_first_placement",
               "days_since_last_placement", "n_payments_90d", "n_payments_365d", "days_since_last_payment",
               "n_insurance_payments_365d", "n_adjustments_365d"]
TIER12_PRE = ["n_payments_90d", "days_since_last_payment", "n_payments_365d", "paid_share_so_far", "last_payment_amount",
              "days_since_first_placement", "days_since_last_placement"]
SETS = {
    "pre": {
        "stable": ["n_payments_90d", "days_since_last_payment"],
        "top5": ["n_payments_90d", "days_since_last_payment", "n_payments_365d", "amount_adjustments_365d",
                 "days_since_first_placement"],
        "l1_one_se": ["n_payments_90d", "days_since_last_payment", "n_payments_365d", "amount_adjustments_365d",
                      "days_since_first_placement", "paid_share_so_far", "n_physician_accounts"],
        "tiers12": TIER12_PRE,
        "current": CURRENT_PRE,
        "all": PRE_ALL,
    },
    "post": {
        "stable": ["n_payments_90d", "days_since_last_payment", "action_payment_plan", "promised_to_pay",
                   "days_since_first_placement"],
        "top5": ["n_payments_90d", "days_since_last_payment", "action_payment_plan", "promised_to_pay",
                 "days_since_last_placement"],
        "l1_one_se": ["n_payments_90d", "days_since_last_payment", "action_payment_plan", "promised_to_pay",
                      "days_since_last_placement", "days_since_first_placement", "paid_share_so_far",
                      "amount_adjustments_365d", "spoke_with_agent", "n_payments_365d", "action_shared_balance",
                      "action_financial_assistance", "n_physician_accounts", "action_statement_or_bill"],
        "tiers12": TIER12_PRE + ["action_payment_plan", "promised_to_pay", "spoke_with_agent"],
        "current": CURRENT_PRE + CALL,
        "all": PRE_ALL + CALL,
    },
}
NEVER_USE = {"paid_30d", "paid_after_call_30d", "paid_amount_30d", "days_to_first_payment", "paid_fraction_30d",
             "paid_on_call", "payment_on_call", "label_usable", "label_reliable", "window_complete", "data_type",
             "split", "post_call_row", "sentiment_source", "qa_source", "guarantor_id", "call_id", "snapshot_id",
             "as_of", "agent", "tcn_record_id", "source", "ever_paid", "n_prior_contacts_90d", "n_prior_calls",
             "days_since_last_contact"}
assert not any(set(f) & NEVER_USE for s in SETS.values() for f in s.values()), "a forbidden column is in a feature set"
for c in PRE_ALL + CALL:
    m[c] = pd.to_numeric(m[c], errors="coerce")
SYN_TRAIN, SYN_TEST, REAL = (d.assign(**{c: pd.to_numeric(d[c], errors="coerce") for c in PRE_ALL + CALL})
                             for d in (SYN_TRAIN, SYN_TEST, REAL))

# ---------- preprocessing (put it INSIDE your pipeline, so it is fitted on training rows only) ----------
SKEWED = ["n_accounts", "n_hospital_accounts", "n_physician_accounts", "placement_balance", "balance_at_t",
          "days_since_first_placement", "days_since_last_placement", "days_since_last_discharge", "n_payments_90d",
          "amount_paid_90d", "n_payments_365d", "amount_paid_365d", "n_payments_ever", "amount_paid_ever",
          "days_since_last_payment", "n_insurance_payments_365d", "amount_insurance_365d", "n_adjustments_365d",
          "amount_adjustments_365d", "last_payment_amount", "n_actions", "talk_seconds"]

class Prep(BaseEstimator, TransformerMixin):
    """days_since_last_payment is empty when the patient never paid: fill with (train max + 1) and add a 0/1
    'never_paid' column. Any other gap -> 0. Optional log1p on skewed non-negative columns."""
    def __init__(self, log=True):
        self.log = log
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.has_dslp_ = "days_since_last_payment" in X
        self.fill_ = (np.nanmax(X["days_since_last_payment"].to_numpy()) + 1) if self.has_dslp_ else None
        self.cols_ = list(X.columns) + (["never_paid"] if self.has_dslp_ else [])
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.has_dslp_:
            X["never_paid"] = X["days_since_last_payment"].isna().astype(float)
            X["days_since_last_payment"] = X["days_since_last_payment"].fillna(self.fill_)
        X = X.fillna(0.0)
        if self.log:
            for c in X.columns:
                if c in SKEWED:
                    X[c] = np.log1p(X[c].clip(lower=0))
        return X[self.cols_]
    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_)

# ---------- folds: everyone uses exactly these ----------
def folds(df, target, k=5, seed=SEED):
    """Grouped by patient: all calls of one patient are in the same fold."""
    y = df[TARGET[target]].astype(int)
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(df, y, df["guarantor_id"]))

# ---------- metrics ----------
def calibration(y, p):
    """Slope and intercept of the logistic recalibration (ideal: 1 and 0)."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def metrics(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    top = np.argsort(-p)[: max(1, int(round(0.2 * len(p))))]
    slope, icpt = calibration(y, p)
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "calib_slope": slope, "calib_intercept": icpt,
            "top20_capture": y[top].sum() / max(y.sum(), 1), "mean_pred": p.mean(), "pay_rate": y.mean()}

def boot_ci(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); y, p = np.asarray(y).astype(int), np.asarray(p, float); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def fit(model, X, y, w=None):
    """Fit with sample weights when the final step supports them (pipelines: last step must be named 'model')."""
    if w is None:
        return model.fit(X, y)
    try:
        return model.fit(X, y, model__sample_weight=w)
    except (TypeError, ValueError):
        return model.fit(X, y)          # estimator without sample_weight: real rows count once each

# ---------- 1) synthetic cross-validation: the main way to compare and choose ----------
def cv_eval(make_model, feats, target):
    """Grouped 5-fold CV on the synthetic training rows. Returns mean metrics, their SE, and out-of-fold predictions."""
    df = rows(SYN_TRAIN, target); y = df[TARGET[target]].astype(int).to_numpy()
    oof = np.full(len(df), np.nan); per = []; t0 = time.time()
    for tr, te in folds(df, target):
        mdl = fit(make_model(), df.iloc[tr][feats], y[tr])
        oof[te] = mdl.predict_proba(df.iloc[te][feats])[:, 1]
        per.append(metrics(y[te], oof[te]))
    per = pd.DataFrame(per)
    out = {f"cv_{k}": v for k, v in per.mean().items()}
    out["cv_auc_se"] = per["auc"].std(ddof=1) / np.sqrt(len(per))
    out["fit_seconds"] = round(time.time() - t0, 1)
    return out, pd.DataFrame({"call_id": df["call_id"].to_numpy(), "y": y, "p": oof})

# ---------- 2) real rows, measured fairly: repeated grouped cross-validation over the real patients ----------
REAL_LOOKS = {"n": 0}
def real_eval(make_model, feats, target, real_weight=0.0, repeats=10):
    """Score the real rows fairly. The real patients are split into 5 stratified groups (all calls of a patient
    together); each group is scored by a model trained on synthetic data + the OTHER real groups (each real row with
    weight real_weight; 0 = synthetic only). Repeated with 10 different splits. No real row is ever scored by a model
    that saw it.
    real_auc = AUC computed WITHIN each held-out group, averaged over the 50 groups. (Pooling the groups' predictions
    is not used for AUC: many real rows have identical features, and tiny differences between the groups' models
    would rank them at random. Leave-one-patient-out is not used either: it biases AUC downwards.)
    With real_weight = 0 there is one model, so the single-model AUC and its 95% bootstrap interval are also given.
    Every call counts as one 'look' at the real data."""
    REAL_LOOKS["n"] += 1
    syn = rows(SYN_TRAIN, target); real = rows(REAL, target).reset_index(drop=True)
    ys, yr = syn[TARGET[target]].astype(int).to_numpy(), real[TARGET[target]].astype(int).to_numpy()
    base = fit(make_model(), syn[feats], ys) if real_weight == 0 else None
    within, preds = [], []
    for r in range(repeats):
        p_r = np.full(len(real), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(real, yr, real["guarantor_id"]):
            if base is not None:
                mdl = base
            else:
                X = pd.concat([syn[feats], real.iloc[tr][feats]]); y = np.r_[ys, yr[tr]]
                w = np.r_[np.ones(len(ys)), np.full(len(tr), float(real_weight))]
                mdl = fit(make_model(), X, y, w)
            p_r[te] = mdl.predict_proba(real.iloc[te][feats])[:, 1]
            if yr[te].min() != yr[te].max():
                within.append(roc_auc_score(yr[te], p_r[te]))
        preds.append(p_r)
    p = np.mean(preds, axis=0)
    met = metrics(yr, p)
    out = {"real_auc": float(np.mean(within)), "real_auc_folds_sd": float(np.std(within, ddof=1))}
    out.update({f"real_{k}": v for k, v in met.items() if k != "auc"})
    if base is not None:   # one model for every real row: a plain AUC with a bootstrap interval is valid
        lo, hi = boot_ci(yr, p)
        out.update({"real_auc_single_model": met["auc"], "real_auc_lo": lo, "real_auc_hi": hi})
    out.update({"real_n": len(yr), "real_weight": real_weight, "real_looks_so_far": REAL_LOOKS["n"]})
    return out, pd.DataFrame({"call_id": real["call_id"].to_numpy(), "y": yr, "p": p})

# ---------- baseline everyone must beat: the rule "paid before, most recent first" ----------
def rule_scores(df):
    """Patients who paid before first, the most recent payers first. No training involved."""
    paid = df["days_since_last_payment"].notna().astype(float)
    return paid * 100000 - df["days_since_last_payment"].fillna(0)

def rule_real_auc(target, repeats=10):
    """The rule scored on the real rows exactly like real_eval (within-fold AUC, same splits): the number to beat."""
    real = rows(REAL, target).reset_index(drop=True); yr = real[TARGET[target]].astype(int).to_numpy()
    s = rule_scores(real).to_numpy(); within = []
    for r in range(repeats):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(real, yr, real["guarantor_id"]):
            if yr[te].min() != yr[te].max():
                within.append(roc_auc_score(yr[te], s[te]))
    return float(np.mean(within))

# ---------- 3) final: fit once, score the synthetic TEST split once, save predictions ----------
def final_model(make_model, feats, target, real_weight=0.0, variant="", set_name="", params=None, notes=""):
    """Call this ONCE per target, for your chosen model. Fits on synthetic train (+ all real rows with real_weight),
    scores the synthetic test split, re-measures real fairly, logs a 'FINAL' row and saves predictions:
    synthetic out-of-fold + test predictions -> OUT (committed); real predictions -> PRIVATE (never committed)."""
    syn, test = rows(SYN_TRAIN, target), rows(SYN_TEST, target)
    ys, yt = syn[TARGET[target]].astype(int).to_numpy(), test[TARGET[target]].astype(int).to_numpy()
    if real_weight > 0:
        real = rows(REAL, target)
        X = pd.concat([syn[feats], real[feats]]); y = np.r_[ys, real[TARGET[target]].astype(int).to_numpy()]
        w = np.r_[np.ones(len(ys)), np.full(len(real), float(real_weight))]
        mdl = fit(make_model(), X, y, w)
    else:
        mdl = fit(make_model(), syn[feats], ys)
    p_test = mdl.predict_proba(test[feats])[:, 1]
    st = {f"test_{k}": v for k, v in metrics(yt, p_test).items()}
    st["test_auc_lo"], st["test_auc_hi"] = boot_ci(yt, p_test)
    st["test_rule_auc"] = roc_auc_score(yt, rule_scores(test))
    cv, oof = cv_eval(make_model, feats, target)
    real_m, real_p = real_eval(make_model, feats, target, real_weight)
    real_m["real_rule_auc"] = rule_real_auc(target)
    log_result("FINAL " + variant, target, set_name, feats, params or {}, cv, real_m, st, notes)
    pd.concat([oof.assign(kind="synthetic_oof"), pd.DataFrame({"call_id": test["call_id"].to_numpy(), "y": yt,
               "p": p_test, "kind": "synthetic_test"})]).assign(family=FAMILY, target=target, variant=variant,
               feature_set=set_name).to_csv(OUT / f"synthetic_predictions_{target}.csv", index=False)
    real_p.assign(family=FAMILY, target=target, variant=variant, feature_set=set_name).to_csv(
        PRIVATE / f"real_predictions_{target}.csv", index=False)
    return mdl, {**cv, **real_m, **st}

# ---------- logging: one row per experiment ----------
RESULTS = OUT / "results.csv"
def log_result(variant, target, set_name, feats, params, cv, real=None, synthetic_test=None, notes=""):
    row = {"family": FAMILY, "variant": variant, "target": target, "feature_set": set_name, "n_features": len(feats),
           "params": json.dumps(params, default=str), **cv, **(real or {}), **(synthetic_test or {}), "notes": notes,
           "logged_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    df = pd.DataFrame([row])
    df.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
    return row

print(f"synthetic train: pre {len(rows(SYN_TRAIN, 'pre')):,} / post {len(rows(SYN_TRAIN, 'post')):,} rows; "
      f"synthetic test: pre {len(rows(SYN_TEST, 'pre')):,} / post {len(rows(SYN_TEST, 'post')):,}; "
      f"real: pre {len(rows(REAL, 'pre'))} / post {len(rows(REAL, 'post'))}")
# =========================================== END OF SHARED SETUP ===========================================
```

Typical use:

```python
from sklearn.pipeline import Pipeline
target, set_name = "pre", "tiers12"
feats = SETS[target][set_name]
make = lambda: Pipeline([("prep", Prep()), ("model", ...)])
cv, oof = cv_eval(make, feats, target)                       # main measure
real, _ = real_eval(make, feats, target, real_weight=0)      # optional second opinion (counts as a look)
log_result("my variant", target, set_name, feats, {"param": 1}, cv, real)
```

## 8 · What to try for logistic regression (L1 / L2 / elastic net)

Logistic regression is the reference model: simple, fast and explainable. Earlier work found it level with
gradient boosting on this data, so it has a real chance of being the final choice.

**Variants (each with `Prep()` + `StandardScaler()` before the model unless stated):**

| Variant | Model | Settings to tune |
|---|---|---|
| no penalty | `LogisticRegression(C=1e6, max_iter=5000)` | — |
| L2 (ridge) | `LogisticRegression(penalty="l2", C=C, max_iter=5000)` | `C` in `np.logspace(-4, 2, 13)` |
| L1 (lasso) | `LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=5000, random_state=SEED)` | `C` as above |
| elastic net | `LogisticRegression(penalty="elasticnet", C=C, l1_ratio=r, solver="saga", max_iter=5000, random_state=SEED)` | `C` as above, `r` in 0.2 / 0.5 / 0.8 |
| scaling | the best of the above with `Prep(log=False)` + `StandardScaler()`, and with `Prep(log=False)` + `QuantileTransformer(output_distribution="normal", random_state=SEED)` | shows whether the log step matters |
| curves (GAM-like) | `Prep()` → `SplineTransformer(n_knots=4, degree=3)` → `StandardScaler()` → L2 | `n_knots` 3–6, `C` | lets `days_since_last_payment` and debt age bend |
| class weights | best variant with `class_weight="balanced"` | once; expected to change little |

**Notes**
- Always set `random_state=SEED` for `liblinear` and `saga`: without it, L1 results change between runs.
- Near-duplicate features (payment counts vs amounts) get weights of opposite sign under weak penalties; that's normal
  and cancels out. Don't interpret single weights of near-duplicates.
- For the final model per target, add a table of the standardized coefficients and odds ratios per one standard
  deviation (`np.exp(coef)`) to your report, and say which direction each feature pushes.
- Expected range (all features, honest CV): about 0.70–0.71 pre-call and 0.79–0.80 post-call. A big jump above that
  usually means a leak; check the feature list.

## 9 · Choosing and finishing (same for everyone)

1. **Screen:** every variant in section 8 with sensible default settings, on **all six sets × both targets**, with
   `cv_eval` (no real data). Log every run.
2. **Tune:** for each target, take the 2–3 best sets (by `cv_auc`) and tune the settings listed in section 8 (grid or
   random search; at most ~40 settings per set and target). Log every run.
3. **Use the real rows:** for your best 3–5 configurations per target, run `real_eval` with `real_weight` 0, 1, 5 and 20.
4. **Pick one final model per target:** among configurations whose `cv_auc` is within one SE (`cv_auc_se`) of your best,
   pick the one with the highest `real_auc`; if real AUCs are within 0.02 of each other, pick the simpler one (fewer
   features, shallower, stronger regularization). Write down why.
5. **Finish:** call `final_model(...)` **once per target** with that choice. It scores the synthetic test split once,
   re-measures real data fairly, logs a `FINAL …` row and saves the prediction files. If you change your mind after
   seeing the test score, say so in the report (and how many times you looked).

## 10 · What to compare against

| | Pre-call | Post-call |
|---|---|---|
| Rule baseline, real (`rule_real_auc`) | 0.716 | 0.720 |
| Rule baseline, synthetic test | 0.677 | 0.677 |
| Logistic regression, all features, honest CV | ~0.707 | ~0.797 |

## 11 · Deliverables (exact files)

Commit on branch `task/logistic_regression`:

| File | Content |
|---|---|
| `Phase 2/notebooks/logistic_regression/logistic_regression.ipynb` | your work; runs top to bottom on a fresh kernel; outputs stripped by nbstripout |
| `Phase 2/notebooks/logistic_regression/results.csv` | written by `log_result` / `final_model`; one row per experiment, two `FINAL` rows |
| `Phase 2/notebooks/logistic_regression/synthetic_predictions_pre.csv`, `…_post.csv` | written by `final_model` (synthetic rows only) |
| `Phase 2/notebooks/logistic_regression/report.md` | your report, using the template below |

**Do not commit:** `Phase 2/reports/` (the data and `Phase 2/reports/tasks/logistic_regression/real_predictions_*.csv`), model files,
anything with real rows. Send the `Phase 2/reports/tasks/logistic_regression/` folder to your lead privately.
**Do not change:** anything under `src/`, `tests/`, `configs/`, other people's folders, or the `Phase 2/Task/*.md` files.

Report template (`Phase 2/notebooks/logistic_regression/report.md`):

```markdown
# Logistic regression (L1 / L2 / elastic net) · results
## Final models
| Target | Variant | Feature set (n) | Real weight | CV AUC ± SE | Real AUC (within-group) | Synthetic test AUC (95% CI) | Calib. slope (test) | Top-20% capture (test) |
## Compared with the rule baseline
## What I tried (screening and tuning: sets, settings, how many runs)
## What worked, what didn't (with numbers)
## Use of real data (weights tried, number of real looks, what changed)
## Feature notes (which features your models relied on; anything surprising)
## Problems, doubts, open questions
## Environment (package versions; anything installed extra)
```

## 12 · Hand back

```bash
git add "Phase 2/notebooks/logistic_regression/" "Phase 2/notebooks/logistic_regression/"
git status            # check: no data files, nothing under Phase 2/reports/
git commit -m "Task 1: Logistic regression (L1 / L2 / elastic net) results"
git push -u origin task/logistic_regression
```

Then tell your lead the branch is pushed and send `Phase 2/reports/tasks/logistic_regression/` privately.

## 13 · Checklist before you hand back

- [ ] Data check in section 3 matched (shape, fingerprint, counts).
- [ ] Both targets done, all six sets screened, the best ones tuned.
- [ ] Every experiment logged in `results.csv`; exactly two `FINAL` rows (pre and post).
- [ ] Real data measured only with `real_eval`; number of looks reported.
- [ ] Synthetic test split used only through `final_model`.
- [ ] Notebook runs top to bottom; outputs stripped in the commit (`git show --stat` shows a small notebook).
- [ ] No real rows, IDs or real predictions in the commit, the report or any chat.
- [ ] Report written with the template.
