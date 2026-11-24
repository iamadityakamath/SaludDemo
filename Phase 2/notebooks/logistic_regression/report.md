# Logistic regression (L1 / L2 / elastic net) · results

## Final models

| Target | Variant | Feature set (n) | Real weight | CV AUC ± SE | Real AUC (within-group) | Synthetic test AUC (95% CI) | Calib. slope (test) | Top-20% capture (test) |
|---|---|---|---|---|---|---|---|---|
| pre (`paid_30d`) | L2 + cubic splines (3 knots), C = 0.01 | `all` (21) | 1 | 0.7174 ± 0.0104 | 0.873 | 0.700 (0.676–0.724) | 1.02 | 0.328 |
| post (`paid_after_call_30d`) | Plain L2, C = 0.0316 | `tiers12` (10) | 0 | 0.7983 ± 0.0070 | 0.757 | 0.779 (0.753–0.803) | 1.09 | 0.399 |

Other test metrics: pre PR-AUC 0.790, Brier 0.208, log loss 0.595; post PR-AUC 0.803, Brier 0.178, log loss 0.531.

## Compared with the rule baseline

| Target | Rule AUC, real | Model AUC, real | Rule AUC, synthetic test | Model AUC, synthetic test |
|---|---|---|---|---|
| pre | 0.716 | 0.873 (+0.157) | 0.677 | 0.700 (+0.023) |
| post | 0.720 | 0.757 (+0.037) | 0.677 | 0.779 (+0.102) |

- Both models beat the rule on both checks.
- The post-call gain on the test split is large and clear.
- The pre-call gain on the test split is small: the lower CI bound (0.676) only just reaches the rule's 0.677.
- The big pre-call gain on real data comes from only 58 rows, so treat it with caution (see Problems).

## What I tried (screening and tuning: sets, settings, how many runs)

- **Screening: 8 variants × 6 feature sets × 2 targets = 96 runs.** All used grouped 5-fold CV on `guarantor_id` and default C = 1.
  - The 8 variants: no penalty; L2; L1; elastic net (r = 0.5); L2 without the log transform; L2 without log, with a quantile → normal transform; L2 + cubic splines (5 knots); L2 with balanced class weights.
- **Tuning: 41 settings × 3 sets × 2 targets = 246 runs.**
  - Pre-call sets: `tiers12`, `all`, `current`. Post-call sets: `tiers12`, `stable`, `l1_one_se`.
  - L2 over C = 10⁻⁴ … 10² (13 values).
  - L1 over 7 of those C values.
  - Elastic net with r ∈ {0.2, 0.5, 0.8} × C ∈ {0.01, 0.1, 1}.
  - L2 + splines with 3–6 knots × C ∈ {0.01, 0.1, 1}.
- **Real-data check: 4 candidates per target × real weights {0, 1, 5, 20} = 32 runs,** all through `real_eval`.
- **Final:** one `final_model` call per target.
- `results.csv` holds all 376 rows: 96 + 246 + 32 + 2 FINAL.

## What worked, what didn't (with numbers)

- **Splines were the only change that clearly helped.** Best CV AUC per variant after tuning:

  | Variant | Pre | Post |
  |---|---|---|
  | L2 + splines | **0.7174** | **0.8062** |
  | L2 | 0.7077 | 0.7985 |
  | Elastic net | 0.7075 | 0.7985 |
  | L1 | 0.7076 | 0.7983 |

- **The choice of penalty barely matters.** No penalty, L1, L2 and elastic net all land within 0.001 in screening. The features are few and the data is large, so regularization does little.
- **The log transform helps a little.** Removing it cost 0.003 post-call. Adding the quantile transform on top cost another 0.001–0.009.
- **Balanced class weights made no difference to AUC.** They were dropped so the probabilities stay calibrated.
- **Feature sets (best CV in screening):**
  - Pre: `tiers12` 0.712, `all` 0.711, `current` 0.710, `top5` 0.701, `l1_one_se` 0.701, `stable` 0.692.
  - Post: `tiers12` 0.804, `stable` 0.801, `l1_one_se` 0.800, `top5` 0.799, `all` 0.797, `current` 0.796.
- **Pre-call real check:** splines on `all` scored 0.852 / 0.873 / 0.872 / 0.843 at real weights 0 / 1 / 5 / 20. Plain L2 on `all` scored only 0.73–0.76. That decided splines for pre-call. Weight 1 was best; weight 20 overfit the 58 real rows.
- **Post-call real check:**
  - Splines on `tiers12` scored 0.765 / 0.747 / 0.771 / 0.743 at the four weights.
  - Plain L2 on `tiers12` scored 0.757 / 0.755 / 0.747 / 0.739.
  - The splines' CV gain (+0.008) was inside one SE of plain L2, and the real AUCs differ by less than 0.02. Under the selection rule ("within 0.02, take the simpler one") that means plain L2 at weight 0.

## Use of real data (weights tried, number of real looks, what changed)

- **Weights tried:** 0, 1, 5 and 20 for every candidate.
- **Real looks:** 34 per notebook run. That is 32 in the real-data check plus 1 inside each `final_model` call (the counter reads 33 → 34).
- **Reruns:** I ran the notebook top to bottom twice. The real-data check repeated with the same fixed seeds, and the second run gave identical numbers, so it did not influence any choice. `final_model` (the synthetic test split) was scored in the final run only.
- **What the real data changed:**
  - Pre-call: it confirmed splines over plain L2, and chose weight 1 over 0.
  - Post-call: it changed nothing. CV already preferred the simple L2 model, and real data didn't overturn that.
- **No real rows, IDs or real predictions appear in this report or the commit.** The real predictions are in `Phase 2/reports/tasks/logistic_regression/`, which is sent privately.

## Feature notes (which features your models relied on; anything surprising)

All of these are computed on synthetic training data.

**Post-call (plain L2): odds ratio per 1 SD increase, after preprocessing**

| Feature | OR | | Feature | OR |
|---|---|---|---|---|
| action_payment_plan | 2.21 | | spoke_with_agent | 1.12 |
| n_payments_90d | 1.98 | | never_paid | 1.09 |
| days_since_last_payment | 0.59 | | last_payment_amount | 0.89 |
| n_payments_365d | 1.47 | | paid_share_so_far | 0.83 |
| promised_to_pay | 1.22 | | days_since_first_placement | 0.79 |
| | | | days_since_last_placement | 0.95 |

**Pre-call (splines): change in odds when one feature moves from its 10th to its 90th percentile, others held at their medians**

| Feature | Odds × |
|---|---|
| n_payments_90d | 3.76 |
| last_payment_amount | 1.55 |
| n_adjustments_365d | 1.53 |
| days_since_last_placement | 0.66 |
| amount_adjustments_365d | 0.59 |
| paid_share_so_far | 0.54 |
| days_since_last_payment | 0.37 |

- **Recent payment history drives both models.** More payments in the last 90 days raise the odds of payment; a longer gap since the last payment lowers them.
- **The call outcome matters most post-call.** A payment plan set up on the call is the strongest single effect, and a promise to pay adds a little.
- **Some signs look surprising:** `paid_share_so_far` < 1 in both models, and `last_payment_amount` < 1 post-call.
  - These features overlap with the payment counts, so the model splits the shared signal between them.
  - Treat them as adjustments, not as standalone effects.

## Problems, doubts, open questions

1. **Bug in the shared `log_result`.** It writes the CSV header only on the first call. Rows logged later by `real_eval` and `final_model` carry extra columns, which leaves `results.csv` unreadable: pandas reports "Expected 19 fields in line 344, saw 35".
   - My workaround, notebook only (the shared block is unchanged): a wrapper keeps every logged row in a list, and the last cell rewrites `results.csv` with all columns.
   - Other tasks will hit the same bug. It may be worth fixing in the shared block.
2. **The real sample is small (58 rows).**
   - The real AUC differences between candidates (±0.02–0.03) are within noise.
   - The pre-call real AUC (0.873) being much higher than test (0.700) most likely reflects this small sample, not a model that is better on real patients.
3. **Near-duplicate features** (payment counts over 90 and 365 days, paid share, last amount) make the individual coefficients unstable. Predictions are not affected.
4. **The pre-call model is only slightly better than the rule** on the synthetic test split (+0.023, with the lower CI bound at the rule's level). Pre-call signal looks limited for a linear model, even with splines.
5. **The test split scores about 0.02 below CV for both targets.** CV still falls inside both test CIs, so this does not look like leakage.

## Environment (package versions; anything installed extra)

- Python 3.13.11 (project venv, kernel "Salud (project venv)")
- scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, scipy 1.18.1
- Extra install: `nbstripout`, set up for the repo so notebook outputs are stripped on commit. No other packages were added.
