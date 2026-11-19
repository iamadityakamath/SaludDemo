# Feature selection for the propensity models

Which features matter for the **pre-call** and **post-call** models, how that was decided, and how far each result can be
trusted. Source: `notebooks/13_regularization_feature_importance.ipynb` (run 2026-11-18; three complete runs gave identical
feature sets). Saved feature sets: `reports/phase2/experiments/feature_sets_master.json` (rebuilt by running the notebook).

## In short

- **Payment history decides almost everything.** Two features are important by every method, in every fold and under
  every random seed, and are confirmed in Salud's real calls: `n_payments_90d` (more recent payments → more likely to pay)
  and `days_since_last_payment` (longer ago → less likely).
- **After a call, two call fields join them:** `action_payment_plan` (a plan set up on the call) and `promised_to_pay`.
- **Debt age** (`days_since_first_placement`, `days_since_last_placement`) and **one more payment-history field** add a
  little; the rest (account counts, balances, insurance, adjustments, discharge, sentiment, QA, most call actions) add
  little or nothing.
- **Small sets are close to, but not as good as, all features.** Measured honestly (nested), the top 5 loses about
  **0.015 AUC** against all candidates (pre-call 0.692 vs 0.707; post-call 0.782 vs 0.797).

## Data

| | Pre-call model | Post-call model |
|---|---|---|
| File | `reports/phase2/pre_training_data.xlsx` (master file) | same |
| Rows used | synthetic `split = train`, `label_usable`: **6,647 calls, 1,477 patients** | + `post_call_row` (no payment on the call): **5,368 calls, 1,421 patients** |
| Target | `paid_30d` (pay rate 0.57) | `paid_after_call_30d` (0.46) |
| Candidates | **21** history fields | the 21 + **21** call fields = **42** |
| Real rows (direction check only) | 58 | 47 |

The synthetic **test** split and the **real** rows were not used to choose anything.

### Candidates and what was left out

| Group | Fields |
|---|---|
| Accounts and balance | `n_accounts`, `n_hospital_accounts`, `n_physician_accounts`, `placement_balance`, `balance_at_t` |
| Debt age | `days_since_first_placement`, `days_since_last_placement`, `days_since_last_discharge` |
| Payment history | `n_payments_90d`, `n_payments_365d`, `n_payments_ever`, `amount_paid_90d`, `amount_paid_365d`, `amount_paid_ever`, `days_since_last_payment`, `paid_share_so_far` (new), `last_payment_amount` (new) |
| Insurance and adjustments | `n_insurance_payments_365d`, `amount_insurance_365d`, `n_adjustments_365d`, `amount_adjustments_365d` |
| Call (post-call only) | 16 outcome fields (`inbound`, `spoke_with_agent`, `collection_attempted`, `promised_to_pay`, `non_committal`, `contested_or_refused`, `pursued_insurance`, `transferred`, 5 `action_*` flags, `n_actions`, `asked_financial_assistance`, `talk_seconds`), 4 sentiment fields, `qa_score` |

| Left out | Why |
|---|---|
| `paid_amount_30d`, `days_to_first_payment`, `paid_fraction_30d`, `paid_on_call` | contain the answer |
| ids, dates, `data_type`, `split`, `label_usable`, `post_call_row`, `*_source` | describe the row, not the patient |
| `n_prior_contacts_90d`, `n_prior_calls`, `days_since_last_contact` | Salud's data has no contact history |
| `ever_paid` | exact copy of the automatic "never paid" flag added with `days_since_last_payment` |

The two new fields come from Salud's own data: `paid_share_so_far` = patient payments since placement ÷ placement balance;
`last_payment_amount` = amount of the most recent patient payment before the call (0 if none). Both were checked against
the raw tables by `salud-propensity verify` (0 mismatches).

## Method

| Step | What | Why |
|---|---|---|
| Preprocessing (inside each fold) | never-paid fill + flag, log1p for skewed fields, standardize | comparable weights; nothing learned from the test fold |
| Penalty strength | grouped 5-fold CV (by patient) over C = 10⁻⁴…10²; L1 by the one-SE rule, L2 at its best | no choice at the grid edge |
| L1 / L2 / elastic net | weights and coefficient paths | which features enter first |
| Stability selection | L1 on 100 half-samples of patients | L1 picks between near-duplicates at random |
| Permutation importance | logistic regression **and** LightGBM, on held-out folds | importance a linear model misses (curved effects) |
| Real-data direction | each feature alone, 95% bootstrap interval | effects that exist only in the synthetic data |
| Combined rank | mean of L1 entry order, stability, permutation LR, permutation LightGBM | no single method decides |
| Nested selection | the selection redone inside each fold, then scored on the held-out fold | honest accuracy for chosen sets |
| Seed check | the ranking with seeds 7, 123, 2024 | answers that depend on chance |

## Results

### Feature tiers

| Tier | Pre-call | Post-call adds | Evidence |
|---|---|---|---|
| **1 · always keep** | `n_payments_90d`, `days_since_last_payment` | `action_payment_plan`, `promised_to_pay` | stability 1.00; first in L1 entry order; top of LR or LightGBM permutation; in the nested top 5 in 5/5 folds; top 5 under every seed; payment fields confirmed in real data |
| **2 · useful** | payment history, **one or two of**: `n_payments_365d`, `paid_share_so_far`, `last_payment_amount` (all 0.86–0.98 alike); debt age: `days_since_first_placement`, `days_since_last_placement` | `spoke_with_agent`, `days_since_first_placement` | stability 0.30–0.42 pre-call (sharing the role with twins); post-call `days_since_first_placement` 0.95, `spoke_with_agent` 0.78; LightGBM uses `days_since_last_placement` and `last_payment_amount`; payment fields confirmed in real data, debt age not (interval includes 0.5) |
| **3 · little or no value** | account counts, balances, insurance, adjustments, days since discharge; amounts paid and `n_payments_ever` (duplicates of the counts) | sentiment, QA, transfers, other actions, talk time, inbound, collection attempted | stability ≤ 0.26 (pre-call) / ≤ 0.39 (post-call, apart from `action_shared_balance` 0.69); near-zero permutation importance; none confirmed in real data |

Two borderline cases:
- `amount_adjustments_365d` ranks 4th pre-call and stays in the top 5 under all three other seeds, because the linear
  model's permutation importance leans on it (0.020). But L1 keeps it in only 18% of half-samples, LightGBM barely uses
  it (0.007), and real data does not confirm it (0.57, 0.47–0.68). Treated as tier 3; it is in `top5` and `l1_one_se`, so
  the model notebooks will show whether it helps.
- `action_shared_balance` is fairly stable post-call (0.69) but points weakly the other way in the 47 real calls
  (0.44 vs 0.56 synthetic). Treated as tier 3 until more real calls exist.

### Pre-call: every feature

Stability = share of 100 half-samples where L1 keeps it. Permutation = AUC lost when shuffled. Real = the feature alone
(> 0.5: higher value → pays more), with its 95% interval.

| Feature | Combined rank | Stability | Perm. LR | Perm. LightGBM | Synthetic alone | Real alone (95% CI) | Tier |
|---|---|---|---|---|---|---|---|
| `n_payments_90d` | 1 | 1.00 | 0.267 | 0.014 | 0.69 | 0.72 (0.62–0.83) | 1 |
| `days_since_last_payment` | 2 | 1.00 | 0.015 | 0.103 | 0.32 | 0.28 (0.17–0.39) | 1 |
| `n_payments_365d` | 3 | 0.42 | 0.041 | 0.004 | 0.68 | 0.73 (0.63–0.83) | 2 |
| `amount_adjustments_365d` | 4 | 0.18 | 0.020 | 0.007 | 0.52 | 0.57 (0.47–0.68) | 3 |
| `days_since_first_placement` | 5 | 0.30 | 0.004 | 0.002 | 0.59 | 0.64 (0.48–0.78) | 2 |
| `days_since_last_placement` | 6 | 0.10 | 0.007 | 0.012 | 0.52 | 0.62 (0.47–0.76) | 2 |
| `paid_share_so_far` (new) | 7 | 0.35 | 0.008 | 0.000 | 0.66 | 0.72 (0.61–0.83) | 2 |
| `n_physician_accounts` | 8 | 0.26 | 0.001 | 0.002 | 0.55 | 0.46 (0.32–0.61) | 3 |
| `amount_paid_365d` | 9 | 0.00 | 0.080 | 0.003 | 0.67 | 0.73 (0.63–0.83) | 3 (twin) |
| `n_accounts` | 10 | 0.02 | 0.009 | 0.004 | 0.57 | 0.48 (0.35–0.63) | 3 |
| `placement_balance` | 11 | 0.05 | 0.020 | −0.002 | 0.56 | 0.55 (0.41–0.69) | 3 |
| `n_adjustments_365d` | 12 | 0.00 | 0.018 | 0.002 | 0.53 | 0.57 (0.47–0.67) | 3 |
| `amount_insurance_365d` | 13 | 0.17 | 0.005 | 0.000 | 0.54 | 0.52 (0.44–0.61) | 3 |
| `days_since_last_discharge` | 14 | 0.03 | 0.001 | 0.002 | 0.54 | 0.55 (0.38–0.69) | 3 |
| `amount_paid_90d` | 15 | 0.00 | 0.016 | −0.001 | 0.67 | 0.73 (0.63–0.83) | 3 (twin) |
| `n_payments_ever` | 16 | 0.04 | 0.005 | 0.001 | 0.68 | 0.73 (0.63–0.83) | 3 (twin) |
| `last_payment_amount` (new) | 17 | 0.00 | 0.001 | 0.009 | 0.66 | 0.73 (0.63–0.83) | 2 (trees) |
| `amount_paid_ever` | 18 | 0.00 | 0.019 | 0.000 | 0.67 | 0.73 (0.63–0.83) | 3 (twin) |
| `n_insurance_payments_365d` | 19 | 0.04 | 0.000 | 0.001 | 0.54 | 0.53 (0.45–0.63) | 3 |
| `balance_at_t` | 20 | 0.02 | 0.015 | −0.002 | 0.53 | 0.52 (0.39–0.66) | 3 |
| `n_hospital_accounts` | 21 | 0.02 | 0.002 | 0.001 | 0.57 | 0.52 (0.39–0.68) | 3 |

"Twin" = a near-duplicate of a payment count; its information is already in tier 1–2.

### Post-call: top of the ranking

| Feature | Combined rank | Stability | Perm. LR | Perm. LightGBM | Tier |
|---|---|---|---|---|---|
| `n_payments_90d` | 1 | 1.00 | 0.196 | 0.010 | 1 |
| `days_since_last_payment` | 2 | 1.00 | 0.029 | 0.126 | 1 |
| `action_payment_plan` | 3 | 1.00 | 0.063 | 0.069 | 1 |
| `promised_to_pay` | 4 | 1.00 | 0.007 | 0.009 | 1 |
| `days_since_last_placement` | 5 | 0.26 | 0.007 | 0.013 | 2 |
| `days_since_first_placement` | 6 | 0.95 | 0.004 | 0.004 | 2 |
| `paid_share_so_far` | 7 | 0.21 | 0.007 | 0.001 | 2 |
| `amount_adjustments_365d` | 8 | 0.15 | 0.007 | 0.001 | 3 |
| `spoke_with_agent` | 9 | 0.78 | 0.004 | 0.000 | 2 |
| `n_payments_365d` | 10 | 0.09 | 0.002 | 0.005 | 2 |
| `action_shared_balance` | 13 | 0.69 | 0.001 | 0.000 | 3 |
| sentiment (4 fields), `qa_score` | 12–38 | ≤ 0.10 | ≤ 0.003 | ≤ 0.003 | 3 |

The full 42-row table is in the notebook (section 9).

### Accuracy (AUC, synthetic training rows, grouped 5-fold)

**Honest (nested: features chosen inside each fold):**

| | Pre-call LR | Pre-call LightGBM | Post-call LR | Post-call LightGBM |
|---|---|---|---|---|
| All candidates | 0.707 | 0.705 | 0.797 | 0.795 |
| L1 one-SE (≈ 13 pre / 9 post features) | 0.705 | 0.698 | 0.790 | 0.793 |
| Top 5 | 0.692 | 0.692 | 0.782 | 0.783 |

**Per saved set (optimistic: chosen and scored on the same rows; use for comparing sets, not for absolute level):**

| Set | Pre-call n | LR | LightGBM | Post-call n | LR | LightGBM |
|---|---|---|---|---|---|---|
| `stable` | 2 | 0.692 | 0.692 | 5 | 0.796 | 0.795 |
| `top5` | 5 | 0.706 | 0.692 | 5 | 0.791 | 0.792 |
| `l1_one_se` | 7 | 0.703 | 0.697 | 14 | 0.799 | 0.802 |
| `top10` | 10 | 0.703 | 0.702 | 10 | 0.798 | 0.800 |
| `current` | 11 | 0.700 | 0.697 | 32 | 0.795 | 0.790 |
| `top15` | 15 | 0.710 | 0.701 | 15 | 0.799 | 0.799 |
| `all` | 21 | 0.708 | 0.705 | 42 | 0.797 | 0.795 |

SE ≈ 0.01: most sets are within two SE of each other.

## The saved feature sets

| Set | Pre-call | Post-call |
|---|---|---|
| `stable` | `n_payments_90d`, `days_since_last_payment` | + `action_payment_plan`, `promised_to_pay`, `days_since_first_placement` |
| `top5` | `n_payments_90d`, `days_since_last_payment`, `n_payments_365d`, `amount_adjustments_365d`, `days_since_first_placement` | `n_payments_90d`, `days_since_last_payment`, `action_payment_plan`, `promised_to_pay`, `days_since_last_placement` |
| `l1_one_se` | top5 + `paid_share_so_far`, `n_physician_accounts` (7) | 14: tier 1 + debt age, `paid_share_so_far`, `amount_adjustments_365d`, `spoke_with_agent`, `n_payments_365d`, `action_shared_balance`, `action_financial_assistance`, `n_physician_accounts`, `action_statement_or_bill` |
| `top10` | top5 + `days_since_last_placement`, `paid_share_so_far`, `n_physician_accounts`, `amount_paid_365d`, `n_accounts` | top5 + `days_since_first_placement`, `paid_share_so_far`, `amount_adjustments_365d`, `spoke_with_agent`, `n_payments_365d` |
| `top15` | 15 by combined rank | 15 by combined rank |
| `current` | the 11 fields the current models use | the 32 fields the current models use |
| `elastic_net` | all 21 (the tuned elastic net keeps every pre-call feature) | 34 |
| `all` | 21 | 42 |

## How the results were checked

- 9 guard checks pass before anything runs (no answer-like or bookkeeping column; synthetic training rows only; label
  usable; no payment on the call in post-call rows; no exact duplicate or constant column).
- Two complete runs: identical output in every cell and an identical feature-set file.
- Three other random seeds: the top 5 keeps 4–5 of its features, the top 10 keeps 8–10; tier 1 is in the top 5 every time.
  The fifth pre-call place changes (`n_physician_accounts`, `n_adjustments_365d` or `paid_share_so_far`); post-call the
  fifth place is one of the two debt-age fields.
- Real data: no feature points clearly the opposite way; every payment-history field is confirmed.
- The training file itself: every field rebuilt from Salud's raw tables with separate code (`salud-propensity verify`,
  0 mismatches) and 26 integrity checks on every build.

## Mistakes found and corrected along the way

| First version | Problem | Now |
|---|---|---|
| "5 features are as good as all" | sets chosen and scored on the same rows | nested selection: top 5 loses ~0.015 |
| L2 at C = 10, then one-SE at C = 0.001 (grid edges) | twins got opposite-sign weights; at 0.001 every weight is "the feature alone" | wider grid, L2 at its best C, weight size not used for ranking |
| linear model only | missed curved effects | LightGBM permutation importance added |
| `ever_paid` included | exact copy of the never-paid flag | dropped; a guard checks it |
| L1 sets changed between runs (14 vs 7 features) | solvers shuffle internally | seeds fixed; runs identical |
| "insurance action" flag | counted "Explained Coinsurance/Deductible/Copay" | insurance work only (data audit) |

## Limits

- **The importance comes from synthetic data.** The world was tuned to match 55 statistics of Salud's 84 calls, and
  real data confirms the direction of the payment-history fields, but the size of each effect is the generator's.
- **Sentiment and QA cannot be judged here:** synthetic values were sampled from the call outcome, so they can only
  repeat it. Only real calls can show whether they add anything.
- **Payment history starts at placement with Salud.** Payments on the hospital's own (Epic) plans are not in Salud's
  data (reflections watch item 24); with them, payment history would probably matter even more.
- **Real checks rest on 58 / 47 rows**, so "not confirmed" means "not shown", not "no effect".

## Recommendation for the model notebooks

Train every model family on `stable`, `l1_one_se`, `current`, `top10` and `all` (`top5` and `top15` are optional;
`elastic_net` equals `all` pre-call). Compare on the synthetic **test** split and the **real** rows. Worth adding as a
hand-made set, since it follows the tiers rather than one method: **tiers 1 + 2** — pre-call `n_payments_90d`,
`days_since_last_payment`, `n_payments_365d`, `paid_share_so_far`, `last_payment_amount`, `days_since_first_placement`,
`days_since_last_placement`; post-call these plus `action_payment_plan`, `promised_to_pay`, `spoke_with_agent`.
It has not been scored yet.
