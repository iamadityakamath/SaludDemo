# Model evaluation log

Every propensity experiment, how it was evaluated, and its result. Source of truth for numbers:
`reports/phase2/experiments/results.csv` (one row per run × fold), shown by `modeling.leaderboard()`.
Updated after each notebook.

## Evaluation protocol

| Item | Choice | Why |
|---|---|---|
| Target (pre-call) | `paid_30d`: any patient payment from the as-of day to 30 days after | our choice, not Salud's (to confirm): brief says any payment counts; 30 days ≈ one statement / plan cycle; short enough to credit the call; most predictable window in the Salud-tuned world (notebook 09). Salud's data has no due or promised dates, so "paid on time" can't be the target |
| Target (post-call) | `paid_after_call_30d`, rows without a payment taken on the call | the call's outcome isn't known in advance |
| Training rows | **fixed-date design**: one row per synthetic patient with an open balance on 2026-04-01 (Salud-tuned world: 1,794; dev 1,435, holdout 359. Original world: 1,350) | the call design has almost no signal (oracle 0.58 vs 0.80) — notebook 00 §6 |
| Validation | repeated 5×5 stratified K-fold, grouped by patient (25 scores per run) | stable (SD of estimate ±0.001 vs ±0.03 for one split) |
| Final checks | synthetic holdout (20%, once) and the 58 real rows (once, bootstrap CI) | never used to choose |
| Primary metric | AUC (ranking) | the deliverable is a ranked call list |
| Also | Brier, log loss, PR-AUC, calibration slope / in-the-large, payers captured in top 20% | probabilities must be usable for tiers and expected dollars |
| Choosing | corrected resampled t-test (Nadeau–Bengio) + one-standard-error rule (simplest within 1 SE) | fold scores are correlated; prefer explainable models |
| Feature sets | `FS_ALL` 23 · `FS_PRUNED` 14 (duplicates removed) · `FS_TRANSFER` 11 (also no prior-contact features, which the real data lacks) | notebook 00 §3–4 |

## Next data: the master file (2026-11-18), quick check only

`reports/phase2/pre_training_data.xlsx` (`salud-propensity master`): one row per call, synthetic (master world, Salud's
time frame to 2026-11-18, 51/51 statistics) + Salud's 84 real calls. `split`: random 80/20 by patient for synthetic,
`real_test` for real. It fixes three training-data problems: no calls in the pre-call label window, pre-call patients
unlike the real ones, and no payment plans in post-call rows (reflections watch items 16, 17, 21).

Quick check (untuned, not logged to `results.csv`; train = synthetic `train`, `label_usable` rows):

| Model | Features | Synthetic test | Real (n) | Mean prediction vs actual (real) |
|---|---|---|---|---|
| Pre-call LR | 11 | 0.694 | 0.766 (58) | 0.48 vs 0.50 |
| Pre-call LightGBM (monotone) | 11 | 0.708 | 0.750 (58) | 0.55 vs 0.50 |
| Post-call LR | 32 | 0.793 | 0.696 (47) | 0.34 vs 0.36 |
| Post-call LightGBM | 32 | 0.793 | 0.765 (47) | 0.41 vs 0.36 |
| Rule B-03 | – | – | 0.718 (58) | – |

Real CIs are about ±0.15, so these are directions. The notebooks below still use the old files and must be re-run on the
master file before any of this replaces the current status.

## Current status (Salud-tuned world, 2026-11-17): the model now ranks real patients the right way, but does not beat the rule

**World:** `configs/synthetic_world_salud.json` (realism mode). In notebook 00 it matches **50/51** real statistics within their
95% bootstrap intervals; the only miss is the share of patients with an adjustment in the last 365 days (0.250 against an upper
bound of 0.242). It matches all 11 relationships (payment history about 0.67, days since last payment about 0.34, debt age positive).
Training: 1,435 fixed-date patients, pay rate 0.35.

**Chosen by cross-validation, before any test: GB-06**, monotone LightGBM on `FS_TRANSFER`. CV AUC 0.876 (SE 0.003), more
than one SE above SC-05 (0.871) and LR-12 (0.866). Sigmoid calibration lowers its Brier score from 0.132 to 0.119.

| One-time test | GB-06 (LightGBM) | SC-05 (scorecard) | LR-12 (L2) | Rule B-03 |
|---|---|---|---|---|
| Synthetic holdout (359 patients) | 0.829 (0.777–0.877) | 0.814 (0.760–0.865) | 0.802 (0.746–0.854) | – |
| **Real, 58 rows (Jul 1 2026)** | **0.697 (0.548–0.837)** | 0.631 (0.476–0.784) | 0.633 (0.470–0.789) | **0.718 (0.610–0.829)** |
| Real Brier, frozen → level recalibrated (LOO) | 0.287 → 0.219 | 0.267 → 0.223 | 0.253 → 0.206 | – |
| Real pay rate in the lowest / middle / highest third (GB-06) | **35% / 42% / 74%**, the right order | | | |

Other real tests:
- the real-only 2-feature LR (leave-one-out) reaches 0.465;
- the post-call model on 47 real calls reaches 0.704 with pre-call features and 0.680 with call outcome and sentiment added.

**Real QA (Stage 1, run 2026-11-18 on all 84 real calls, prompt assist-v2 via Groq; `salud-propensity qa`):**

| Post-call test on 47 real calls (no payment on the call) | AUC (95% CI) |
|---|---|
| pre-call features | 0.704 (0.525–0.859) |
| + call outcome + sentiment | 0.680 (0.478–0.859) |
| + the real QA score | 0.684 (0.480–0.861) |
| the real QA score alone, no model | 0.594 (0.422–0.753) |

The mean QA score is 0.54 for calls followed by payment and 0.49 for the rest. Call quality shows no measurable link to payment, on synthetic data
(notebook 10) or on real data, but the real verdict rests on only 47 rows.

**Caveat: this is not an independent test.** The world was tuned to these same 84 calls, so 0.697 shows that the world and the
model are consistent with Salud's data. It does not show that the model generalises. Only new Salud calls can test that.

**Leaderboard (cross-validation, 86 runs):**
- the best is RF-05 at 0.887; at most 0.015 separates the top run of every family (LR 0.875, scorecard 0.872, tree 0.877, GB 0.883, ensemble 0.880, SVM 0.872);
- the rule B-03 reaches 0.843 and the best single feature 0.848, so most of the signal is payment history;
- six features (RF-04, GB-05) match the full models.

The "oracle" (hidden traits) now reaches only 0.60, because payment is driven by the observable payment habit. It is no longer a ceiling.

**Recommendation:**
- use "paid before, most recent first" as the baseline call order, with the GB-06 score (level-recalibrated) shown next to it;
- adopt the model once new real calls show it beating the rule;
- the scores file (`reports/phase2/propensity_scores.xlsx`, 46 patients, 5 tiers) states this caveat.

**Original world:** its local files were deleted on 2026-11-18 (clean-up); regenerate with `salud-synth --config configs/synthetic_world.json`. Its results are kept here: all sections below
this line are **original-world** numbers.

---

## Original world: the synthetic-trained model did NOT transfer to real data

**Chosen by cross-validation (before any test): SC-05**, the monotone WoE scorecard on `FS_TRANSFER` — AUC 0.687
(SE 0.008), ≈ 9 features, the simplest model within one SE of the best.

| One-time test | SC-05 (scorecard) | LR-12 (L2) | GB-06 (LightGBM) | Rule B-03 (paid before, most recent first) |
|---|---|---|---|---|
| Synthetic holdout (270 patients) | 0.639 (0.570–0.703) | 0.624 (0.555–0.691) | 0.618 (0.548–0.684) | – |
| **Real, 58 rows from Jul 1 2026** | **0.448 (0.294–0.610)** | 0.424 (0.272–0.582) | 0.399 (0.251–0.562) | **0.718 (0.610–0.829)** |
| Real Brier (frozen → level recalibrated, LOO) | 0.315 → 0.334 | 0.351 → 0.373 | 0.345 → 0.364 | – |
| Real pay rate, lowest / middle / highest third of scores (SC-05) | 60% / 47% / 42% — reversed | | | |

Other real tests: real-only 2-feature LR (leave-one-out) 0.465 (0.28–0.65); post-call model on 47 real calls
0.42–0.44.

**Why it fails:** direction of effect, synthetic vs real (single-feature AUC, > 0.5 = higher value → pays):
days since first / last placement 0.45 / 0.40 vs **0.64 / 0.62** (reversed); number of accounts 0.57 vs 0.48;
insurance payments 0.47 vs 0.53; adjustments 0.43 vs 0.57 (reversed); payments in 90 / 365 days 0.58 / 0.55 vs
**0.72 / 0.73** (same direction, much stronger in real).

**Recommendation now:** don't use the synthetic-trained scores (the scores file opens with a READ FIRST warning).
Interim ranking: a payment-history rule/scorecard. Next: more real data from Salud; re-tune the synthetic world to
the real relationships.

## Calibration (notebook 11, 25 folds)

| Model | none | Platt (sigmoid) | isotonic |
|---|---|---|---|
| SC-05 log loss / slope | 0.600 / 0.93 | 0.599 / 1.13 | 0.613 / 0.74 |
| LR-12 | 0.603 / 1.10 | 0.601 / 1.13 | 0.613 / 0.75 |
| GB-06 | 0.617 / 1.96 | 0.619 / 1.07 (AUC 0.656) | 0.629 / 0.70 |

## Data size, design, window (notebook 09)

| n training rows | 50 | 100 | 200 | 400 | 650 | 864 |
|---|---|---|---|---|---|---|
| Logistic regression | 0.615 | 0.656 | 0.679 | 0.677 | 0.688 | 0.690 |
| Scorecard | 0.587 | 0.643 | 0.677 | 0.664 | 0.682 | 0.686 |
| Extra trees | 0.565 | 0.617 | 0.656 | 0.653 | 0.671 | 0.677 |
| LightGBM (fixed settings) | 0.592 | 0.586 | 0.614 | 0.628 | 0.655 | 0.658 |
| Neural network | 0.460 | 0.449 | 0.456 | 0.501 | 0.573 | 0.563 |

Training on call rows instead of fixed-date rows (same patients): LR 0.615 vs 0.691; scorecard 0.591 vs 0.688.
Label window, **Salud-tuned world** (scorecard, fixed-date data rebuilt per window): **30 days 0.861** · 60 days 0.831 · 90 days 0.801.
Original world (the numbers above in this section are from it): 30 days 0.674 · 60 days 0.692 · 90 days 0.690, i.e. no clear winner there.
Real 58: 29 paid within 30 days; 12 on the call day, 4 in days 1–7, 7 in days 8–14, 6 in days 15–30 (median 7 days).

## Call quality (notebook 10, post-call target, synthetic)

| Step | LR AUC | Gain | p |
|---|---|---|---|
| CQ-01 pre-call only | 0.594 | | |
| CQ-02 + call outcome | 0.727 | +0.132 | < 0.001 |
| CQ-03 + sentiment | 0.742 | +0.015 | 0.010 |
| CQ-04 + QA score (clean) | 0.742 | +0.0004 | 0.67 |
| CQ-04n + QA score (AI-reviewer noise) | 0.742 | −0.0003 | 0.69 |
| CQ-05 QA + sentiment only | 0.666 | | |

LightGBM gives the same picture (0.576 → 0.719 → 0.738 → 0.737).

## Recommendation before the real test (after notebooks 00–08)

**SC-05 — monotone weight-of-evidence scorecard on `FS_TRANSFER`**: AUC 0.687 (SE 0.008), Brier 0.209,
calibration slope 0.93, ≈ 9 features. The simplest model within one standard error of the best (LR-10, 0.692);
readable as points; monotone.

## Best run per family (fixed-date design, pre-call target, 25 grouped folds)

| Family | Best run | AUC | SE | Brier | Cal. slope | Features | vs LR-12 (corrected t-test) |
|---|---|---|---|---|---|---|---|
| Oracle (hidden traits, ceiling) | B-05 | 0.801 | 0.004 | 0.174 | 1.01 | 3 | – |
| Logistic regression | LR-10 (L2, duplicates removed) | 0.692 | 0.008 | 0.209 | 1.06 | 15 | – |
| Logistic regression (transfer set) | LR-12 (L2 tuned, FS_TRANSFER) | 0.690 | 0.008 | 0.210 | 1.10 | 12 | reference |
| Ensemble | EN-04 bagged LR / EN-03 stacking | 0.691 / 0.689 | 0.008 | 0.209 / 0.211 | 1.08 / 1.25 | 12 / 11 | n.s. |
| Scorecard (WoE, smart bins) | SC-05 monotone, FS_TRANSFER | 0.687 | 0.008 | 0.209 | 0.93 | 9 | +0.003, p 0.74 |
| Naive Bayes | OT-03 | 0.682 | 0.006 | 0.304 | 0.26 | 12 | – |
| Random forest / extra trees | RF-03 extra trees tuned | 0.680 | 0.007 | 0.212 | 1.33 | 25 | +0.009, p 0.23 |
| Gradient boosting | GB-04 / GB-06 monotone LightGBM | 0.679 | 0.007 | 0.214 | 1.81 | 25 / 12 | +0.011, p 0.26 |
| k-NN | OT-01c log1p + z | 0.676 | 0.007 | 0.213 | 1.00 | 12 | – |
| SVM | OT-02 | 0.660 | 0.008 | 0.218 | 1.15 | 12 | – |
| Decision tree | DT-05 depth 5 + isotonic | 0.656 | 0.009 | 0.216 | 1.28 | 12 | DT-03: +0.045, p 0.002 |
| Neural network | OT-04 | 0.653 | 0.009 | 0.218 | 0.92 | 12 | OT-04b: +0.050, p 0.01 |
| Best rule | B-03 paid before, most recent first | 0.598 | 0.009 | – | – | – | – |

## What each experiment showed

| Question | Result |
|---|---|
| Normalisation (LR, no penalty) | z 0.684 · none 0.682 · log1p+z 0.678 · robust 0.678 · quantile 0.671 — small effect on ranking; tuned penalty fixes calibration (slope 0.7–0.8 → ≈ 1.05) |
| Normalisation (k-NN) | raw 0.576 → z 0.668 → log1p+z 0.676 — decisive for distance models |
| Regularisation | L2 0.686 · L1 0.685 · elastic net 0.688 — equal within noise |
| Number of features (L1 path) | 0 → 0.500 · 3 → 0.645 · 8 → 0.676 · 11–14 → 0.685–0.689 (plateau) · 22 → 0.685 |
| Train only on selected features | L1-then-refit 0.686 · RFECV 0.681 · MI top-k 0.674 · top-6 permutation (RF 0.648, GB 0.650) · stability (2.5 feats) 0.623 — never better than all features; aggressive selection hurts |
| Feature set | FS_TRANSFER 0.690 vs FS_ALL 0.686 — dropping features the real data lacks costs nothing |
| Splines / class weights | 0.679 / 0.689 (balanced weights worsen Brier 0.210 → 0.223) |
| WoE binning | quantile bins 0.62 → smart bins 0.686–0.691 |
| Boosting tuning | defaults 0.627–0.630 → tuned 0.674–0.679; monotone constraints cost nothing; tuned boosting under-confident (slope 1.7–2.0) |
| Ensembles | 0.678–0.691 — no gain over LR |

## Baselines and diagnostics

| Run | Description | AUC (mean ± SD, 25 folds) | Brier | Notebook |
|---|---|---|---|---|
| B-05 | Oracle ceiling: hidden traits (not observable) | 0.801 ± 0.022 | 0.174 | 01 |
| REF-LR | L2 logistic regression, log1p + z-score, FS_TRANSFER | 0.686 ± 0.036 | 0.210 | 01 |
| B-03 | Rule: paid before, most recent payment first | 0.598 ± 0.043 | (not a probability) | 01 |
| B-04 | Best single feature (picked inside each fold) | 0.592 ± 0.040 | – | 01 |
| B-02 | Rule: paid in the last year | 0.538 ± 0.025 | – | 01 |
| B-01 | Random | 0.500 ± 0.041 | – | 01 |

Design check (notebook 00 §6): call design LR 0.562 / oracle 0.583 · fixed design LR 0.686 / oracle 0.801.

Validation check (notebook 01): single 80/20 split SD 0.030 · 5-fold 0.003 · 10-fold 0.002 · repeated 5×5 0.001.
At n = 60: typical error 0.09 (5×5) to 0.17 (single split); leave-one-patient-out biased −0.07.

## Real-data test

See "Current status" at the top (notebook 11): failed for every synthetic-trained model.
