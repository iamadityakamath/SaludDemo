# Reflections

Decisions, the reasoning behind them, what went well, and what I'd do differently. One entry per task, newest
first. Kept up to date with every task.

---

## Model and data watch list (maintained; update whenever a new behaviour is found or an item changes)

Known behaviours of the models and data. Each item says what we saw, the evidence, what it affects, and what was
decided. **Status:** `accepted` = known and left as is on purpose; `open` = not yet acted on; `fixed` = resolved.
Real-data numbers come from very few rows, so treat them as directions, not measurements.

### Pre-call model (GB-06, monotone LightGBM on FS_TRANSFER, trained on the Salud-tuned synthetic world)

1. **New accounts score low ("the new-account dip").** `accepted` (user decision 2026-11-18: keep these scores as they are)
   - **Seen:** patients placed in the last 60 days who have never paid get a low score. The model learned
     "no payment history = unlikely to pay".
   - **Evidence (2026-11-18):** on the 58 real test rows, the 25 new, never-paid patients were predicted **13%** but **48%** paid.
     In the synthetic training data such patients pay only **19%** (130 of 1,794 patients). Within this group the model can't rank at all
     (real AUC 0.47, synthetic 0.62 on 15 rows).
   - **Impact:**
     - new placements will sit in the low tiers even though about half pay within 30 days;
     - only 1 of today's 46 scored real patients is new, so the current scores file is barely affected, but daily use would be;
     - new accounts are the largest group in the real test (25 of 58).
   - **If revisited:**
     - add "new, never paid → pay rate" to `salud-synth-calibrate` and retune the world;
     - add new-account signals (balance size, hospital or physician, insurance stage, days since placement);
     - or score new accounts with a simple rule.
2. **All of the ranking comes from "has paid before or not".** `open`
   - **Evidence:** on the real rows, ranking within each group is no better than chance (has paid before: 0.30 on 17 rows; older, never paid: 0.11 on 16 rows).
     The synthetic within-group AUCs are 0.86 for has paid before and 0.39 for older, never paid.
   - **Impact:** this explains why the model (real 0.70) doesn't beat the rule "paid before, most recent payment first" (0.72).
     The recommendation stays: the rule sets the call order, with the score shown beside it.
3. **Under-predicts Salud's overall pay rate.** `accepted`
   - **Evidence:** the model predicts about 25% on average against 50% actual on the real rows. Synthetic patients pay at 35% in the fixed design, against 50% for Salud's call rows.
   - **Handled by:** an intercept shift of +1.7 log-odds (leave-one-out), which takes the Brier score from 0.287 to 0.219.
   - **Likely cause (found 2026-11-18):** the training label has no calls in its window; the real one does (item 21).
   - **Update 2026-11-18:** trained on the master file, the mean prediction on the real rows is 0.48–0.55 against 0.50, with
     no shift (quick check). Confirm when the notebooks are re-run, then mark `fixed`.
   - **Impact:** the ranking is unaffected; the probabilities and expected dollars depend on the shift.
4. **Raw boosted-tree probabilities are too cautious.** `fixed`
   - **Evidence:** calibration slope 1.92.
   - **Handled by:** sigmoid calibration (Brier 0.132 → 0.119).
5. **Holdout scores lower than cross-validation.** `accepted`
   - **Evidence:** 0.83 on the holdout against 0.88 in cross-validation.
   - The holdout is a random 20% of patients, so this is **not** drift over time; I wrongly called it drift at first (corrected 2026-11-18).
     The likely cause is optimism from picking the best of 86 runs on the same folds.
6. **The real test is not independent.** `open`
   - The world was tuned to the same 84 real calls, so a real AUC of 0.70 shows consistency, not generalisation.
   - **Needs:** new Salud calls.

### Post-call model (comparison only; not chosen or tuned)

7. **The QA score and sentiment add nothing.** `accepted`
   - **Evidence:**
     - synthetic cross-validation: adding them gains at most 0.001;
     - real (47 rows): the QA score alone gives AUC 0.59 (0.42–0.75), and adding it to the model moves 0.680 to 0.684.
   - **Impact:** only how the call ended adds signal (+0.04 to +0.06 on synthetic data).
     In the synthetic world, call quality affects payment only through how the call ends, so this partly reflects how the world was built.
8. **On real calls, adding how the call ended scored lower than pre-call features alone.** `open`
   - **Evidence:** 0.704 → 0.680 on 47 rows, inside the noise. Watch it when more real calls arrive.
9. **The post-call model hasn't been chosen or tuned.** `open`
   - Notebook 10 compares feature groups only. Tune it when there is real outcome data to train or test it on.

### Data

10. **Salud's extract has no contact history.** `accepted`
    - Adversarial validation tells synthetic from real at AUC 0.945, almost entirely through `days_since_last_contact`.
    - Prior-contact features are therefore excluded from FS_TRANSFER.
11. **The synthetic world matches 50 of 51 real statistics.** `accepted`
    - **Miss:** the share of patients with an adjustment in the last 365 days is 0.250 in synthetic data, against a real upper bound of 0.242.
    - **Not checked yet:** the new-account pay rate (see item 1).
12. **All real test rows come from one day (2026-07-01).** `open`
    - There is no time variation in the real test, and 58 rows give AUC intervals of about ±0.15.
13. **The "oracle" is no longer a ceiling.** `accepted`
    - Hidden traits explain only 0.60, because an observable payment habit drives payment in the Salud-tuned world.
14. **The neural network fails on small samples.** `accepted`
    - Its AUC is below 0.5 when trained on 200 rows or fewer (early stopping on tiny validation sets).
      Don't use it at real-data sizes.

15. **26 of Salud's 84 calls can't be used for the 30-day payment test.** `open` (found 2026-11-18)
    - **23 calls have no account data in the extract.** 18 have a guarantor ID, but none of that patient's accounts or
      transactions were included; 5 have no guarantor ID. Without transactions, "did they pay?" can't be answered.
      Missing data isn't the same as no payment: 4 of these calls record a payment on the call, yet no receipt appears.
      18 are from July, 5 from October.
    - **3 calls (Oct 8–9) are too recent.** The data ends Nov 2, so only 24–25 of the 30 days can be seen.
      1 paid on day 25, so it is really a known "paid"; the other 2 can't be called yet.
      They are excluded together so that only positives aren't let in, which would inflate the pay rate.
    - **Impact:** all 58 usable rows come from one day (2026-07-01), so the real test has no spread over time (item 12).
    - **Fix:** ask Salud for the accounts and transactions of the 18 known guarantors, and for transactions after
      Nov 2, which would make the 3 recent calls usable. Added to the data request.
16. **Synthetic payment plans always come with a payment on the call.** `fixed` (found 2026-11-18)
    - **Update 2026-11-18:** fixed in the master world (`plan_on_promise = 0.25`): 579 post-call training rows now have a plan.
    - 670 synthetic calls set up a payment plan, and **all** were paid on the call, so the post-call training rows
      (`paid_on_call = 0`) have **0** payment plans. In the real usable rows, 5 of 47 post-call rows have a plan.
    - **Impact:** the post-call model can't learn anything from `action_payment_plan`; on real calls it ignores a plan entirely.
    - **Fix:** let the generator create plans whose first payment comes later (days 1–30), then re-check calibration.
17. **The pre-call training patients don't look like the real test patients.** `fixed` (found 2026-11-18)
    - **Update 2026-11-18:** fixed in the master file: pre-call rows are call days, the same population as the real test.
    - The fixed-date file (everyone with an open balance on Apr 1) is older, bigger debt: median 4 accounts, $855, 251 days
      since first placement, 36% never paid. The call file and the real test rows are closer to each other:
      2 accounts, $446 / $566, 97 / 67 days, 66% / 71% never paid.
    - **Why:** calls happen early in a debt's life; a one-day photo of everyone catches many old accounts.
    - **Impact:** the pre-call model is trained on a different mix from the one it's tested on (real rows are taken at
      call time). May explain part of the level shift (item 3) and the gap to the simple rule.
    - **Next:** compare training the pre-call model on the call file (no call fields) on the real 58; notebook 09 only
      compared them on synthetic data.
18. **Synthetic call volume grows fast, so the post-call holdout is only 3 weeks.** `accepted` (found 2026-11-18)
    - Calls per quarter: 85 (2024 Q4) → 4,459 (2026 Q1). The "latest 20% by date" holdout is Mar 8–31, 2026.
    - **Impact:** the holdout tests a narrow window, not a long drift. Fine for a comparison model; revisit if the post-call
      model becomes a deliverable.
19. **Salud's data has no amount due, due date or promised-payment date.** `open` (found 2026-11-18)
    - Calls: `CollectionResult` says *what* happened (17 "Promised to Pay"), but not when or how much. `CallbackDate` is
      empty and `RequestedCallback` is "No" on all 84. `PaymentAmount` is filled on 15 calls (paid on the call).
    - Accounts (290, one snapshot as of the extract): current balance, receipts, adjustments, statement count and last
      statement date, Epic plan payment counts (see item 24: these are hospital-side plan payments, not Salud transactions). Empty: `OnPaymentPlan`, plan last payment
      date, `FirstStatementDate`, `LastPlacementDate`, `SelfPayBalanceDate`. No amount due, minimum payment,
      statement amount or due date anywhere.
    - Transactions (499, Feb 2024 – Nov 2 2026, 44 guarantors) are the only dated history: 230 patient payments,
      45 insurance payments, 224 insurance adjustments. They record what was paid, never what was due.
    - **Impact:** we can't check whether a promise was kept on time, or model "pays by the due date". The plan
      counts can't be features: they describe the account after the call (in `LEAKING_SNAPSHOT_FIELDS`).
    - **Ask Salud:** promised amount and date per call, plan start date, monthly amount, due day, and plan history
      with dates. Added to the data request.
20. **The pre-call model drops the dollar-amount fields, and the rule for dropping them was set on the original world.** `open` (found 2026-11-18)
    - `modeling.DUPLICATES` drops each amount field whose count twin has Spearman ρ > 0.9 (and `placement_balance`,
      `days_since_last_discharge`, `ever_paid`). In the Salud-tuned world, amount paid vs number of payments is only
      0.85–0.86, below the rule's own cut-off (real 58: 0.99). The list is hard-coded, not recomputed per world.
    - **Quick test (2 × 5 folds, not logged):** adding the amounts changes CV by at most +0.006 (LR 0.866 → 0.872; LightGBM
      0.867 → 0.871) and the synthetic holdout by ≤ 0.003. The real-58 moves (0.643–0.719) are inside its CI (±0.15), so they are noise.
    - **Recommended (user to confirm):** keep 11 fields for now (simpler, same accuracy). Revisit with more real data, when amounts could matter.
    - **Update 2026-11-18 (master file EDA):** in the master world the amount twins are duplicates again (ρ 0.96–0.99), so the
      drop is justified there; `days_since_last_discharge` is not (ρ 0.71) and could be tried.
21. **The pre-call label means something different in training and in the real test.** `fixed` (found 2026-11-18)
    - **Update 2026-11-18:** fixed in the master world: calls run to 2026-11-18, so every 30-day window includes calls, as in the real test.
    - **Training (fixed date):** `paid_30d` = any patient payment from Apr 1 to May 1, 2026. The synthetic world's calls end on
      Mar 31, so **no one is called during that window**: the label is "pays without a call".
    - **Real test:** the snapshot date is the call day, so `paid_30d` includes the call itself. 12 of the 29 real payers paid
      on the call day.
    - **Evidence:** synthetic pay rate 35% (fixed date, no calls) vs 47% on synthetic call days; real 50%.
    - **Impact:** likely a main cause of the level shift (item 3) and a reason the model learns "habitual payers pay anyway"
      (item 2). It ranks who pays unprompted, while Salud's question is who pays when called.
    - **Options:** (a) extend the synthetic world's calls past Apr 1 so the fixed-date label includes calls;
      (b) train pre-call on the call file (call-day snapshots, call fields ignored), which matches the real test;
      (c) keep as is and document. (b) is the cheapest and also addresses item 17.

22. **Salud leaves `collection_attempted` blank on agent calls.** `fixed` (found 2026-11-18, notebook 12)
    - Real: blank on 64 of 67 agent calls, "Yes" on 94% of phone-menu calls; blank reads as "no", so in real data the field
      only means "phone-menu call". Synthetic fills it properly (85% of agent calls). It is the main field that tells
      synthetic post-call rows from real ones (adversarial AUC 0.962 → 0.897 without it).
    - **Recommended:** drop it from the post-call fields.
    - **Update 2026-11-18:** the master world now fills CollectionAttempt like Salud (`salud_collection_attempt`): blank on 96% of agent calls (real 96%), "Yes" on campaign IVR calls. In both worlds the field now only repeats `spoke_with_agent`; it can be dropped from the model without loss. Post-call adversarial AUC 0.962 → 0.877.
23. **Other call fields off between synthetic and real.** `fixed` (found 2026-11-18, notebook 12)
    - Asked about financial assistance: real 14%, synthetic 1.5% (generator: hardship calls only). Insurance action: real 5%,
      synthetic 12%. Neither is in the calibration list.
    - Synthetic payments taken on a call post the next day 60% of the time (real: same day). Doesn't change `paid_30d`;
      shifts `days_to_first_payment`.
    - **Fix:** tune the generator and add the first two to `salud-synth-calibrate`.
    - **Update 2026-11-18:** `asked_fa_rate` 0.17 (synthetic 17.4% vs real 14.3%), `insurance_review_p` 0.1 (1.4% vs 1.2%, after the coinsurance fix), `same_day_posting` (97% vs 91% on day 0). Four statistics added to `salud-synth-calibrate`; master world 55/55.
24. **Salud's transactions only cover payments made through Salud since placement; hospital (Epic) plan payments are missing.** `open` (found 2026-11-18)
    - Traced one patient (4 accounts placed Jun 22–28 2026, called Jul 1): transactions match Salud's received totals exactly, and
      our row is right (4 accounts, $4,026 owed, no payment before the call, first payment Aug 18 → `paid_30d` = 0).
    - Across all 290 accounts, patient-payment transactions match `SelfPayReceipts` for 283 (7 short by $265 in total), so
      the history **since placement** is complete.
    - But `EpicPaymentPlanNumberofPayments` is set per patient (same value on all accounts for 53 of 59 patients), is > 0 for
      20 patients (132 accounts), and is larger than the months since placement for 113 of the 132 (median 24.5 vs 5
      Salud transactions). So it counts payments on the **hospital's** payment plan, which never appear in Salud's
      transactions. Our "payment history" fields therefore start at zero for every new placement (see item 1) even when the
      patient has been paying the hospital for years.
    - The field is a snapshot as of the extract, so it can't be used as is (it includes payments after the call).
    - **Ask Salud:** the dated Epic plan history (start date, monthly amount, payments with dates), or at least the plan count
      as of the placement date.
25. **Our balance on the call day ignores adjustments and later charges.** `open` (found 2026-11-18)
    - `balance_at_t` = placement balance − patient payments before the call. Salud's own identity (placement balance −
      receipts − adjustments = current balance) fails for 67 of 290 accounts, mostly because the balance **went up** after
      placement (median −$104; up to $25k): new charges or re-billing that no transaction shows.
    - **Impact:** `balance_at_t` is right at placement and after payments, but misses adjustments and added charges.
    - **Ask Salud:** a dated balance history, or the charge transactions.
---

## 2026-11-18 · Task files for six people (`Task/`)

**Asked:** split the model building across six people, one self-contained instruction file per model family, each read
by that person's AI assistant.

**Decided with the user:** families = logistic regression, WoE scorecard, decision tree, random forest + extra trees,
gradient boosting, other (kNN / SVM / MLP / naive Bayes); ensemble later from their saved predictions. The lead shares
`pre_training_data.xlsx` privately. Each person may use the real rows to improve their model, **measured fairly**
(never scored by a model that saw them; every look logged). Hand-back: own branch `task/<family>`, fixed formats.

**Built:** `Task/1_…` to `Task/6_….md` (~560 lines each): project brief, setup (incl. `nbstripout --install`, which a
fresh clone doesn't have), data check (shape, fingerprint, counts), which rows are for what, feature sets, real-data
rules, one shared code block (identical in all six), family-specific variants and grids, choosing procedure,
deliverables, git hand-back, checklist. `AGENTS.md` points assistants to their task file.

**Caught by testing the code as written in the files:**
1. Leave-one-patient-out on the real rows biased AUC down (removing a patient shifts the training pay rate against it).
2. Pooling per-fold predictions on the real rows: 41 of 58 real rows share identical values on the 2 stable features,
   so tiny differences between fold models ranked them at random (0.718 → 0.686 even with weight 0.001). Now: repeated
   stratified grouped 5-fold over real patients, AUC within each fold, averaged (0.714 at every weight).
3. The WoE binning put yes/no fields into one bin (post-call scorecard 0.701 vs 0.798 for LR); now one bin per value.
The rule baseline is scored the same way: 0.716 (pre) / 0.720 (post) on real, 0.677 on synthetic test.

**Would do differently:** run the instruction files' code exactly as written before calling them done (it is what
found 2 and 3).

---

## 2026-11-18 · Feature importance with L1 / L2 / elastic net (notebook 13, rebuilt and verified)

**Asked:** a notebook that gives regularized regression every feature and shows which carry weight; then, "run it again
and make sure it is absolutely right", because the user will pick the features the models train on from it.

**Prepared first:** two candidates from Salud's own transactions (`paid_share_so_far`, `last_payment_amount`), checked by
`salud-propensity verify` (0 mismatches) and two new integrity checks (26/26); kept out of the older feature sets
(`D.NEW_PRE_CANDIDATES`).

**First version had four weaknesses (fixed in v2):**
1. L2 strength at the grid edge: first C = 10 (twins with large opposite-sign weights), then one-SE at C = 0.001, where
   L2 weights collapse to "each feature alone". Now: a wider grid (10⁻⁴…10²), L2 at its best C, and weight size is not
   used for ranking.
2. Accuracy per set was chosen and scored on the same rows. Added **nested selection**: it corrects "5 features are as
   good as all" — honestly, top 5 loses ~0.015 AUC (pre-call 0.692 vs 0.707; post-call 0.782 vs 0.797).
3. Only a linear model judged importance. Added LightGBM permutation importance: it ranks `days_since_last_payment` first
   and uses `days_since_last_placement` and `last_payment_amount` (curved effects).
4. `ever_paid` was an exact copy of the never-paid flag; dropped (guard cell asserts it).
Also: unseeded liblinear/saga gave different L1 sets per run (fixed earlier).

**Verified:** 9 guards pass; two full runs give identical outputs in every cell and the same feature-set file (md5
709b8599…); top-5 overlap 0.8–1.0 under three other seeds; no feature points the opposite way in real data.

**Result (tiers):** keep `n_payments_90d`, `days_since_last_payment` (+ post-call `action_payment_plan`, `promised_to_pay`);
one per family of payment history and debt age; account counts, balances, insurance, adjustments, discharge, sentiment,
QA add little. Sets saved for the model notebooks: stable, l1_one_se, current, top5/10/15, elastic_net, all.

**Written up:** `feature-selection.md` (tiers, every feature's evidence, honest accuracy, the sets, limits); a
`**Result:**` note under every output in notebooks 12 and 13, since outputs are stripped on commit.

---

## 2026-11-18 · Making sure the training data is right (master file audit)

**Asked:** make sure the data we train on is right, after the user traced one account and doubted the mapping.

**Done:**
1. **Independent recomputation** (`salud-propensity verify`, `propensity/verify.py`: separate code from `snapshots.py`, raw tables, codes
   decoded by number): all 11 pre-call fields, both targets and `window_complete` for all 79 real calls with a patient
   and 600 random synthetic calls. **0 mismatches**; later transactions correctly ignored; every `call_id` and `as_of`
   matches the raw contact history.
2. **Integrity checks** `master.audit()`: 24 checks (unique ids, split, no shared patients, label logic, value ranges,
   QA values, leak check: max single-field AUC 0.68). `salud-propensity master` refuses to write the file if any fails.
3. **Mapping bug fixed:** `action_insurance` matched "Explained Coinsurance/Deductible/Copay" (a billing explanation).
   Real rate 4.8% → 1.2%. Test added.
4. **Generator made to fill call fields like Salud** (watch items 22, 23): blank CollectionAttempt on agent calls,
   financial-assistance questions, rare insurance actions, same-day posting. Calibration extended to 55 statistics;
   master world 55/55 (seeds 11 and 12); the Salud-tuned world still regenerates byte for byte.
5. Master file rebuilt (8,833 rows: synthetic train 6,971 / test 1,778 calls, real 84); EDA notebook 12 re-run.

**Known limits (not fixable from Salud's data):** history starts at placement with Salud; hospital plan payments are
missing (item 24); the balance ignores later charges (item 25).

**Would do differently:** reconcile the tables against each other (receipts vs transactions, action labels vs flags) at
the first build, not after the user asked.

---

## 2026-11-18 · EDA of the master file (notebook 12)

`notebooks/12_master_data_eda.ipynb` (generator kept in the session scratchpad as `make_nb_12.py`).
- **Healthy:** steady synthetic call volume May 2023 – Nov 2026; train/test balanced (pay 0.556 / 0.555, no shared patients);
  pay rates inside the real intervals (pre 0.56 vs 0.50; post 0.45 vs 0.36); the same fields carry signal in the same
  direction, at similar strength (payments 90d: 0.68 synthetic, 0.72 real); 53% of repeat patients have mixed outcomes,
  so grouped folds stay necessary.
- **Fixed during the EDA:** synthetic QA was continuous (5,019 values) against 12 real values, so it gave synthetic rows
  away. The sampler now snaps QA to the real values; master file rebuilt.
- **Found:** watch items 22 (`collection_attempted` blank in Salud's data) and 23 (financial assistance, insurance action,
  next-day posting). Adversarial AUC: pre-call 0.831 (old world 0.945), driven by placement timing (all real rows are from
  one day); post-call 0.962, 0.897 without `collection_attempted`.
- **Also checked (user question):** balance and payment fields come from Salud's transactions. "Share of the placement
  balance paid so far" carries signal (AUC 0.65 synthetic, 0.72 real) but adds little to the 11 fields (LR: synthetic
  test 0.694 → 0.701, real 0.766 → 0.784, inside the noise).

---

## 2026-11-18 · The master file `pre_training_data.xlsx`: new synthetic world over Salud's time frame

**Asked:** redo the synthetic data over Salud's time frame up to 2026-11-18, as one master file for both models, with
`data_type`, random splits, and synthetic sentiment based on the real calls.

**Decided with the user:** one flat sheet (one row per call); Salud's 84 real calls included as `data_type = real`, test only;
synthetic sentiment and QA sampled from the real calls by outcome (no transcripts); random 80/20 split **by patient**.

**Built:**
- **World** `configs/synthetic_world_master.json` → `data/phase2/synthetic_world_master` (the Salud-tuned settings plus):
  - time frame: history from 2023-08 (first calls May 2023, like Salud's first placements), calls up to **2026-11-18**, no data after;
  - `call_delay_days = 35`: calls come soon after one of the patient's placements (before: spread over the whole history,
    which put calls on 276-day-old debt against Salud's 67);
  - `phone_pay_max_months = 3`: pay-by-phone only for recent placements;
  - `adjustment_p = 0.09`;
  - `plan_on_promise = 0.25`: a quarter of promise-to-pay calls set up a plan with the first payment later (real: 4 of 17).
  - **51/51** calibration statistics (seed 11 and 12), 59/59 validation checks. The new options default to off, and the
    Salud-tuned world still regenerates byte for byte.
- **File** `reports/phase2/pre_training_data.xlsx`, sheet "Model rows" (`salud-propensity master`): 8,693 rows × 74 columns.
  - Synthetic: 8,609 calls, 1,931 patients, May 2023 – Nov 17 2026; train 6,909 / test 1,700 (no patient on both sides);
    379 calls after Oct 19 have an incomplete 30-day window (`label_usable = False`).
  - Real: 84 calls (58 usable), `split = real_test`.
  - Added columns: `data_type`, `split`, `label_usable`, `post_call_row`, `sentiment_source`, `qa_source`.
  - Sentiment/QA sampler: per outcome group (paid, promised, non-committal, transferred, contested, insurance,
    no agent), real means shrunk towards the overall mean (8 pseudo-calls), real within-group covariance, clipped to the real range.

**Quick check (untuned LR / LightGBM, not logged):**

| | Old data | Master file |
|---|---|---|
| Pre-call, real 58 | 0.697 (GB-06) | 0.750–0.766 |
| Rule "paid before, most recent first" | 0.718 | 0.718 |
| Mean prediction vs actual (real) | 0.25 vs 0.50 | 0.48–0.55 vs 0.50 |
| Pre-call, synthetic test | 0.83 (fixed date) | 0.69–0.71 (call days: a harder, more realistic task) |
| Post-call, real 47 | 0.68 | 0.70–0.77 |

The real 58 has a CI of about ±0.15, so "beats the rule" is not established; the level now matches without a correction.

**Not done yet:** notebooks 00–11 still use the old files; they need re-running on the master file.

**Would do differently:** check what the label window contains (calls or not) before building the first design.

---

## 2026-11-18 · Why the models predict payment within 30 days

**The 30-day window is our choice, not a Salud requirement.** Reasons:
1. **A call needs a deadline to count as worked.** "Will they ever pay?" can't rank this week's calls, and the longer the window,
   the more a payment is caused by something else (a new statement, another call, insurance).
2. **It matches the monthly cycle:** statements and payment-plan payments are monthly, so 30 days ≈ one bill and one chance to pay.
3. **Most payments a call drives come fast.** Real 58: 29 paid within 30 days; 12 on the call day, 4 in days 1–7, 7 in days 8–14,
   6 in days 15–30 (median 7 days).
4. **It's the most predictable window** in the Salud-tuned world (notebook 09, scorecard): 30 days 0.861, 60 days 0.831, 90 days 0.801.
   In the original world there was no clear winner (0.674 / 0.692 / 0.690), so this reason depends on the world.

**Limit:** with no due dates, plan amounts or promised dates (watch item 19), "paid on time" can't be the target; a fixed
window is the fallback. The window is one setting, so changing it is cheap.

**Next:** ask Salud which time frame matters to them (data request item 7).

---

## 2026-11-18 · Design note: live scoring with temporary tables (agreed, not built)

**Decided (with the user):** the training CSVs stay local (synthetic, rebuilt from config + seed); they are not stored in Neon.
Live scoring builds model rows on demand in a temporary table and drops it afterwards:
- **pre-call:** one row per patient with an open balance, as of today → pre-call score → call list;
- **post-call:** one row per call (patient history + the 21 call fields from Stage 1) → post-call score.

**Keep permanently anyway:** the scores (ID, date, score, tier, model version) and ideally the feature row. They are small,
they let us check predictions against the 30-day outcome, and they become the real training data we lack (58 rows today).
Some fields (call sentiment and QA, corrected balances) can't be rebuilt exactly later.

**Risk:** the temporary table must be built by the same code as the training files (`snapshots.py`), or tested to give identical
rows if done in SQL. Otherwise features drift and scores quietly degrade.

---

## 2026-11-18 · The two training files, and why they are not one file

**What they are** (both synthetic CSVs with the same 68 columns; neither is an Excel file):

| | `synthetic_fixed_date_salud.csv` | `synthetic_snapshots_salud.csv` |
|---|---|---|
| One row = | one patient, as of **2026-04-01** | one **call**, as of the moment the call happened |
| Rows | 1,794 patients (1 row each) | 9,283 calls from 1,974 patients (median 4 calls each, up to 15) |
| Dates | all the same day | Oct 2024 – Mar 2026 |
| Call fields (21) | all empty / 0 (no call happened) | filled in |
| Pay rate | 35% (`paid_30d`) | 43% after the call (`paid_after_call_30d`); 20% paid on the call |
| Trains | pre-call model | post-call model (the 7,400 rows with `paid_on_call = 0`) |

**Why not one file:** a row is a photo of a patient on one date. The features (balance, days since the last payment) and the
30-day label change with that date, so the same patient yields different rows on different dates. The pre-call question
("who to call on a given day") needs one photo per patient on one day; the post-call question needs a photo at every call,
with what happened on it. 1,770 patients appear in both files, with different values.

**Could the pre-call model use the call file?** Yes; the 21 call columns would just be ignored. Notebook 09 found training on
either gives the same AUC (LR 0.861 both). The fixed file is kept for pre-call because it matches how the model is used
(score everyone on one day), gives one row per patient (no patient counted 5×), and is the design whose effects
the world was built around (0.867 vs 0.684 when *evaluated* on call rows, notebook 00).

**Learned:** "one file per question" was never explained to the user; the dictionary's `in_current_models` column plus this table do.

---

## 2026-11-18 · The synthetic data: what we built, the criteria, and what we learned (summary)

**Why synthetic:** Salud's final package has 84 real calls, and only 58 have a usable 30-day payment label. That is far too few
to train and compare models. So models learn on a simulated call centre in Salud's table format and are tested once on the real calls.

**How it evolved:**
1. **Original world (Oct 2026), `configs/synthetic_world.json`.**
   - Hidden patient traits drove payment, agents had skill levels, and calls had planted QA items and outcomes.
   - Loaded into Neon `big_synthetic`.
   - The models looked fine on it (CV 0.69) but **failed on Salud's data** (real AUC 0.45, ranking reversed). Key effects pointed
     the wrong way: older debt paid *less* in synthetic data and *more* in real data, and payment history was much weaker than in real data.
   - **Lesson:** matching averages is not enough. Relationships must match too.
2. **Calibration tool, `salud-synth-calibrate`.** It computes 51 statistics on both worlds and requires each synthetic value to fall inside
   the real 95% bootstrap interval. This turned "similar to Salud" into a pass/fail list.
3. **Realism mode** (new `realism` block; the original world still reproduces byte for byte). New mechanisms:
   - a skewed account count, with visits spread over time;
   - insurance processing after placement;
   - a monthly payment habit: patients start with ability, willingness and debt age, then continue like a plan;
   - campaign calls that mostly reach the phone menu, transfers, and campaigns that skip patients who are paying;
   - phone payments driven by the habit, and call-result offsets.
4. **Tuning** to `configs/synthetic_world_salud.json` (seed 11): **50/51 statistics matched**, stable on 4 seeds.
   - **Size:** 2,000 guarantors, 10,780 accounts, 13,288 contacts, 15,069 receipts, 2,831 QA reviews and 8,062 calls with transcripts.
   - **Validation:** all 59 checks pass.

**The six criteria** (now in the README):
1. the same structure as Salud's tables, with 59 automatic checks;
2. statistically like Salud's calls: 51 statistics inside the real 95% intervals, *including the relationships*;
3. a believable cause-and-effect story;
4. hidden true answers kept in `truth/`, never used as features;
5. realistic pitfalls built in: the current-state worklist leaks the outcome, contacts and accounts are many-to-many, and not every receipt is a patient payment;
6. reproducible (fixed seed), with no real record copied and only aggregate targets.

**Result on the models:**
- On real data the pre-call model moved from AUC 0.45 to **0.70**, ranking real patients the right way (35% → 42% → 74% by score third).
- It still doesn't beat the simple rule (0.72).
- The real test is no longer independent, because the world was tuned to the same calls.

**Gaps that remain** (tracked in the watch list):
- new accounts pay 19% in the synthetic world against 48% real; this isn't one of the 51 checks;
- real balances and adjustments have heavier tails (mean adjustments 30 times higher);
- there is no contact history in Salud's extract;
- one statistic is just outside its interval (the adjustment share);
- the targets come from only 84 calls on few days.

**What I'd do differently:**
- write the calibration tool *before* the first world, and include relationship checks from the start;
- hold some real calls out of tuning so the real test stays independent;
- add the new-account pay rate as a target.

**Updated:** README "Phase 2" section rewritten: two worlds, commands, output, the six criteria, known gaps, the propensity
models and real QA. The layout now lists `final_package.py`, `propensity/`, `notebooks/` and the migrations.

---

## 2026-11-18 · Why 26 of Salud's 84 real calls can't be used for the payment test

**Question:** Salud gave 84 real calls, but only 58 are usable for "paid within 30 days". What about the rest?

**Found (aggregates only):**

| Group | Calls | Reason | Payment seen |
|---|---|---|---|
| Usable | 58 | linked patient and accounts, full 30 days visible | all from 2026-07-01 |
| No account data | 23 | 18 have a guarantor ID but none of their accounts or transactions are in the extract; 5 have no guarantor ID | none visible, but 4 of these calls record a payment on the call, so the data is missing, not a non-payment |
| Too recent | 3 | called Oct 8–9; data ends Nov 2, so only 24–25 of 30 days visible | 1 paid on day 25; 2 not yet |

- **Results of the 23 calls, as recorded:**

  | Result | Calls |
  |---|---|
  | Transferred | 9 |
  | Promised to pay | 4 |
  | Paid on the call | 4 |
  | Non-committal | 3 |
  | Insurance follow-up | 2 |
  | Contested balance | 1 |

  18 are from July, 5 from October.

**Decided:**
- keep both groups out of training and testing;
- treat no visible payment as unknown, not as non-payment;
- keep out the 1 recent payer as well, because letting in only known payers would inflate the pay rate.

**Why it matters:** the whole real test comes from one day, so it says nothing about stability over time. That is watch-list items 12 and 15.

**Data request to Salud (updated):**
1. More call records with the same fields as the final package, ideally from a different time period, to test on unseen calls.
2. **The accounts and transactions for the 18 guarantors whose calls are in the package but whose accounts are not.**
3. **Transactions after Nov 2, 2026**, which would complete the 30-day window for the 3 calls from Oct 8–9.
4. The average cost per call, to set the calling cut-off.
5. The Guarantors tab, plus the ExcludeTransaction and FinancialClass fields.
6. *(added 2026-11-18, watch item 19)* Per call: the promised amount and promised date. Per account: payment plan start date,
   monthly amount, due day, and plan payment history with dates (`OnPaymentPlan` and the plan's last payment date are empty).
7. *(added 2026-11-18)* Which time frame matters to Salud's business: their call cycle, statement cycle, and when an
   account counts as "worked". The 30-day label is our choice (most predictable in notebook 09, matches the monthly
   statement cycle), not a Salud requirement. In the real 58, 29 paid within 30 days: 12 on the call day, 11 within 2 weeks.
8. *(added 2026-11-18, watch item 24)* The hospital-side (Epic) payment plan history with dates: what does
   `EpicPaymentPlanNumberofPayments` count, and from when?
9. *(added 2026-11-18, watch item 25)* A dated balance history or the charge transactions (67 accounts' balances rose after placement).

Items 2 and 3 alone could raise the real test from 58 to 79 rows.

**Would do differently:** I should have explained the 58-of-84 drop when first reporting the real test, instead of
leaving it as a bare number.

---

## 2026-11-18 · Clean-up for a fresh start

**Deleted, at the user's choice:**
- caches (`__pycache__`, `.pytest_cache`, `.DS_Store`);
- obsolete local outputs: `reports/qa`, `reports/sentiment`, `reports/phase1/backups` and `reports/phase1/ai_versions`, all of which are held in Neon;
- the original synthetic world: its data, snapshots and sentiment cache, plus the `reports/phase2/original_world` archive;
- Salud's raw transcript copy (the code reads only the "Inferred Speakers" version);
- `configs/synthetic_world_null_call_quality.json` (no code used it).

**Kept:** all code (the original world can be regenerated from `configs/synthetic_world.json`), the Salud-tuned world, the
final package, the real QA scores, the experiment log and Neon. Tests pass and the propensity data still loads.

---

## 2026-11-18 · Stage 1 QA on Salud's 84 real calls

**Result:**
- all 84 real calls scored with the dashboard's prompt (assist-v2, Groq);
- the real QA score alone ranks payment after the call at AUC 0.59 (0.42–0.75, 47 rows), and it adds nothing to the post-call model (0.680 → 0.684).

**Decided:**
- ran the QA only after the user's explicit go-ahead, because it sends the transcripts to Groq;
- kept the results local (`reports/phase2/qa_real.csv`, gitignored), with nothing written to the shared database;
- reused the dashboard's prompt and its QA-score formula, so the scores mean the same as on the dashboard.

**Went well:** every review is cached, so the run resumed after a network timeout and after two key changes without
repeating any request.

**Would do differently:**
- the first run crashed on a network timeout because I caught only `RuntimeError` and `ValueError`. Provider calls should also treat `OSError` as a clean stop;
- check a new API key with one cheap request before starting a long run.

---

## 2026-11-18 · Clearing the pending items (code-review fixes, migration 008, cleanups, commit)

**Decided:**
- **Live prompt back to v2.** The uncommitted v3 trial had made `assist-v3` the live prompt and rewritten the shared
  definitions, so the dashboard and Re-analyze would have used the prompt that lost its trial (0.84 vs 0.94).
  - v2 is restored as the live prompt.
  - v3 now lives in `phase1/prompt_v3.py` and is chosen with `salud-phase1 trial --version assist-v3`.
  - Both prompt texts were checked to be byte-identical to the originals, so existing LLM cache entries still match.
- **Trial command fixes:**
  - it refuses a mixed or empty baseline;
  - it scores both versions on exactly the same calls;
  - its JSON export is re-read from the database.
- **Tests:** a cleanup step that always runs (`try/finally`), and a test that the live prompt is v2.
- **Migration 008 deleted, not applied.** Keeping a few old tags costs nothing; dropping the table can't be undone.
  This also fixes the one failing test.
- **No ON DELETE CASCADE.** The review asked for it on `ai_analysis_versions`, but `ai_analyses` and `reviewed_tags` deliberately
  have none, so a call with analyses can't be deleted by accident. I fixed the misleading comment instead of changing the shared database.
- **Notebook cleanups:**
  - the oracle is no longer called a "ceiling";
  - the scores file's READ FIRST sheet now says the simple rule is at least as good on the real rows.

**Not done, waiting on the user:**
- QA scoring of the 84 real calls: it sends transcripts to Groq, and "do everything pending" is too broad a go-ahead to send real calls to an outside provider;
- the data request to Salud: it has to come from the team.

**Would do differently:** the v3 trial should never have changed the live default. A trial should take its version as
a parameter from the start.

---

## 2026-11-17 · Option 2: retune the synthetic world to Salud's data, then rerun notebooks 00–11

**Result:**
- **World:** the Salud-tuned world matches 50/51 real statistics, including the direction and strength of every relationship.
- **Choice:** GB-06, monotone LightGBM, chosen on cross-validation (0.876).
- **Real test:** 0.697 (0.55–0.84), up from 0.448. It ranks real patients the right way (35% → 42% → 74% pay by score third), but it is still a little below the rule "paid before, most recent first" (0.718).

**Decided:**
- built a calibration tool (`salud-synth-calibrate`) that sets every statistic against the real 95% bootstrap interval;
- tuned only a new `realism` block in the generator config, so the original world still reproduces byte for byte;
- archived the original-world results before rerunning;
- kept the choice rule unchanged (cross-validation only, one-SE among the finalists).

**Why:**
- the original world failed because it got the direction of key effects wrong (debt age, account counts, payment-history strength);
- a mechanism-level fix was needed, through the payment habit, insurance processing and IVR campaign calls; shifting distributions alone would not have fixed the relationships.

**Went well:**
- each statistic had a target interval, so "similar to Salud" became a pass/fail check;
- the fix was checked on four seeds, not one, to avoid overfitting a single draw;
- the real-test verdict and the warning in the scores file are computed from the result, not written in advance.

**Would do differently / caveats:**
- **The real test is now not independent.** The world was tuned to the same 84 calls the model is tested on, so 0.697 is a consistency check and should not be read as evidence that the model generalises. I should have proposed keeping some real rows out of the tuning. With 58 usable rows that would have left almost nothing for either job, but the trade-off should have been stated up front.
- The "oracle ceiling" (0.60) is below ordinary models (0.87), because outcomes are now driven by an observable habit. The notebook heading still says "ceiling", and the findings explain it.
- The tuning cache had reported 51/51; notebook 00's fresh bootstrap gives 50/51 (adjustment share 0.250 against an upper bound of 0.242). Report the number from the notebook, not from the tuning scratch run.

---

## 2026-11-17 · Propensity notebooks 09–11 (data size, call quality, final tests)

**Result:** the chosen scorecard (SC-05) works on synthetic patients (CV 0.687, holdout 0.639) but **fails on Salud's
58 real rows** (AUC 0.45, ranking reversed). The a-priori rule "paid before, most recent first" reaches 0.72 there.
The scores file was written to show the output format, but opens with a READ FIRST warning not to use it.

**Why it happened:** the synthetic world's relationships differ from real ones: debt age reversed, account counts
and insurance activity reversed, payment history far weaker. The models learned the synthetic world faithfully;
the world was wrong.

**What else we learned:** linear models win at every data size we have; tuned boosting would need much more data.
On synthetic calls, the call's outcome (+0.13 AUC) and sentiment (+0.015) carry the post-call signal and the QA
score adds nothing beyond them.

**Went well:** the choice was fixed before any test, so the real result is honest; the rule baseline from notebook
01 gave an immediate, interpretable benchmark; the failure was explained (feature directions), not just reported.

**Would do differently:** check feature directions against the real rows *in notebook 00*, before eight model
notebooks. I had the real single-feature AUCs in the profile table (payment history 0.72 real vs 0.55 synthetic)
and noted the gap, but didn't check signs. That one check would have redirected the work to fixing the synthetic
world first.

**Open / needs the user:** how to proceed: (1) ask Salud for more real data (more days of calls + transactions);
(2) re-tune the synthetic generator to the real directions measured here; (3) deliver an interim payment-history
scorecard. Not chosen yet.

---

## 2026-11-17 · Propensity notebooks 02–08 (model families, ~55 runs)

**Decided:** recommend **SC-05, the monotone WoE scorecard on `FS_TRANSFER`** (AUC 0.687, ≈ 9 features) as the
pre-call model, by the one-standard-error rule; keep LR-12 (L2 logistic regression) as the reference.

**Why:** the top of the leaderboard is flat — LR, scorecard and ensembles within 0.687–0.692, and no pairwise
difference among them is significant (corrected t-test). When accuracy ties, explainability decides: a points
scorecard with monotone bins is what collections teams use and what Salud's "ethical AI" positioning needs.

**Went well:** the same 25 folds for every run made every comparison paired; nested tuning per family kept the
comparison fair; investigating the weak scorecard instead of accepting it found a real binning flaw (0.62 → 0.69).

**Would do differently:** count features from the fitted model from the start (the first counts were wrong and the
L1 path chart had to be rerun); filter "best of" cells to the runs being compared (broke on rerun); log to the
scratchpad rather than /tmp.

**Open:** learning curves (does boosting win with more data?), the post-call model with call-quality features, the
one-time holdout and real-data tests (notebooks 09–11). Real-data QA scores still need approval.

---

## 2026-11-17 · Propensity notebooks 00–01 (data review, baselines, validation)

**Decided:** repeated 5×5 grouped stratified K-fold; log1p before scaling; "never" fill for recency; duplicates
pruned for linear models; `FS_TRANSFER` (no prior-contact features) as the main feature set; train the pre-call
model on **fixed-date** synthetic rows.

**Why:** each choice comes from a measurement in notebook 00/01 (see `eval-model.md`). The biggest one: the
call-anchored synthetic rows have an oracle ceiling of only 0.58, so no method comparison on them could mean
anything. Checking the ceiling *before* building eight model notebooks saved a lot of wasted work.

**Went well:** profiling the data first changed the plan for the better (training design, transfer feature set).
Shuffled-label and grouping tests on the shared code catch leakage early.

**Would do differently:** I set "one row per call" as the training design before measuring its signal. An oracle
check should be the first step whenever training on generated data.

**Open:** the user continued without objecting to the fixed-date recommendation; the `DESIGN` switch keeps the
call design one line away. QA scores for real calls still need approval (Groq). Ask Salud for cost per call.

---

## 2026-11-17 · Combined modelling sheet (Step 1)

**Decided:** one row per patient as of a call date (industry scores the person who pays; 75 of 77 patients have one
call); two models (pre-call ranking, post-call follow-through); train on synthetic, test on real; 30-day label.

**Why:** the brief asks for a score per patient account and a "call first" list; vendors (Experian, RevSpring,
Cedar) score the guarantor; payments and calls happen at the person level.

**Went well:** reconciled the sheet against an independent SQL query (0 mismatches) and proved the leak of the
snapshot fields with numbers instead of asserting it.

**Would do differently:** ask about grain with a short explanation of how propensity scoring works up front — the
first version of the question was too abstract.

---

## 2026-11-17 · Loading Salud's final package into Neon (`final` schema)

**Decided:** store the package as received (labels, prefixed IDs), one table per tab, plus transcripts and option
codes; apply only migration 009; enforce only links that hold for every row.

**Why:** the user prioritised the latest client data; converting to the old handoff's codes would add a
translation step that can only lose information.

**Went well:** tested on the Neon test branch first, verified every row field by field, proved the verifier catches
tampering, and confirmed `phase1`/`big_synthetic` were untouched.

**Would do differently:** catch that the package folder wasn't gitignored at the very first look (it was flagged,
but before any other step).
