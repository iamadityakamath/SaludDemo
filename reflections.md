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
      statement date, plan payments made (132 > 0) and missed (17 missed 1). Empty: `OnPaymentPlan`, plan last payment
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
