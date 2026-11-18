# Lessons learned

Practical lessons from building the Salud pipeline: what tripped us up, and the rule to follow next time.
Newest entries go at the top of each section. (Kept up to date with every task; aggregates only, no patient data.)

## Data

- **Match relationships, not just distributions.** The first world had realistic-looking columns but key effects pointed the
  wrong way (older debt paid *less* in synthetic data, *more* in real data). Calibrate each feature's single-feature AUC against the
  real data, as well as its mean and share.
- **Give every calibration target an interval.** `salud-synth-calibrate` compares each synthetic statistic with the real 95%
  bootstrap interval, which turns "similar to Salud" into a pass/fail list. Check the result on several seeds.
- **Tuning on the test data uses it up.** Once the world is tuned to the 84 real calls, testing on those calls measures
  consistency, not generalisation. Hold real rows out of tuning, or get new ones.

- **A model can be excellent on synthetic data and useless on real data.** SC-05 scored 0.69 in cross-validation and
  0.64 on the synthetic holdout, then 0.45 on Salud's real rows, with the ranking reversed. The synthetic world had
  debt age and account counts pointing the wrong way. **Rule:** before building models on generated data, compare
  each feature's direction of effect with whatever real data exists (single-feature AUCs, signs). It takes minutes,
  and it would have caught this in notebook 00.
- **A simple rule defined up front is a powerful sanity check.** "Paid before, most recent payment first" was set as
  a baseline in notebook 01 and reached 0.72 on real data, where every trained model failed.
- **Recalibration fixes the level, not the order.** Shifting the intercept to the real pay rate can't help a model
  whose ranking is reversed (Brier got worse).
- **Snapshot fields leak the outcome.** Salud's Accounts tab was exported *after* the calls (last updated Nov 3, 2026).
  Fields like `SelfPayReceipts`, `LastPatientPaymentFromTxnsDate`, `PlacementStatus = Zero Balance` already reflect
  payments made after the call; on the real rows they "predict" payment with AUC 0.88–0.91. **Rule:** rebuild every
  feature from dated rows before the as-of date; never use a current-state field as a feature.
- **Don't join the tables directly.** One call covers up to 39 accounts and each account has many transactions, so
  a raw join multiplies rows. **Rule:** aggregate each table to the row grain (patient as of a date) first, then combine.
- **Missing data can look like "no".** 18 real calls have a patient but no accounts or transactions in the extract;
  their "didn't pay" may just mean "not in the extract". **Rule:** flag such rows (`label_reliable = False`) and keep
  them out of training and evaluation.
- **Check dates before trusting a test set.** 76 of the 84 real calls happened on one day (Jul 1, 2026); all 58
  usable real rows come from that day. A real test is a single-day snapshot, not a time series.
- **The synthetic world only has signal where it was designed to.** Its effects were planted for one scoring date
  (2026-04-01). Rows anchored on each call carry almost no signal (oracle AUC 0.58) while the fixed-date rows do
  (oracle 0.80). **Rule:** measure an oracle ceiling (model on the hidden truth) before trusting a synthetic design.
- **Synthetic ≠ real.** An adversarial classifier separates them with AUC 0.95 (real: more accounts per patient,
  higher balances, much more insurance activity, no contact history). **Rule:** only use features that exist in
  both, and expect to recalibrate on real data.
- **Latest client data wins over older schema docs.** The final package stores option labels and prefixed IDs,
  unlike the Oct 14 handoff (codes, plain GUIDs). We stored it as received and kept the handoff codes as a lookup.

## Modelling

- **An "oracle" built on hidden traits is a ceiling only if the outcome runs through those traits.** In the Salud-tuned
  world, an observable payment habit drives payment, so ordinary models (0.87) beat the oracle (0.60).
- **When a simple rule scores 0.84, model families differ by about 0.01.** Most of the signal is payment history. Six features match
  25, and stacking and voting add nothing. Choose on calibration and explainability, then compare with the rule on real data.
- **Defaults for boosted trees and naive Bayes give poor probabilities** (calibration slope about 0.5 or about 0.2). Always look at
  Brier and slope, not only AUC, and calibrate inside the CV.

- **Quantile binning silently drops zero-heavy features.** A 98%-zero column becomes one bin, so WoE throws it away;
  the scorecard scored 0.62 until zero got its own bin and small counts one bin per value (→ 0.69). **Rule:** check
  the number of bins per feature after binning.
- **On ~1,000 rows, simple beats complex.** Logistic regression ≈ scorecard ≈ ensembles (0.687–0.692); tuned
  boosting/forests 0.673–0.680; trees, SVM, neural network lower. Matches the benchmarking doc (flexible models need
  far more data). **Rule:** always keep a well-tuned logistic regression as the reference.
- **Boosting defaults overfit small data badly** (0.627, Brier 0.26); tuning recovers +0.05. Never compare an untuned
  boosting model.
- **Normalisation matters for distance models, not much for logistic regression's ranking.** k-NN 0.576 raw → 0.676
  log1p+z; LR moved by ≤ 0.013. For LR the penalty, not the scaler, fixes calibration.
- **"Train only on the important features" didn't help;** aggressive selection (2–6 features) lost 0.04–0.07 AUC.
  The L1 path plateaus at ~11–14 features.
- **Report what the model really uses.** The first feature counts showed the input columns, not the non-zero L1
  weights, and looked at the search wrapper instead of the fitted model. **Rule:** unwrap `best_estimator_` and count
  non-zero coefficients.
- **Don't reuse settings tuned on all training rows inside a CV comparison** — it leaks the test folds. Ensemble base
  models used fixed settings chosen up front.
- **Use the same folds for every run** so differences can be tested pairwise (Nadeau–Bengio corrected t-test);
  fold scores overlap, so plain t-tests are over-confident.
- **Group folds by patient.** Patients repeat across rows; ungrouped folds put the same person in train and test.
  (The leak was small for logistic regression here, but grows with flexible models.)
- **Single train/test splits are noisy.** On the same model, a single 80/20 split's AUC swung ±0.03 between seeds;
  repeated 5×5 K-fold swung ±0.001.
- **At real-data size (n ≈ 60) every validation method is off by ~±0.1 AUC.** Never choose models on the real rows;
  use them only for the final test, with a bootstrap CI.
- **Pooled leave-one-out AUC is biased low** on small samples (−0.07 here). Prefer repeated K-fold.
- **Skewed money columns need log1p before scaling** (skew 6–34); plain z-scores are dominated by a few balances.

## Tooling

- **Network timeouts are `OSError`, not `RuntimeError`.** A long LLM run crashed on a read timeout. Catch `OSError` too,
  cache each result as it arrives, and check a new API key with one cheap request (`GET /models`) before a long run.

- **A trial must not change the live default.** Bumping `PROMPT_VERSION` for a trial silently moved the dashboard to the
  trial prompt. Pass the version as a parameter (`CallAssistant(..., version)`, `trial --version`). When refactoring a
  prompt, compare the built text byte for byte with the old one so the LLM cache stays valid.

- **pandas 3: `to_excel(writer, sheet_name=...)` must be a keyword.** Passing the sheet name positionally raises a
  TypeError and leaves an empty workbook ("At least one sheet must be visible").
- **Notebook cells that pick "the best run" must filter to the runs they compare.** On a rerun, a leaderboard
  filtered only by notebook contained later runs too (KeyError on LR-10).
- **sklearn `clone()` keeps only constructor parameters** — attributes set on an instance (e.g. a monotone flag) are
  lost. Pass everything through `__init__` or compute it up front.
- **Importing a generator script runs its top-level code.** Guard writes with `if __name__ == "__main__":`.
- **A long notebook re-run picks up code changes only at kernel start.** Edits made while a batch runs may or may
  not be in a given notebook — rerun the affected ones.
- **scikit-learn ≥ 1.8:** `LogisticRegression(penalty=...)` is deprecated. Use `l1_ratio` (0 = L2, 1 = L1) and
  `C=np.inf` for no penalty.
- **LightGBM on macOS needs `libomp`** (`brew install libomp`); sklearn's `HistGradientBoosting` works without it.
- **CSV round-trips change types:** a notebook id `"01"` came back as the number 1, so the leaderboard filter
  found nothing. Read logs with explicit `dtype`.
- **pandas `NaN` in text columns** is a float: `(value or "")` doesn't catch it. Check `isinstance(v, str)`.
- **Notebook outputs can hold real data.** `nbstripout` is installed as a git filter (`.gitattributes`), so outputs are
  stripped on commit; still only show aggregates of real data.
- **`sed` with `#` as the delimiter breaks when the replacement contains `#`.** Use the Edit tool for multi-character edits.
- **A Neon migration runner applies every pending file.** `salud-db setup` would also have run an unrelated pending
  migration (008, drops a table). Loaders that change one thing should apply only their own migration.
- **A row is a snapshot at one date, so each prediction question gets its own file.** Pre-call = one row per patient on a
  fixed day; post-call = one row per call. Same columns, different dates and rows. Explain this with a side-by-side table
  and the dictionary's `in_current_models` column, not with a column-group count.
- **A label window is a business choice, so write down why and confirm it with the client.** Check it in every world:
  30 days was best in the Salud-tuned world (0.861 vs 0.801 at 90) but not in the original one (0.674 vs 0.692 at 60).
- **Check what happens inside the label window, not only the features.** The fixed-date design's 30 days had no
  calls at all (the world's calls ended the day before), so the pre-call model learned "pays without a call"
  while the real test asked "pays when called". Fixed by a world whose calls run through the whole period.
- **New generator options default to off** so older worlds regenerate byte for byte; compare the CSVs after any change.
