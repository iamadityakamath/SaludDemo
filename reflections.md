# Reflections

Decisions, the reasoning behind them, what went well, and what I'd do differently. One entry per task, newest
first. Kept up to date with every task.

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
