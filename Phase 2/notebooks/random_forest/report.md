# Random forest and extra trees · results

## Final Models

| Target | Variant | Feature Set | Real Weight | CV AUC & SE | Real AUC | synthetic test AUC (95% CI) | Calib. slope (test) | Top-20% capture (test) | 
|---|---|---|---:|---:|---:|---:|---:|---:|
| Pre call | |Extra Trees, row 110 | 'tiers12' (7) | 5 | 0.7126 (+/- 0.0070) | 0.8414 | 0.7024 (0.6774-0.7262) | 1.273 | 33.1% |
| Post Call | Random Forest, row 5 | 'tiers12' (10) | 1 | 0.8028 (+/- 0.0067) | 0.7469 | 0.7827 (0.7568-0.8068) | 1.127 | 41.0% |

## Compared with rule baseline 

Compared with the ones laid out in the github, 0.716, 0.677, and 0.707 for pre call and 0.720, 0.677, 0.797 for post call. On the random forest pre call model, it scored 0.8414 on the real evaluation, compared to with 0.7164 for the payment recency rule. On the held out synthetic test set, it scored 0.7024 versus 0.6772. The post call model scored 0.7469 versus 0.7204 on real cases and 0.7827 versus 0.6745 on the synthrtic test set.

The highest-scoring fifth of the synthetic test list contained 33.1% of eventual payers for pre call and 41.0% for post call.

## WWhat I tried

using Ranodom Forest and Extra Trees on all six feature sets for both targets, for 24 initial configurations. I then tuned three sets per target, being 'tiers12', 'all', and 'current' for pre-call, and 'tiers12', 'l1_one_se', and 'stable' for post-call. I sampled 15 additional settings per algorithm and set, coming to 180 tuning configirations. The search varied tree count, maximum depth, minimum leaf size, feature sampling, and class weighting. Then fivve-fold grouped class-validation was used on the synthetic training data and kept calls from the same patient together

## What worked, what didn't

The best synthetic CV AUC increased from 0.7121 to 0.7140 for pre-call and from 0.8025 to 0.8034 for post-call. Several configurations remained close after tuning. 

Both selected models cleared their one-standard-wrror CV cutoffs (0.705462 for pre call and 0.796814 fpr post call). The selected pre-call model was 0.0182 below the strongest real AUC and used 7 features vs 21. The selected post-call model was 0.0100 below the strongest real AUC and used 10 features versus 14. Both gaps were within the assignment's 0.02 margin, so I favored the smaller input sets 

## use of real data 
 used real_eval() to compare three configurations per target at weights 0,1,5,and 20. The evaluation kept the patients being scored out of the corresponding model fit. The saved run history recorded 33 real evaluation calls, including exploratory repeats and the calls inside final_model()

 ## Feature notes 
 I measured permutation importance on grouped synthetic validations folds using the selected settings and real-data training weights. For each feature its values wre shuffled in the validation fold and observed how the AUC decreased

 Pre-call: Days since the last payment had the largest effect on ranking,
  with a mean AUC drop of 0.1311. Days since first placement and the
  number of payments in the past 90 days each had a much smaller mean
  drop of 0.0084.

  Post-call: Days since the last payment had the largest mean AUC drop
  (0.1048), followed by the payment-plan action (0.0553) and whether the
  patient promised to pay (0.0129). Days since first placement (0.0092)
  and the last payment amount (0.0077) had smaller effects.

  ## Problems, doubts, open questions
The real evaluation included 58 pre-call and 47 post-call cases, so
the estimated gains may change on a larger sample. The results measure
how well the models rank payment likelihood. Estimating whether changing
the call order increases payments would require a separate evaluation.

I ran the final synthetic evaluation once for pre-call and three times
for post-call. The post-call runs included an alternate configuration
and a repeat of the selected model. The submitted results contain one
final entry per target. Duplicate screening and tuning entries were
removed from the submission CSV, leaving 204 distinct configurations.

## Environment

Python: 3.11.9
numpy: 2.4.6
pandas: 3.1.1
scikit-learn: 1.9.1
openpyxl: 3.1.5
ipykernel: 7.4.0
nbstripout: 0.9.2
