# LLM vs traditional ML: call QA scoring

Test set: 60 LLM-written synthetic calls (`run_llm`), scored against planted labels.
95% bootstrap confidence intervals in brackets.

## Accuracy

| target | LLM (openai/gpt-oss-120b) | ML (LLM calls, n=37) | ML (template, n=4092) | Majority |
|---|---|---|---|---|
| greeting | 0.97 [0.92, 1.00] | 0.87 [0.77, 0.95] | 0.93 [0.87, 0.98] | 0.87 [0.77, 0.95] |
| culture | 0.87 [0.77, 0.93] | 0.85 [0.77, 0.93] | 0.82 [0.72, 0.90] | 0.83 [0.75, 0.92] |
| capturing_data | 0.95 [0.88, 1.00] | 0.83 [0.73, 0.92] | 0.90 [0.82, 0.97] | 0.83 [0.73, 0.92] |
| understanding_needs | 0.88 [0.80, 0.97] | 0.80 [0.70, 0.90] | 0.72 [0.62, 0.83] | 0.70 [0.58, 0.82] |
| correct_info | 0.82 [0.72, 0.92] | 0.82 [0.72, 0.92] | 0.78 [0.67, 0.88] | 0.82 [0.72, 0.92] |
| first_contact_resolution | 0.70 [0.58, 0.80] | 0.75 [0.65, 0.85] | 0.75 [0.63, 0.85] | 0.62 [0.50, 0.73] |
| compliance | 0.93 [0.87, 0.98] | 0.82 [0.70, 0.92] | 0.83 [0.72, 0.92] | 0.82 [0.70, 0.92] |
| sentiment | 0.75 [0.63, 0.85] | 0.82 [0.72, 0.92] | 0.27 [0.17, 0.38] | 0.72 [0.60, 0.83] |
| outcome | 0.32 [0.20, 0.43] | 0.27 [0.17, 0.38] | 0.30 [0.18, 0.42] | 0.28 [0.17, 0.42] |
| checklist average (7 items) | 0.87 | 0.82 | 0.82 | 0.78 |

## Cohen's kappa (agreement beyond chance)

| target | LLM (openai/gpt-oss-120b) | ML (LLM calls, n=37) | ML (template, n=4092) | Majority |
|---|---|---|---|---|
| greeting | 0.87 [0.65, 1.00] | 0.00 [0.00, 0.00] | 0.76 [0.48, 0.95] | 0.00 [0.00, 0.00] |
| culture | 0.64 [0.41, 0.82] | 0.16 [0.00, 0.46] | 0.48 [0.20, 0.72] | 0.00 [0.00, 0.00] |
| capturing_data | 0.84 [0.64, 1.00] | 0.00 [0.00, 0.00] | 0.69 [0.42, 0.89] | 0.00 [0.00, 0.00] |
| understanding_needs | 0.73 [0.53, 0.91] | 0.45 [0.20, 0.69] | 0.24 [-0.02, 0.49] | 0.00 [0.00, 0.00] |
| correct_info | 0.38 [0.06, 0.66] | 0.00 [0.00, 0.00] | -0.06 [-0.13, 0.00] | 0.00 [0.00, 0.00] |
| first_contact_resolution | 0.40 [0.18, 0.61] | 0.43 [0.21, 0.64] | 0.42 [0.19, 0.63] | 0.00 [0.00, 0.00] |
| compliance | 0.78 [0.53, 0.95] | 0.00 [0.00, 0.00] | 0.14 [0.00, 0.41] | 0.00 [0.00, 0.00] |
| sentiment | 0.51 [0.33, 0.70] | 0.53 [0.30, 0.74] | 0.11 [0.05, 0.20] | 0.00 [0.00, 0.00] |
| outcome | 0.17 [0.03, 0.31] | 0.03 [-0.06, 0.13] | 0.11 [0.04, 0.21] | 0.00 [0.00, 0.00] |
| checklist average (7 items) | 0.66 | 0.15 | 0.38 | 0.00 |

## ML learning curve (checklist-average accuracy on the same test set)

| training calls | accuracy |
|---|---|
| 25 | 0.80 |
| 37 | 0.82 |

## LLM evidence quality

- Calls that failed: 2
- Checklist answers backed by a verified transcript quote: 88%
- Quotes dropped per call (not found in transcript): 1.20
