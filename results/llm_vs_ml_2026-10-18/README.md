# LLM vs traditional ML for call QA scoring (Oct 18, 2026)

**Decision:** Phase 1 call scoring uses the LLM only. This folder is the evidence the roadmap asks for
when an ML classifier is not pursued ("document that explicitly rather than forcing it").

**Setup.** 60 held-out synthetic calls written by Qwen (`qwen/qwen3.8-27b`) from planted labels, scored by:
- the LLM (`openai/gpt-oss-120b`, zero-shot, prompt `assist-v1`)
- TF-IDF + logistic regression trained on 37 other LLM-written calls
- the same model trained on 4,092 template calls
- a majority-class baseline

95% bootstrap confidence intervals throughout.

**Result (7 transcript-visible checklist items):**

| | LLM | ML (37 realistic calls) | ML (4,092 template calls) | Majority |
|---|---|---|---|---|
| Average accuracy | 0.87 | 0.82 | 0.82 | 0.78 |
| Average Cohen's kappa | **0.66** | 0.15 | 0.38 | 0.00 |

The LLM agrees with the planted labels well beyond chance, with no training data. The ML model needs
many realistic labeled calls, and Salud has about 10. Trained on formulaic template calls, it reaches
~100% on templates but does not transfer to natural speech.

**Known issues at the time:**
- Call outcome was poor for every approach (~0.3), because "paid", "payment plan" and "promise to pay"
  overlap. Prompt `assist-v2` adds precise outcome definitions; its rerun was not completed (Groq daily
  token cap).
- Qwen does not always follow the planted labels, so the labels carry some noise.
- These are synthetic labels, not human ones. Phase 1 replaces them with manual tags (10 real + 50 synthetic calls).

Files: `comparison.md` (full tables), `results.csv`, `llm_predictions.csv`, `ml_learning_curve.csv`.
The ML code was removed after this run and remains in git history (commit 487a6a7).
