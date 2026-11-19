# Plan: storing trained propensity models (Neon object storage, ONNX)

Status: **proposed** (2026-11-19), not built yet. Decided with the user: models live in **Neon object storage** (the
team's Neon project, region aws-us-east-2, where storage is available) as **ONNX** files; the backend reads them through
its own module. Not in the public git repo: a model can carry information learned from Salud's real calls (a
k-nearest-neighbours model even stores training rows).

## Where things live

| What | Where | Why |
|---|---|---|
| Model file | private bucket `propensity-models`, key `models/<target>/<family>/<model_id>/model.onnx` | bytes in object storage, never in git |
| Preprocessing parameters | same folder, `preprocess.json` (never-paid fill value, log columns, scaler means / scales, WoE bins and values) | the custom `Prep` / `WoE` steps don't convert to ONNX cleanly; their fitted numbers are applied by shared code instead |
| Registry (one row per model) | new table `phase2.models` (migration 011): id, family, variant, target (pre / post), feature set + features, object keys, sha256 of each file, metrics (CV, real, synthetic test), data fingerprint, trained by / at, library versions, status (candidate / selected) | Neon's recommended pattern: store the key in Postgres, the bytes in the bucket; both branch together, so the **test branch** gets its own copy to try things safely |

## Backend: `app/backend/salud_app/ml/` (the "separate place" to access models)

| File | Job |
|---|---|
| `storage.py` | S3 client for Neon storage (boto3, path-style addressing, credentials from the `AWS_*` variables Neon provides) |
| `registry.py` | reads `phase2.models`; downloads a model once, checks its sha256, caches it on disk and in memory |
| `preprocessing.py` | applies `preprocess.json` exactly as training did (the same code the task files import) |
| `runtime.py` | runs the ONNX model with onnxruntime (already a backend dependency for sentiment) |
| `scoring.py` | builds a call's features (from the master file or the database), returns probability and tier |

New routes: `GET /api/phase2/models` (registry and metrics), `GET /api/phase2/models/{id}`, `POST /api/phase2/score`.
The Payment Propensity page then shows the models and their results.

## Training side (the six task files)

A `save_model(model, feats, target, ...)` helper in the shared code:
1. converts the fitted pipeline to ONNX (`skl2onnx` for scikit-learn models, `onnxmltools` for LightGBM);
2. writes the preprocessing parameters to `preprocess.json`;
3. **parity check**: ONNX vs scikit-learn probabilities on the synthetic test rows must agree within 1e-4, or nothing is saved;
4. uploads both files and adds the registry row (or writes them to the person's folder for the lead to upload; see open questions).

## Things to check before building

- Each family converts to ONNX: logistic regression, trees, forests, LightGBM, SVM and MLP are supported; calibrated
  wrappers (`CalibratedClassifierCV`) and naive Bayes need a test; WoE is handled by `preprocess.json`.
- Bucket setup: `neon.ts` with `buckets: { "propensity-models": {} }` and `neon deploy` (private by default), or
  `neon bucket create`; the backend needs the branch's `AWS_*` credentials in `.env` (never committed).

## Open questions for the user

1. Who uploads: each teammate (they'd need storage credentials) or the lead (teammates hand over ONNX files privately)?
2. Should the task files gain `save_model()` now (teammates are working) or after the first results come back?
