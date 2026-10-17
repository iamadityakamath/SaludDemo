"""The labels every QA scorer predicts, and loaders that put synthetic truth and human ground
truth into the same shape: one row per call with `contact_id`, `text` and one column per target.

Targets come only from the transcript. Call Statusing and Grammar are judged from the CRM
disposition and agent note, so they are scored separately (not by these models).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import schema as S

# Checklist items visible in a transcript.
TRANSCRIPT_ITEMS = [k for k in S.QA_KEYS if k not in ("call_statusing", "grammar")]
SENTIMENTS = ["negative", "neutral", "positive"]
OUTCOMES = ["paid", "payment_plan", "promise_to_pay", "non_committal", "refused_or_disputed",
            "insurance_followup", "transferred", "hung_up"]
TARGETS = TRANSCRIPT_ITEMS + ["sentiment", "outcome"]
# Ground-truth CSV columns. llm_model/overrides record LLM pre-fill provenance (blank = human-only).
LABEL_COLUMNS = ["file_id", "rater"] + TARGETS + ["notes", "llm_model", "overrides"]

ITEM_DEFINITIONS = {
    "greeting": "Agent opens with the approved script: identifies themself and the department "
                "(and the recorded-line notice on outbound calls).",
    "culture": "Agent is courteous, patient and empathetic throughout; no pressure, blame or "
               "dismissive remarks.",
    "capturing_data": "Agent confirms or updates key contact data (callback number, mailing address).",
    "understanding_needs": "Agent identifies the caller's actual concern and responds to it with fitting "
                           "options (e.g. financial assistance or a plan for hardship) instead of a generic push to pay.",
    "correct_info": "Information about the balance, insurance and next steps is clear and accurate; "
                    "no vague or contradictory explanations.",
    "first_contact_resolution": "The caller's issue is resolved on this call without a callback, "
                                "transfer or 'call back later'.",
    "compliance": "Agent verifies identity (e.g. name and date of birth) BEFORE disclosing any account "
                  "or balance information, and does not disclose to unverified third parties.",
}

_OUTCOME_MAP = {
    "paid_guarantor_full": "paid", "paid_account_full": "paid", "partial_payment": "paid",
    "paid_plan_full": "payment_plan", "promised_to_pay": "promise_to_pay",
    "non_committal": "non_committal", "contested_balance": "refused_or_disputed",
    "would_not_pay": "refused_or_disputed", "pursued_insurance": "insurance_followup",
    "na_transferred": "transferred",
}


def clean_transcript(text: str) -> str:
    """Drop the editorial header line (real and synthetic files both carry one)."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("Editorial note:")]
    return "\n".join(lines).strip()


def load_synthetic(run_dir: Path) -> pd.DataFrame:
    calls = pd.read_csv(run_dir / "truth" / "truth_calls.csv")
    tdir = run_dir / "transcripts"
    calls = calls[calls["contact_id"].map(lambda c: (tdir / f"{c}.txt").exists())].copy()
    calls["text"] = [clean_transcript((tdir / f"{c}.txt").read_text()) for c in calls["contact_id"]]
    out = calls[["contact_id", "text", "agent", "scenario"]].copy()
    for k in TRANSCRIPT_ITEMS:
        out[k] = calls[f"qa_{k}"].map({True: "Yes", False: "No"})
    out["sentiment"] = calls["sentiment_label"]
    out["outcome"] = calls["true_result"].map(_OUTCOME_MAP)
    out.loc[calls["hung_up"].astype(bool), "outcome"] = "hung_up"
    return out.reset_index(drop=True)


def load_ground_truth(labels_csv: Path, transcripts_dir: Path) -> pd.DataFrame:
    """Human tags in templates/ground_truth_template.csv format. Multiple raters allowed."""
    # keep_default_na=False: pandas would otherwise read a rater's literal "N/A" as missing.
    gt = pd.read_csv(labels_csv, dtype=str, keep_default_na=False).replace("", None).dropna(subset=["file_id"])
    for col in TRANSCRIPT_ITEMS:
        gt[col] = gt[col].str.strip().str.capitalize().replace({"Na": "N/A", "N/a": "N/A"})
    gt["sentiment"] = gt["sentiment"].str.strip().str.lower()
    gt["outcome"] = gt["outcome"].str.strip().str.lower()
    gt["contact_id"] = gt["file_id"].str.replace(".txt", "", regex=False)
    gt["text"] = [clean_transcript((transcripts_dir / f"{c}.txt").read_text()) for c in gt["contact_id"]]
    return gt


def write_template(path: Path, transcripts_dir: Path | None = None) -> None:
    cols = LABEL_COLUMNS
    rows = []
    if transcripts_dir is not None:
        rows = [{"file_id": p.name} for p in sorted(transcripts_dir.glob("*.txt"))]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
