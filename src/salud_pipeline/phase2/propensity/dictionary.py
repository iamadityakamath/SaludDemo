"""Every column of the patient snapshot sheet: what it means, where it comes from, and who may use it.

`use` decides which model sees a column:
    id      identity / bookkeeping, never a feature
    pre     known before the call: used by the pre-call model ("who to call first") and the post-call model
    post    known only after the call: used by the post-call model only
    label   what happened after the as-of date: the targets, never a feature
    audit   kept for checks (fairness), never a feature
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Col:
    name: str
    use: str
    group: str
    meaning: str
    source: str
    rule: str = ""
    note: str = ""


T = "t = the call's date and time (the as-of date). 'Before t' = posted on an earlier calendar day."

COLUMNS: list[Col] = [
    # ---- identity ----
    Col("snapshot_id", "id", "Identity", "Unique row id", "-", "'<source>/<call_id>'"),
    Col("source", "id", "Identity", "real (Salud final package) or synthetic (BIG synthetic world)", "-"),
    Col("guarantor_id", "id", "Identity", "The patient (guarantor) the row describes", "contact_history.salud_GuarantorId"),
    Col("call_id", "id", "Identity", "The call that sets the as-of date", "contact_history.salud_gphpvcontacthistoryId"),
    Col("tcn_record_id", "id", "Identity", "Transcript / call-recording id", "contact_history.salud_TCNRecordID"),
    Col("as_of", "id", "Identity", "The as-of date t (call date and time)", "contact_history.salud_DateandTimeofContact"),
    Col("agent", "id", "Identity", "Agent on the call (for coaching views, not a feature)", "contact_history.salud_Agent"),
    Col("n_linked_accounts", "id", "Identity", "Accounts the call is linked to", "account_contact_links"),
    Col("label_reliable", "id", "Eligibility",
        "True when the patient is known and has at least one account in the extract", "contact_history, accounts",
        note="False rows: 'no payment' may only mean the data is missing. Kept out of training and evaluation."),
    Col("window_complete", "id", "Eligibility", "True when the full 30-day outcome window is inside the data",
        "transactions", "t + 30 days <= last transaction date"),
    Col("paid_on_call", "id", "Eligibility", "A payment was taken on the call itself",
        "contact_history.salud_PaymentAmount / salud_CollectionResult", "amount > 0 or result starts with 'Paid'",
        "Post-call model excludes these rows: the outcome is already known."),

    # ---- pre-call: accounts at t ----
    Col("n_accounts", "pre", "Accounts at t", "Patient's accounts (visits) placed on or before t", "accounts"),
    Col("n_hospital_accounts", "pre", "Accounts at t", "Of those, hospital (HB) accounts", "accounts.salud_HBPB"),
    Col("n_physician_accounts", "pre", "Accounts at t", "Of those, physician (PB) accounts", "accounts.salud_HBPB"),
    Col("placement_balance", "pre", "Accounts at t", "Sum of patient balances when the accounts were placed ($)",
        "accounts.salud_PlacementPatientBalance"),
    Col("balance_at_t", "pre", "Accounts at t", "Estimated patient balance still owed at t ($)",
        "accounts + transactions", "placement balance - patient payments before t, floored at 0",
        "Rebuilt because the Accounts tab's current balance is a snapshot taken after the calls (leaks the outcome)."),
    Col("days_since_first_placement", "pre", "Accounts at t", "Age of the oldest placed account (days)",
        "accounts.salud_PlacementDate"),
    Col("days_since_last_placement", "pre", "Accounts at t", "Age of the newest placed account (days)",
        "accounts.salud_PlacementDate"),
    Col("days_since_last_discharge", "pre", "Accounts at t", "Days since the most recent discharge (service)",
        "accounts.salud_DischargeDate"),

    # ---- pre-call: payment history ----
    Col("ever_paid", "pre", "Payment history", "Any patient payment before t", "transactions", "self-pay rule (below)"),
    Col("n_payments_90d", "pre", "Payment history", "Patient payments in the 90 days before t", "transactions"),
    Col("amount_paid_90d", "pre", "Payment history", "$ of those payments", "transactions"),
    Col("n_payments_365d", "pre", "Payment history", "Patient payments in the 365 days before t", "transactions"),
    Col("amount_paid_365d", "pre", "Payment history", "$ of those payments", "transactions"),
    Col("n_payments_ever", "pre", "Payment history", "All patient payments before t", "transactions"),
    Col("amount_paid_ever", "pre", "Payment history", "$ of all patient payments before t", "transactions"),
    Col("days_since_last_payment", "pre", "Payment history", "Days since the last patient payment (empty if none)",
        "transactions"),
    Col("paid_share_so_far", "pre", "Payment history",
        "Share of the placement balance the patient has paid since the first placement, before t (0-1)",
        "accounts + transactions", "patient payments since the first placement / placement balance, capped at 1",
        "added 2026-11-18 as a candidate (not in the old CSVs)"),
    Col("last_payment_amount", "pre", "Payment history", "$ of the most recent patient payment day before t (0 if none)",
        "transactions", "", "added 2026-11-18 as a candidate (not in the old CSVs)"),

    # ---- pre-call: insurance ----
    Col("n_insurance_payments_365d", "pre", "Insurance activity", "Insurance payments in the 365 days before t",
        "transactions", "TXN Type = Payment, Payment Type = Insurance Payment"),
    Col("amount_insurance_365d", "pre", "Insurance activity", "$ of those payments", "transactions"),
    Col("n_adjustments_365d", "pre", "Insurance activity", "Adjustments in the 365 days before t", "transactions",
        "TXN Type = Adjustment"),
    Col("amount_adjustments_365d", "pre", "Insurance activity", "Absolute $ of those adjustments", "transactions",
        note="Sign convention of adjustments isn't documented, so absolute values are used."),

    # ---- pre-call: contacts ----
    Col("n_prior_contacts_90d", "pre", "Prior contacts", "Contact attempts (any channel) in the 90 days before t",
        "contact_history", note="Real extract holds only the 84 calls, so this is mostly 0 there; synthetic has all attempts."),
    Col("n_prior_calls", "pre", "Prior contacts", "Earlier calls with this patient", "contact_history"),
    Col("days_since_last_contact", "pre", "Prior contacts", "Days since the previous contact (empty if none)",
        "contact_history"),

    # ---- post-call: this call ----
    Col("inbound", "post", "This call", "The patient called in", "contact_history.salud_ConnectionType",
        note="Known at the call, but not useful for choosing whom to call: post-call model only."),
    Col("connection_type", "post", "This call", "Connection type (label)", "contact_history.salud_ConnectionType"),
    Col("contact_type", "post", "This call", "Who was on the line (label)", "contact_history.salud_Contact_Type"),
    Col("disposition", "post", "This call", "Call disposition (label)", "contact_history.salud_CallDisposition"),
    Col("spoke_with_agent", "post", "This call", "Disposition = Spoke with CSR", "contact_history.salud_CallDisposition"),
    Col("collection_attempted", "post", "This call", "Agent attempted collection", "contact_history.salud_CollectionAttempt"),
    Col("collection_result", "post", "This call", "Recorded outcome of the call (label)",
        "contact_history.salud_CollectionResult", note="'Paid…' results are the outcome itself; used only to set paid_on_call."),
    Col("promised_to_pay", "post", "This call", "Result = Promised to Pay", "contact_history.salud_CollectionResult"),
    Col("non_committal", "post", "This call", "Result = Non-Committal", "contact_history.salud_CollectionResult"),
    Col("contested_or_refused", "post", "This call", "Result = Contested Balance or Said Would Not Pay",
        "contact_history.salud_CollectionResult"),
    Col("pursued_insurance", "post", "This call", "Result = Pursued Insurance Payment", "contact_history.salud_CollectionResult"),
    Col("transferred", "post", "This call", "Result = N/A, Transferred, or a transfer was needed",
        "contact_history.salud_CollectionResult / salud_needsomeoneelsetransfer"),
    Col("action_shared_balance", "post", "This call", "Agent verbally shared the balance", "contact_history.salud_ActionsTaken"),
    Col("action_payment_plan", "post", "This call", "Payment plan set up or updated", "contact_history.salud_ActionsTaken"),
    Col("action_financial_assistance", "post", "This call", "Financial-assistance options, paperwork sent or requested",
        "contact_history.salud_ActionsTaken"),
    Col("action_statement_or_bill", "post", "This call", "Statement or itemized bill provided or requested",
        "contact_history.salud_ActionsTaken"),
    Col("action_insurance", "post", "This call", "Insurance corrected, added or review requested",
        "contact_history.salud_ActionsTaken"),
    Col("n_actions", "post", "This call", "Number of actions recorded", "contact_history.salud_ActionsTaken"),
    Col("reviewed_balance", "post", "This call", "Balance was reviewed", "contact_history.salud_ReviewBalance"),
    Col("asked_financial_assistance", "post", "This call", "Patient asked about financial assistance",
        "contact_history.salud_inquireaboutfinancialassistance"),
    Col("talk_seconds", "post", "This call", "Agent talk time (seconds)", "contact_history.salud_AgentCallTalkDuration"),
    Col("payment_on_call", "post", "This call", "$ taken on the call", "contact_history.salud_PaymentAmount",
        note="Part of the outcome: used only to set paid_on_call, never as a feature."),

    # ---- post-call: call quality (Stage 1) ----
    Col("qa_score", "post", "Call quality", "Share of the QA checklist items met (0-1)",
        "real: Stage 1 AI review (salud-propensity qa, dashboard prompt); synthetic: truth_calls.quality_score_transcript_items",
        note="Empty for real calls until Stage 1 runs on them (needs approval: it sends transcripts to Groq)."),
    Col("caller_sentiment_overall", "post", "Call quality", "Caller's average sentiment over the call (-1..1)",
        "transcripts, local sentiment model (cardiffnlp roberta, ONNX)", "sentiment.results.score_call"),
    Col("caller_sentiment_end", "post", "Call quality", "Caller's sentiment over the last 30% of their turns",
        "transcripts, local sentiment model"),
    Col("caller_sentiment_shift", "post", "Call quality", "End minus start of the caller's sentiment",
        "transcripts, local sentiment model"),
    Col("agent_sentiment_overall", "post", "Call quality", "Agent's average sentiment (tone)",
        "transcripts, local sentiment model"),
    Col("has_caller_turns", "post", "Call quality", "The transcript has at least one caller turn",
        "transcripts", note="False = voicemail / phone menu only; sentiment columns are empty."),

    # ---- audit ----
    Col("preferred_language", "audit", "Fairness audit", "Preferred language",
        "contact_history.salud_PreferredLanguage",
        note="Never a model input (proxy for ethnicity); used only to check that scores don't differ unfairly."),

    # ---- labels ----
    Col("paid_30d", "label", "Outcome", "Any patient payment from the call day to 30 days after (target)",
        "transactions", "self-pay rule; post day in [t, t + 30 days]",
        "Main target. Includes a payment taken on the call."),
    Col("paid_after_call_30d", "label", "Outcome", "Any patient payment from the day after the call to 30 days after",
        "transactions", "post day in [t + 1, t + 30 days]", "Target of the post-call (follow-through) model."),
    Col("paid_amount_30d", "label", "Outcome", "$ of patient payments in the 30-day window", "transactions"),
    Col("days_to_first_payment", "label", "Outcome", "Days from t to the first payment in the window (empty if none)",
        "transactions"),
    Col("paid_fraction_30d", "label", "Outcome", "paid_amount_30d / balance_at_t (capped at 1)",
        "transactions, accounts", note="For weighting full or fast payments higher later (brief, section 3)."),
]

SELF_PAY_RULE = ("Patient payment = TXN Type 'Payment' and Payment Type 'Patient Payment'; when Exclude Transaction "
                 "is provided (synthetic only), it must also be 'Include'. The real package has no Exclude "
                 "Transaction column, so all patient payments count there.")

LEAKING_SNAPSHOT_FIELDS = [
    "salud_SelfPayBalance", "salud_SelfPayReceipts", "salud_SelfPayAdjustments", "salud_SelfPayBalanceDate",
    "salud_PlacementStatus", "salud_StatementCount", "salud_LastStatementDate", "salud_LastCallDate",
    "salud_LastAccountContact", "salud_LastPatientPaymentFromTxnsDate", "salud_LastUpdatedfromClientSystem",
    "salud_OnPaymentPlan", "salud_EpicPaymentPlanNumberofPayments", "salud_EpicPaymentPlanPaymentsMissed",
    "salud_EpicPaymentPlanLastPaymentDate",
]


def names(use: str | tuple[str, ...]) -> list[str]:
    uses = (use,) if isinstance(use, str) else use
    return [c.name for c in COLUMNS if c.use in uses]


# Numeric / boolean features each model may use. Label-type text columns (connection_type, disposition,
# collection_result, contact_type) stay in the sheet for reading but are represented by the flags above.
TEXT_ONLY = {"connection_type", "contact_type", "disposition", "collection_result", "payment_on_call"}
# Candidates added on 2026-11-18 exist only in the master file, so the older feature sets (and CSVs) leave them out.
NEW_PRE_CANDIDATES = ["paid_share_so_far", "last_payment_amount"]
PRE_FEATURES = [c for c in names("pre") if c not in NEW_PRE_CANDIDATES]
POST_FEATURES = PRE_FEATURES + [c for c in names("post") if c not in TEXT_ONLY]


# The fields the current models actually train on (notebooks 10-11). The other pre/post columns are candidates that
# were compared and left out: duplicates, or prior-contact fields that Salud's extract doesn't have.
MODEL_PRE = ["n_accounts", "n_hospital_accounts", "n_physician_accounts", "balance_at_t", "days_since_first_placement",
             "days_since_last_placement", "n_payments_90d", "n_payments_365d", "days_since_last_payment",
             "n_insurance_payments_365d", "n_adjustments_365d"]                    # = modeling.FS_TRANSFER
MODEL_POST_EXTRA = ["inbound", "spoke_with_agent", "collection_attempted", "promised_to_pay", "non_committal",
                    "contested_or_refused", "pursued_insurance", "transferred", "action_shared_balance",
                    "action_payment_plan", "action_financial_assistance", "action_statement_or_bill", "action_insurance",
                    "n_actions", "asked_financial_assistance", "talk_seconds",            # how the call ended (16)
                    "caller_sentiment_overall", "caller_sentiment_end", "caller_sentiment_shift",
                    "agent_sentiment_overall",                                          # sentiment (4)
                    "qa_score"]                                                         # QA (1)


def as_frame() -> pd.DataFrame:
    df = pd.DataFrame([c.__dict__ for c in COLUMNS])
    df["known_before_call"] = df["use"].isin(["id", "pre", "audit"])
    df["used_by"] = df["use"].map({"pre": "pre-call + post-call models", "post": "post-call model",
                                   "label": "target (never a feature)", "id": "not a feature",
                                   "audit": "fairness check only"})
    df.loc[df["name"].isin(TEXT_ONLY & set(names("post"))), "used_by"] = "readable only (represented by flags)"
    label = {"paid_30d": "TARGET of the pre-call model", "paid_after_call_30d": "TARGET of the post-call model"}
    df["in_current_models"] = [
        "pre-call + post-call" if n in MODEL_PRE else
        "post-call only" if n in MODEL_POST_EXTRA else
        label.get(n, "row filter (post-call keeps rows = 0)" if n == "paid_on_call" else "no")
        for n in df["name"]]
    return df[["name", "group", "meaning", "in_current_models", "source", "rule", "known_before_call", "used_by", "note"]]
