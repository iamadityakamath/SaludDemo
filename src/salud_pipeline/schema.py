"""Schema constants from the Oct 14, 2026 handoff (BIG_Schema.json / BIG_Synthetic_Schema.sql).

Option codes are mirrored here so the generator runs without the handoff files; the validator
cross-checks them against BIG_Schema.json when it is present.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_DIR = REPO_ROOT / "data" / "BIG_Schema_Only_Handoff"
SCHEMA_JSON = HANDOFF_DIR / "BIG_Schema.json"

# --- Table names (CRM entity views, reproduced in the big_synthetic schema) ---
GUARANTOR = "salud_gphpvguarantor"
ACCOUNTS = "salud_gphpvaccounts"
CONTACTS = "salud_gphpvcontacthistory"
BRIDGE = "salud_Account_X_ContactHistory"
RECEIPTS = "salud_gphpvreceipts"
QA_REVIEW = "salud_gphpvqualityreview"

# --- Column order per table, matching BIG_Synthetic_Schema.sql ---
COLUMNS: dict[str, list[str]] = {
    GUARANTOR: ["salud_gphpvguarantorId", "salud_DoNotCall"],
    ACCOUNTS: [
        "salud_gphpvaccountsId", "salud_GuarantorId", "salud_HBPB", "salud_FinancialClass",
        "salud_PlacementStatus", "salud_PlacementDate", "salud_LastPlacementDate",
        "salud_DischargeDate", "salud_PlacementPatientBalance", "salud_SelfPayBalance",
        "salud_SelfPayReceipts", "salud_SelfPayAdjustments", "salud_SelfPayBalanceDate",
        "salud_OnPaymentPlan", "salud_EpicPaymentPlanNumberofPayments",
        "salud_EpicPaymentPlanPaymentsMissed", "salud_EpicPaymentPlanLastPaymentDate",
        "salud_StatementCount", "salud_FirstStatementDate", "salud_LastStatementDate",
        "salud_LastCallDate", "salud_LastAccountContact", "salud_LastPatientPaymentFromTxnsDate",
        "salud_LastUpdatedfromClientSystem",
    ],
    CONTACTS: [
        "salud_gphpvcontacthistoryId", "salud_GuarantorId", "salud_DateandTimeofContact",
        "salud_ConnectionType", "salud_Contact_Type", "salud_CallDisposition",
        "salud_CollectionAttempt", "salud_CollectionResult", "salud_ActionsTaken",
        "salud_PaymentAmount", "salud_ReviewBalance", "salud_inquireaboutfinancialassistance",
        "salud_RequestedCallback", "salud_CallbackDate", "salud_needsomeoneelsetransfer",
        "salud_TransferredTo", "salud_PreferredLanguage", "salud_AgentCallTalkDuration",
        "salud_AgentCallWrapUpDuration", "salud_Agent", "salud_TCNRecordID", "salud_Note",
    ],
    BRIDGE: ["salud_gphpvaccountsid", "salud_gphpvcontacthistoryid"],
    RECEIPTS: [
        "salud_gphpvreceiptsId", "salud_ReceiptsId", "salud_Guarantor", "salud_PostDate",
        "salud_Amount", "salud_TXNType", "salud_PaymentType", "salud_ExcludeTransaction",
        "salud_HBPB", "salud_LastOutboundCall",
    ],
    QA_REVIEW: [
        "salud_gphpvqualityreviewId", "salud_QualityReviewId", "salud_reviewdate",
        "salud_AppropriateGreetings", "salud_AppropriateGreetingsComment",
        "salud_UpholdingCulture", "salud_UpholdingCultureComments",
        "salud_CapturingData", "salud_CapturingDataComments",
        "salud_UnderstandingCustomerNeeds", "salud_UnderstandingCustomerNeedsComments",
        "salud_CorrectInformation", "salud_CorrectInformationComments",
        "salud_FirstContactResolution", "salud_FirstContactResolutionComments",
        "salud_Compliance", "salud_ComplianceComments",
        "salud_CorrectCallStatusing", "salud_CorrectCallStatusingComments",
        "salud_Grammar", "salud_GrammarComments", "salud_lessonslearned",
    ],
}

# Max lengths for bounded text columns (nvarchar(n)/varchar(n)).
MAX_LEN = {
    (ACCOUNTS, "salud_FinancialClass"): 100,
    (CONTACTS, "salud_ActionsTaken"): 1700,
    (CONTACTS, "salud_Agent"): 200,
    (CONTACTS, "salud_TCNRecordID"): 100,
}

# --- Option codes ---
YES, NO = 785570000, 785570001          # two-option Yes/No sets (payment plan, DNC, QA, collection attempt)
HB, PB = 785570000, 785570001

PLACEMENT_ACTIVE, PLACEMENT_WITHDRAWN, PLACEMENT_ZERO = 785570000, 785570001, 785570002

TXN_PAYMENT, TXN_ADJUSTMENT = 785570000, 785570001
PAY_PATIENT, PAY_INSURANCE = 785570000, 785570001
TXN_EXCLUDE, TXN_INCLUDE = 785570000, 785570001

CONNECTION = {
    "inbound": 785570000,
    "inbound_returned": 785570001,
    "outbound_manual": 785570002,
    "outbound_campaign": 785570003,
    "outbound_no_connection": 785570004,
    "non_call": 785570005,
    "text_campaign": 785570006,
    "text_no_connection": 785570007,
    "email_campaign": 785570008,
    "email_no_connection": 785570009,
}

CONTACT_TYPE = {
    "patient": 785570000, "non_patient_guarantor": 785570001, "attorney": 785570002,
    "insurance": 785570003, "client": 785570004, "other": 785570005,
}

DISPOSITION = {
    "busy": 100000000, "fax": 100000001, "vm_no_message": 100000002, "vm_left_message": 100000003,
    "no_answer": 100000004, "spoke_with_csr": 100000005, "english_ivr": 100000006,
    "spanish_ivr": 100000007, "invalid_number": 100000008, "left_message_with_person": 100000009,
    "text_successful": 100000010, "text_failed": 100000011, "email_sent": 100000012,
    "hung_up": 100000013,
}

RESULT = {
    "paid_guarantor_full": 785570000,
    "paid_account_full": 785570001,
    "partial_payment": 785570002,
    "promised_to_pay": 785570003,
    "non_committal": 785570004,
    "contested_balance": 785570005,
    "would_not_pay": 785570006,
    "pursued_insurance": 785570007,
    "na_transferred": 785570008,
    "refund_request": 785570009,
    "applied_overpayment": 785570010,
    "paid_plan_full": 785570011,
    "paid_plan_partial": 785570012,
}

ACTION = {
    "shared_balance": 785570000,
    "provided_statement": 785570001,
    "provided_itemized": 785570002,
    "corrected_balance": 785570003,
    "provided_fa_options": 785570004,
    "applied_discount": 785570005,
    "set_up_plan": 785570006,
    "explained_cost_share": 785570007,
    "corrected_insurance": 785570009,
    "added_insurance": 785570010,
    "need_claim_review": 785570011,
    "fa_paperwork_sent": 785570013,
    "update_payment_info": 785570014,
    "requested_itemized": 785570020,
    "requested_statement": 785570023,
    "requested_fa_paperwork": 785570024,
    "requested_insurance_review": 785570029,
    "balance_review_completed": 785570031,
    "updated_contact_info": 785570033,
}

TRANSFERRED_TO = {
    "supervisor": 785570000, "another_csr": 785570001, "provider": 785570002,
    "other": 785570003, "switchboard": 785570004, "billing_office": 785570005,
}

LANGUAGE = {"english": 785570000, "spanish": 785570001, "other": 785570002}

# QA checklist: (key, score column, comment column, criterion text)
QA_ITEMS: list[tuple[str, str, str, str]] = [
    ("greeting", "salud_AppropriateGreetings", "salud_AppropriateGreetingsComment",
     "Appropriate Greetings"),
    ("culture", "salud_UpholdingCulture", "salud_UpholdingCultureComments",
     "Upholding the culture and values of the call center"),
    ("capturing_data", "salud_CapturingData", "salud_CapturingDataComments",
     "Accurately capturing key customer data"),
    ("understanding_needs", "salud_UnderstandingCustomerNeeds",
     "salud_UnderstandingCustomerNeedsComments", "Understanding customer needs"),
    ("correct_info", "salud_CorrectInformation", "salud_CorrectInformationComments",
     "Providing customers with correct and relevant Information"),
    ("first_contact_resolution", "salud_FirstContactResolution",
     "salud_FirstContactResolutionComments", "First-contact resolution"),
    ("compliance", "salud_Compliance", "salud_ComplianceComments",
     "Compliance with privacy acts"),
    ("call_statusing", "salud_CorrectCallStatusing", "salud_CorrectCallStatusingComments",
     "Correct call statusing"),
    ("grammar", "salud_Grammar", "salud_GrammarComments", "Grammar and spelling in Notes"),
]
QA_KEYS = [k for k, *_ in QA_ITEMS]

# Synthetic convention (handoff leaves serialization open): ActionsTaken = ';'-joined codes.
ACTIONS_SEP = ";"


def load_schema_json(path: Path = SCHEMA_JSON) -> dict | None:
    """Return the handoff metadata, or None when the (gitignored) handoff folder is absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def allowed_codes_from_json(meta: dict) -> dict[tuple[str, str], set[int]]:
    """(table, lower-case column) -> allowed codes, excluding QA N/A per the project checklist."""
    out: dict[tuple[str, str], set[int]] = {}
    for o in meta["option_codes"]:
        if o["entity"] == QA_REVIEW and o["label"] == "N/A":
            continue
        out.setdefault((o["entity"], o["field"]), set()).add(int(o["code"]))
    return out
