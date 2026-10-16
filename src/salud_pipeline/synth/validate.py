"""Integrity checks for a generated run: keys, codes, completeness, dates, money, labels."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import labels
from .. import schema as S

PK = {
    S.GUARANTOR: ["salud_gphpvguarantorId"], S.ACCOUNTS: ["salud_gphpvaccountsId"],
    S.CONTACTS: ["salud_gphpvcontacthistoryId"], S.RECEIPTS: ["salud_gphpvreceiptsId"],
    S.QA_REVIEW: ["salud_gphpvqualityreviewId"],
    S.BRIDGE: ["salud_gphpvaccountsid", "salud_gphpvcontacthistoryid"],
}
FK = [  # (child table, child col, parent table, parent col) -- the 8 handoff relationships
    (S.BRIDGE, "salud_gphpvaccountsid", S.ACCOUNTS, "salud_gphpvaccountsId"),
    (S.BRIDGE, "salud_gphpvcontacthistoryid", S.CONTACTS, "salud_gphpvcontacthistoryId"),
    (S.ACCOUNTS, "salud_GuarantorId", S.GUARANTOR, "salud_gphpvguarantorId"),
    (S.CONTACTS, "salud_GuarantorId", S.GUARANTOR, "salud_gphpvguarantorId"),
    (S.QA_REVIEW, "salud_QualityReviewId", S.CONTACTS, "salud_gphpvcontacthistoryId"),
    (S.RECEIPTS, "salud_ReceiptsId", S.ACCOUNTS, "salud_gphpvaccountsId"),
    (S.RECEIPTS, "salud_LastOutboundCall", S.CONTACTS, "salud_gphpvcontacthistoryId"),
    (S.RECEIPTS, "salud_Guarantor", S.GUARANTOR, "salud_gphpvguarantorId"),
]
BIT_COLS = ["salud_ReviewBalance", "salud_inquireaboutfinancialassistance",
            "salud_RequestedCallback", "salud_needsomeoneelsetransfer"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _codes_from_constants() -> dict[tuple[str, str], set[int]]:
    yn = {S.YES, S.NO}
    out = {
        (S.ACCOUNTS, "salud_hbpb"): {S.HB, S.PB},
        (S.ACCOUNTS, "salud_placementstatus"): {S.PLACEMENT_ACTIVE, S.PLACEMENT_WITHDRAWN, S.PLACEMENT_ZERO},
        (S.ACCOUNTS, "salud_onpaymentplan"): yn,
        (S.GUARANTOR, "salud_donotcall"): yn,
        (S.CONTACTS, "salud_connectiontype"): set(S.CONNECTION.values()),
        (S.CONTACTS, "salud_contact_type"): set(S.CONTACT_TYPE.values()),
        (S.CONTACTS, "salud_calldisposition"): set(S.DISPOSITION.values()),
        (S.CONTACTS, "salud_collectionattempt"): yn,
        (S.CONTACTS, "salud_collectionresult"): set(S.RESULT.values()),
        (S.CONTACTS, "salud_transferredto"): set(S.TRANSFERRED_TO.values()),
        (S.CONTACTS, "salud_preferredlanguage"): set(S.LANGUAGE.values()),
        (S.RECEIPTS, "salud_txntype"): {S.TXN_PAYMENT, S.TXN_ADJUSTMENT},
        (S.RECEIPTS, "salud_paymenttype"): {S.PAY_PATIENT, S.PAY_INSURANCE},
        (S.RECEIPTS, "salud_excludetransaction"): {S.TXN_EXCLUDE, S.TXN_INCLUDE},
        (S.RECEIPTS, "salud_hbpb"): {S.HB, S.PB},
    }
    for _, score_col, _, _ in S.QA_ITEMS:
        out[(S.QA_REVIEW, score_col.lower())] = yn
    return out


def validate(t: dict[str, pd.DataFrame], cutoff: pd.Timestamp, horizon_days: int) -> list[Check]:
    checks: list[Check] = []

    # Option codes in the mirrored constants must agree with the handoff JSON when present.
    allowed = _codes_from_constants()
    meta = S.load_schema_json()
    if meta is not None:
        from_json = S.allowed_codes_from_json(meta)
        drift = [k for k, v in allowed.items() if k in from_json and not v <= from_json[k]]
        checks.append(Check("option codes match BIG_Schema.json", not drift, f"drift: {drift}" if drift else ""))
        actions_ok = {int(o["code"]) for o in meta["option_codes"] if o["field"] == "salud_actionstaken"}
        checks.append(Check("ActionsTaken codes in schema", set(S.ACTION.values()) <= actions_ok))

    for table, cols in PK.items():
        dup = t[table].duplicated(cols).sum()
        nulls = t[table][cols].isna().any(axis=1).sum()
        checks.append(Check(f"PK unique/non-null: {table}", dup == 0 and nulls == 0, f"dup={dup} null={nulls}"))

    for child, ccol, parent, pcol in FK:
        vals = t[child][ccol].dropna()
        missing = (~vals.isin(set(t[parent][pcol]))).sum()
        checks.append(Check(f"FK {child}.{ccol}", missing == 0, f"orphans={missing}"))

    for (table, col_lower), codes in allowed.items():
        col = next(c for c in t[table].columns if c.lower() == col_lower)
        vals = t[table][col].dropna().astype(int)
        bad = (~vals.isin(codes)).sum()
        checks.append(Check(f"codes {table}.{col}", bad == 0, f"bad={bad}"))

    c = t[S.CONTACTS]
    for col in BIT_COLS:
        bad = (~c[col].dropna().isin([0, 1])).sum()
        checks.append(Check(f"bit {col}", bad == 0, f"bad={bad}"))
    acts = c["salud_ActionsTaken"].dropna().str.split(S.ACTIONS_SEP).explode().astype(int)
    checks.append(Check("ActionsTaken codes known", acts.isin(set(S.ACTION.values())).all()))

    for (table, col), n in S.MAX_LEN.items():
        too_long = (t[table][col].dropna().str.len() > n).sum()
        checks.append(Check(f"max length {table}.{col} <= {n}", too_long == 0, f"over={too_long}"))

    qa = t[S.QA_REVIEW]
    score_cols = [s for _, s, _, _ in S.QA_ITEMS]
    comment_cols = [cm for _, _, cm, _ in S.QA_ITEMS]
    incomplete = qa[score_cols + comment_cols].isna().any(axis=1).sum()
    checks.append(Check("QA reviews complete (9 scores + comments)", incomplete == 0, f"incomplete={incomplete}"))

    # Bridge: every contact covers >=1 account of the same guarantor.
    br = t[S.BRIDGE].merge(t[S.ACCOUNTS][["salud_gphpvaccountsId", "salud_GuarantorId"]],
                           left_on="salud_gphpvaccountsid", right_on="salud_gphpvaccountsId")
    br = br.merge(c[["salud_gphpvcontacthistoryId", "salud_GuarantorId"]], left_on="salud_gphpvcontacthistoryid",
                  right_on="salud_gphpvcontacthistoryId", suffixes=("_acct", "_contact"))
    mism = (br["salud_GuarantorId_acct"] != br["salud_GuarantorId_contact"]).sum()
    uncovered = (~c["salud_gphpvcontacthistoryId"].isin(t[S.BRIDGE]["salud_gphpvcontacthistoryid"])).sum()
    checks.append(Check("bridge guarantor consistent", mism == 0, f"mismatch={mism}"))
    checks.append(Check("every contact bridged to an account", uncovered == 0, f"unbridged={uncovered}"))

    # Event order.
    a = t[S.ACCOUNTS]
    bad = ((a["salud_DischargeDate"] > a["salud_PlacementDate"])
           | (a["salud_PlacementDate"] > a["salud_LastPlacementDate"])
           | (a["salud_FirstStatementDate"] > a["salud_LastStatementDate"])).sum()
    checks.append(Check("account dates ordered", bad == 0, f"bad={bad}"))
    late = (c["salud_DateandTimeofContact"] > cutoff).sum()
    checks.append(Check("contacts on/before cutoff", late == 0, f"late={late}"))
    rv = qa.merge(c[["salud_gphpvcontacthistoryId", "salud_DateandTimeofContact"]],
                  left_on="salud_QualityReviewId", right_on="salud_gphpvcontacthistoryId")
    bad = (rv["salud_reviewdate"] < rv["salud_DateandTimeofContact"].dt.normalize()).sum()
    checks.append(Check("QA review after call", bad == 0, f"bad={bad}"))
    r = t[S.RECEIPTS].merge(a[["salud_gphpvaccountsId", "salud_PlacementDate"]],
                            left_on="salud_ReceiptsId", right_on="salud_gphpvaccountsId")
    bad = (r["salud_PostDate"] < r["salud_PlacementDate"]).sum()
    checks.append(Check("receipts after placement", bad == 0, f"bad={bad}"))

    # Money: placement balance - included receipts/adjustments == current balance.
    inc = t[S.RECEIPTS][t[S.RECEIPTS]["salud_ExcludeTransaction"] == S.TXN_INCLUDE]
    posted = inc.groupby("salud_ReceiptsId")["salud_Amount"].sum()
    recon = a.set_index("salud_gphpvaccountsId")
    diff = (recon["salud_PlacementPatientBalance"] - posted.reindex(recon.index).fillna(0)
            - recon["salud_SelfPayBalance"]).abs()
    checks.append(Check("balance reconciles to receipts", (diff < 0.02).all(), f"max diff={diff.max():.2f}"))
    neg = (a["salud_SelfPayBalance"] < 0).sum()
    checks.append(Check("no negative balances", neg == 0, f"neg={neg}"))

    # Labels from receipts reproduce the planted outcome exactly.
    truth = t["truth_accounts"]
    out = labels.account_outcomes(a, t[S.RECEIPTS], cutoff, horizon_days)
    m = truth[truth["open_at_cutoff"]].merge(out, left_on="account_id", right_on="salud_gphpvaccountsId")
    agree = (m["paid_any"] == m["planted_paid"].astype(bool)).mean()
    checks.append(Check("receipt-derived labels == planted outcome", agree == 1.0, f"agreement={agree:.4f}"))
    return checks
