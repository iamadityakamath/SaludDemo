"""Load the real package (Neon schema `final`) and the synthetic world into one common shape.

Both worlds become the same five frames with plain column names and option *labels* (synthetic codes are
translated with the handoff's option codes), so one feature builder serves both:

    calls     one row per recorded call (real: all 84; synthetic: calls that reached an agent or the phone menu,
              or left a message — the same population)
    contacts  every contact attempt (real: the calls themselves; synthetic: also texts, emails, no-answers)
    accounts  one row per account (visit) with its static placement facts
    links     account <-> call pairs
    tx        transactions: post date, amount, type, payer, include flag (None when not provided)

Only columns known at placement or dated rows are carried; the account snapshot fields that change after a
call (current balance, status, statement count, last-contact dates) are deliberately not loaded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...paths import PHASE2_SYNTHETIC_WORLD, SCHEMA_HANDOFF

HANDOFF_JSON = SCHEMA_HANDOFF / "BIG_Schema.json"
RECORDED_DISPOSITIONS = ["Spoke with CSR", "Hung Up", "Sent to English IVR", "Sent to Spanish IVR",
                         "Left Message With Person", "Voicemail Did Leave Message"]

# CRM column -> common name, per frame.
CALL_COLS = {
    "salud_gphpvcontacthistoryId": "call_id", "salud_GuarantorId": "guarantor_id",
    "salud_DateandTimeofContact": "t", "salud_ConnectionType": "connection", "salud_Contact_Type": "contact_type",
    "salud_CallDisposition": "disposition", "salud_CollectionAttempt": "collection_attempt",
    "salud_CollectionResult": "collection_result", "salud_ActionsTaken": "actions",
    "salud_PaymentAmount": "payment_on_call", "salud_ReviewBalance": "review_balance",
    "salud_inquireaboutfinancialassistance": "asked_fa", "salud_RequestedCallback": "requested_callback",
    "salud_needsomeoneelsetransfer": "transfer_needed", "salud_PreferredLanguage": "language",
    "salud_AgentCallTalkDuration": "talk_seconds", "salud_Agent": "agent", "salud_TCNRecordID": "tcn_record_id",
}
ACCOUNT_COLS = {
    "salud_gphpvaccountsId": "account_id", "salud_GuarantorId": "guarantor_id", "salud_HBPB": "hbpb",
    "salud_PlacementDate": "placement_date", "salud_DischargeDate": "discharge_date",
    "salud_PlacementPatientBalance": "placement_balance",
}
LINK_COLS = {"salud_gphpvaccountsid": "account_id", "salud_gphpvcontacthistoryid": "call_id"}
TX_COLS = {
    "salud_gphpvreceiptsId": "tx_id", "salud_ReceiptsId": "account_id", "salud_Guarantor": "guarantor_id",
    "salud_PostDate": "post_date", "salud_Amount": "amount", "salud_TXNType": "txn_type",
    "salud_PaymentType": "payment_type", "salud_ExcludeTransaction": "exclude",
}
CODED = {  # (entity, CRM column) for every option field we translate
    "salud_gphpvcontacthistory": ["salud_ConnectionType", "salud_Contact_Type", "salud_CallDisposition",
                                  "salud_CollectionAttempt", "salud_CollectionResult", "salud_ActionsTaken",
                                  "salud_ReviewBalance", "salud_inquireaboutfinancialassistance",
                                  "salud_RequestedCallback", "salud_needsomeoneelsetransfer",
                                  "salud_PreferredLanguage"],
    "salud_gphpvaccounts": ["salud_HBPB"],
    "salud_gphpvreceipts": ["salud_TXNType", "salud_PaymentType", "salud_ExcludeTransaction"],
}


@dataclass
class World:
    name: str                    # 'real' | 'synthetic'
    calls: pd.DataFrame
    contacts: pd.DataFrame
    accounts: pd.DataFrame
    links: pd.DataFrame
    tx: pd.DataFrame
    data_end: pd.Timestamp       # last date transactions are known for
    transcripts: dict[str, str]  # call_id -> transcript text


# ---------- helpers ----------

def option_labels(handoff_json: Path = HANDOFF_JSON) -> dict[tuple[str, str], dict[int, str]]:
    """(entity, lower-case field) -> {code: label}, from the Oct 14 handoff."""
    out: dict[tuple[str, str], dict[int, str]] = {}
    for o in json.loads(handoff_json.read_text())["option_codes"]:
        out.setdefault((o["entity"], o["field"]), {})[int(o["code"])] = o["label"]
    return out


def _decode(series: pd.Series, codes: dict[int, str], multi: bool = False) -> pd.Series:
    def one(v):
        if pd.isna(v) or v == "":
            return None
        if multi:
            return "; ".join(codes[int(float(c))] for c in str(v).split(";") if c.strip())
        return codes[int(float(v))]
    return series.map(one)


def _finish(calls, contacts, accounts, links, tx) -> tuple:
    for df, cols in ((calls, ["t"]), (contacts, ["t"]), (accounts, ["placement_date", "discharge_date"]),
                     (tx, ["post_date"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    for c in ("payment_on_call", "talk_seconds"):
        calls[c] = pd.to_numeric(calls[c])
    accounts["placement_balance"] = pd.to_numeric(accounts["placement_balance"])
    tx["amount"] = pd.to_numeric(tx["amount"])
    return calls, contacts, accounts, links, tx


# ---------- real: Neon schema `final` ----------

def load_real(url: str | None = None) -> World:
    import psycopg

    from ...phase1.store import database_url

    def q(conn, table, cols):
        sel = ", ".join(f'"{c}"' for c in cols)
        return pd.DataFrame(conn.execute(f'SELECT {sel} FROM final."{table}"').fetchall(), columns=cols)

    with psycopg.connect(url or database_url()) as conn:
        calls = q(conn, "contact_history", list(CALL_COLS)).rename(columns=CALL_COLS)
        accounts = q(conn, "accounts", list(ACCOUNT_COLS)).rename(columns=ACCOUNT_COLS)
        links = q(conn, "account_contact_links", list(LINK_COLS)).rename(columns=LINK_COLS)
        tx = q(conn, "transactions", [c for c in TX_COLS if c != "salud_ExcludeTransaction"]).rename(columns=TX_COLS)
        texts = dict(conn.execute('SELECT "salud_TCNRecordID", transcript FROM final.transcripts').fetchall())
    tx["exclude"] = None  # not in the final package
    for c in ("payment_on_call", "talk_seconds", "placement_balance", "amount"):
        for df in (calls, accounts, tx):
            if c in df:
                df[c] = df[c].astype(float)
    calls, contacts, accounts, links, tx = _finish(calls, calls.copy(), accounts, links, tx)
    transcripts = {cid: texts[tcn] for cid, tcn in zip(calls["call_id"], calls["tcn_record_id"]) if tcn in texts}
    return World("real", calls, contacts, accounts, links, tx, tx["post_date"].max(), transcripts)


# ---------- synthetic: data/phase2/synthetic_world ----------

def load_synthetic(world: Path = PHASE2_SYNTHETIC_WORLD, handoff_json: Path = HANDOFF_JSON) -> World:
    crm = world / "crm"
    labels = option_labels(handoff_json)
    raw = {
        "salud_gphpvcontacthistory": pd.read_csv(crm / "salud_gphpvcontacthistory.csv", dtype=str),
        "salud_gphpvaccounts": pd.read_csv(crm / "salud_gphpvaccounts.csv", dtype=str),
        "salud_gphpvreceipts": pd.read_csv(crm / "salud_gphpvreceipts.csv", dtype=str),
    }
    for entity, cols in CODED.items():
        for c in cols:
            raw[entity][c] = _decode(raw[entity][c], labels[(entity, c.lower())], multi=c == "salud_ActionsTaken")

    contacts = raw["salud_gphpvcontacthistory"][list(CALL_COLS)].rename(columns=CALL_COLS)
    tdir = world / "transcripts"
    have = {p.stem for p in tdir.glob("*.txt")} if tdir.exists() else set()
    # Recorded calls, like the real package: phone calls that reached an agent, the phone menu, or left a message.
    recorded = contacts["connection"].fillna("").str.match(r"(Inbound|Outbound) Call") & \
        contacts["disposition"].isin(RECORDED_DISPOSITIONS)
    calls = contacts[recorded | contacts["call_id"].isin(have)].copy()
    accounts = raw["salud_gphpvaccounts"][list(ACCOUNT_COLS)].rename(columns=ACCOUNT_COLS)
    links = pd.read_csv(crm / "salud_Account_X_ContactHistory.csv", dtype=str).rename(columns=LINK_COLS) \
        .drop_duplicates()
    tx = raw["salud_gphpvreceipts"][list(TX_COLS)].rename(columns=TX_COLS)
    calls, contacts, accounts, links, tx = _finish(calls, contacts, accounts, links, tx)
    transcripts = {cid: (tdir / f"{cid}.txt").read_text() for cid in calls["call_id"] if cid in have}
    return World("synthetic", calls, contacts, accounts, links, tx, tx["post_date"].max(), transcripts)
