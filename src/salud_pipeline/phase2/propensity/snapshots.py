"""One row per patient as of a call date: point-in-time features + 30-day payment labels.

The tables can't be joined directly (a call covers up to 39 accounts, each with many transactions), so each
table is first aggregated to the patient as of the date t, then the pieces are put side by side:

    accounts placed on or before t        -> counts, balances, ages
    transactions posted before t's day    -> payment / insurance history
    contacts before t                     -> prior contact counts
    the call itself                       -> post-call fields (outcome, actions, quality, sentiment)
    transactions on t's day .. t + 30d    -> labels

Day granularity: real transaction dates carry no meaningful time of day, so 'before t' means an earlier
calendar day and a payment on the call day belongs to the outcome window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dictionary import COLUMNS
from .sources import World

WINDOW_DAYS = 30


def self_pay(tx: pd.DataFrame) -> pd.Series:
    """Patient payments (see dictionary.SELF_PAY_RULE). A missing Exclude flag means 'not provided' only when
    the whole column is absent (real package); a NULL flag in data that has the column never qualifies."""
    base = (tx["txn_type"] == "Payment") & (tx["payment_type"] == "Patient Payment")
    if tx["exclude"].isna().all():
        return base
    return base & (tx["exclude"] == "Include")


def _days(a: pd.Timestamp, b) -> float:
    return float((a - b).days) if pd.notna(b) else np.nan


def _text(v) -> str | None:
    return v if isinstance(v, str) and v != "" else None


def _flags(actions) -> dict:
    acts = [a.strip() for a in (_text(actions) or "").split(";") if a.strip()]
    has = lambda *keys: any(any(k.lower() in a.lower() for k in keys) for a in acts)  # noqa: E731
    return {
        "action_shared_balance": has("Verbally Shared Balance"),
        "action_payment_plan": has("Set Up Payment Plan", "Update Payment Plan"),
        "action_financial_assistance": has("Financial Assistance", "FA Paperwork"),
        "action_statement_or_bill": has("Statement", "Itemized Bill"),
        # insurance work only: "Explained Coinsurance/Deductible/Copay" is a billing explanation, not insurance
        "action_insurance": any("insurance" in a.lower() and "coinsurance" not in a.lower() for a in acts),
        "n_actions": len(acts),
    }


def _yes(v) -> bool | None:
    return None if _text(v) is None else v == "Yes"


def build(world: World, sentiment: pd.DataFrame | None = None, qa: pd.Series | None = None,
          window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """window_days: length of the outcome window (labels keep their 30d names).
    sentiment: per call_id, columns from sentiment.results.score_call summaries (optional).
    qa: per call_id, QA score 0-1 (optional)."""
    tx = world.tx.assign(post_date=pd.to_datetime(world.tx["post_date"]),
                         amount=pd.to_numeric(world.tx["amount"]))
    tx["day"] = tx["post_date"].dt.normalize()
    tx["is_patient_payment"] = self_pay(tx)
    tx["is_insurance_payment"] = (tx["txn_type"] == "Payment") & (tx["payment_type"] == "Insurance Payment")
    tx["is_adjustment"] = tx["txn_type"] == "Adjustment"
    tx_by_g = dict(tuple(tx.groupby("guarantor_id")))
    acc_by_g = dict(tuple(world.accounts.groupby("guarantor_id")))
    con_by_g = dict(tuple(world.contacts.dropna(subset=["guarantor_id"]).groupby("guarantor_id")))
    links = world.links.groupby("call_id").size()
    calls_by_g = dict(tuple(world.calls.dropna(subset=["guarantor_id"]).groupby("guarantor_id")))
    empty_tx, empty_acc = tx.iloc[0:0], world.accounts.iloc[0:0]
    end_day = world.data_end.normalize()

    rows = []
    for c in world.calls.itertuples(index=False):
        t, g = c.t, c.guarantor_id
        day = t.normalize()
        has_g = isinstance(g, str) and g != ""
        acc = acc_by_g.get(g, empty_acc) if has_g else empty_acc
        acc_t = acc[acc["placement_date"].dt.normalize() <= day]
        gtx = tx_by_g.get(g, empty_tx) if has_g else empty_tx
        hist = gtx[gtx["day"] < day]
        pay = hist[hist["is_patient_payment"]]
        win = gtx[gtx["is_patient_payment"] & (gtx["day"] >= day) & (gtx["day"] <= day + pd.Timedelta(days=window_days))]
        after = win[win["day"] > day]
        con = con_by_g.get(g) if has_g else None
        prior = con[con["t"] < t] if con is not None else world.contacts.iloc[0:0]
        prior_calls = calls_by_g.get(g) if has_g else None
        result = _text(c.collection_result) or ""

        def recent(df, n):
            return df[df["day"] >= day - pd.Timedelta(days=n)]

        p90, p365 = recent(pay, 90), recent(pay, 365)
        ins365 = recent(hist[hist["is_insurance_payment"]], 365)
        adj365 = recent(hist[hist["is_adjustment"]], 365)
        first_place = acc_t["placement_date"].min() if len(acc_t) else None
        pay_since_place = pay[pay["day"] >= first_place.normalize()]["amount"].sum() if first_place is not None else 0.0
        placement_balance = float(acc_t["placement_balance"].sum())
        balance_at_t = max(placement_balance - float(pay_since_place), 0.0)
        disch = acc_t["discharge_date"][acc_t["discharge_date"] <= t]
        paid_amount = float(win["amount"].sum())
        payment_on_call = c.payment_on_call if pd.notna(c.payment_on_call) else 0.0

        row = {
            "snapshot_id": f"{world.name}/{c.call_id}", "source": world.name, "guarantor_id": g if has_g else None,
            "call_id": c.call_id, "tcn_record_id": c.tcn_record_id, "as_of": t, "agent": c.agent,
            "n_linked_accounts": int(links.get(c.call_id, 0)),
            "label_reliable": bool(has_g and len(acc) > 0),
            "window_complete": bool(day + pd.Timedelta(days=window_days) <= end_day),
            "paid_on_call": bool(payment_on_call > 0 or result.startswith("Paid")),

            "n_accounts": len(acc_t),
            "n_hospital_accounts": int((acc_t["hbpb"] == "Hospital Account").sum()),
            "n_physician_accounts": int((acc_t["hbpb"] == "Physician Account").sum()),
            "placement_balance": round(placement_balance, 2),
            "balance_at_t": round(balance_at_t, 2),
            "days_since_first_placement": _days(t, first_place),
            "days_since_last_placement": _days(t, acc_t["placement_date"].max() if len(acc_t) else None),
            "days_since_last_discharge": _days(t, disch.max() if len(disch) else None),

            "ever_paid": bool(len(pay)),
            "n_payments_90d": len(p90), "amount_paid_90d": round(float(p90["amount"].sum()), 2),
            "n_payments_365d": len(p365), "amount_paid_365d": round(float(p365["amount"].sum()), 2),
            "n_payments_ever": len(pay), "amount_paid_ever": round(float(pay["amount"].sum()), 2),
            "days_since_last_payment": _days(day, pay["day"].max() if len(pay) else None),
            "paid_share_so_far": round(min(float(pay_since_place) / placement_balance, 1.0), 4)
            if placement_balance > 0 else np.nan,
            "last_payment_amount": round(float(pay.loc[pay["day"] == pay["day"].max(), "amount"].sum()), 2)
            if len(pay) else 0.0,

            "n_insurance_payments_365d": len(ins365), "amount_insurance_365d": round(float(ins365["amount"].sum()), 2),
            "n_adjustments_365d": len(adj365), "amount_adjustments_365d": round(float(adj365["amount"].abs().sum()), 2),

            "n_prior_contacts_90d": int((prior["t"] >= t - pd.Timedelta(days=90)).sum()),
            "n_prior_calls": int((prior_calls["t"] < t).sum()) if prior_calls is not None else 0,
            "days_since_last_contact": _days(t, prior["t"].max() if len(prior) else None),

            "inbound": (_text(c.connection) or "").startswith("Inbound"),
            "connection_type": _text(c.connection), "contact_type": _text(c.contact_type),
            "disposition": _text(c.disposition),
            "spoke_with_agent": c.disposition == "Spoke with CSR",
            "collection_attempted": c.collection_attempt == "Yes",
            "collection_result": _text(c.collection_result),
            "promised_to_pay": result == "Promised to Pay",
            "non_committal": result == "Non-Committal",
            "contested_or_refused": result in ("Contested Balance", "Said Would Not Pay"),
            "pursued_insurance": result == "Pursued Insurance Payment",
            "transferred": result == "N/A, Transferred" or c.transfer_needed == "Yes",
            **_flags(c.actions),
            "reviewed_balance": _yes(c.review_balance),
            "asked_financial_assistance": _yes(c.asked_fa),
            "talk_seconds": c.talk_seconds,
            "payment_on_call": payment_on_call,
            "preferred_language": c.language,

            "paid_30d": bool(len(win)),
            "paid_after_call_30d": bool(len(after)),
            "paid_amount_30d": round(paid_amount, 2),
            "days_to_first_payment": _days(win["day"].min(), day) if len(win) else np.nan,
            "paid_fraction_30d": round(min(paid_amount / balance_at_t, 1.0), 4) if balance_at_t > 0 else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    sent_cols = {"caller_overall": "caller_sentiment_overall", "caller_end": "caller_sentiment_end",
                 "caller_shift": "caller_sentiment_shift", "agent_overall": "agent_sentiment_overall"}
    if sentiment is not None and len(sentiment):
        s = sentiment.set_index("call_id")
        for src, dst in sent_cols.items():
            out[dst] = out["call_id"].map(s[src]) if src in s else np.nan
        out["has_caller_turns"] = out["call_id"].map(s["caller_turns"].fillna(0) > 0)
    else:
        for dst in sent_cols.values():
            out[dst] = np.nan
        out["has_caller_turns"] = np.nan
    out["qa_score"] = out["call_id"].map(qa) if qa is not None else np.nan

    order = [c.name for c in COLUMNS]
    missing = [c for c in order if c not in out.columns]
    if missing:
        raise RuntimeError(f"Snapshot builder didn't produce columns: {missing}")
    return out[order]


# ---------- fixed-date design: every patient scored on one date

def fixed_date_world(world: World, as_of: pd.Timestamp) -> World:
    """The same world, but with one pseudo 'call' per patient on `as_of` instead of the real calls: the
    'score every patient today' design. Call fields are empty, so only pre-call features are meaningful."""
    from dataclasses import replace

    g = world.accounts["guarantor_id"].dropna().unique()
    calls = pd.DataFrame({c: None for c in world.calls.columns}, index=range(len(g)))
    calls["call_id"] = [f"asof-{x}" for x in g]
    calls["guarantor_id"] = g
    calls["t"] = pd.Timestamp(as_of)
    for c in ("payment_on_call", "talk_seconds"):
        calls[c] = np.nan
    return replace(world, calls=calls, transcripts={})


def open_balance(snap: pd.DataFrame) -> pd.DataFrame:
    """Patients the worklist would hold on the as-of date: at least one placed account and a balance left."""
    return snap[(snap["n_accounts"] > 0) & (snap["balance_at_t"] > 0)].reset_index(drop=True)


# ---------- call-level extras: sentiment (local model), real QA (LLM) and synthetic QA ----------

def score_sentiment(world: World, cache: Path) -> pd.DataFrame:
    """Caller/agent sentiment per call, from the local ONNX model. Cached per world, since synthetic takes a while."""
    from ...sentiment.model import SentimentModel
    from ...sentiment.results import score_call

    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["call_id"])
    todo = [cid for cid in world.transcripts if cid not in set(done["call_id"])]
    if todo:
        model = SentimentModel()
        new, start = [], len(done)
        for i, cid in enumerate(todo, 1):
            _, summary = score_call(world.transcripts[cid], model)
            new.append({"call_id": cid, **summary})
            if i % 200 == 0 or i == len(todo):
                print(f"  sentiment {world.name}: {start + i}/{start + len(todo)}", flush=True)
                done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
                new = []
                cache.parent.mkdir(parents=True, exist_ok=True)
                done.to_csv(cache, index=False)
    return done


def synthetic_qa(world_dir: Path) -> pd.Series:
    truth = pd.read_csv(world_dir / "truth" / "truth_calls.csv", usecols=["contact_id", "quality_score_transcript_items"])
    return truth.set_index("contact_id")["quality_score_transcript_items"]


def score_qa(world: World, out_csv: Path, limit: int | None = None) -> pd.DataFrame:
    """Stage 1 QA of each call with the dashboard's prompt (phase1.prompt, live version) through the configured LLM.

    Sends the redacted transcripts to the LLM provider. Every review is cached in the phase1 LLM cache (keyed by
    model, prompt version and transcript), so a run stopped by the provider's daily limit resumes where it ended.
    Writes one row per call to out_csv (local, gitignored): the QA score (Yes / (Yes + No) over the transcript items,
    as on the dashboard), each item's answer, and the AI's outcome and sentiment labels. Nothing goes to the database.
    """
    from ...llm_client import LLMEndpoint
    from ...paths import LLM_CACHE
    from ...phase1.checklist import TRANSCRIPT_ITEMS
    from ...phase1.compare import call_quality
    from ...phase1.prompt import PROMPT_VERSION, CallAssistant

    ep = LLMEndpoint()
    if not ep.model:
        raise RuntimeError("No LLM configured: set GROQ_API_KEY in .env")
    ep.max_retries = 8
    assistant = CallAssistant(ep, LLM_CACHE)
    rows, counts = [], {"from_cache": 0, "new_llm_requests": 0, "failed": 0, "not_done": 0}
    stopped = None
    for i, (cid, text) in enumerate(sorted(world.transcripts.items()), 1):
        a = assistant.cached(text)
        if a is not None:
            counts["from_cache"] += 1
        elif stopped or (limit is not None and counts["new_llm_requests"] >= limit):
            counts["not_done"] += 1
            continue
        else:
            try:
                a = assistant.review(text)
                counts["new_llm_requests"] += 1
            except (RuntimeError, ValueError, OSError) as e:   # OSError covers network timeouts
                stopped = f"{type(e).__name__}: {e}"[:200]
                counts["failed"] += 1
                counts["not_done"] += 1
                print(f"  stopping: {stopped}")
                continue
            if not a.get("ok"):
                counts["failed"] += 1
                print(f"  call {i}: the reply could not be parsed; will retry next run")
                continue
        answers = {k: ((a.get("checklist") or {}).get(k) or {}).get("answer") for k in TRANSCRIPT_ITEMS}
        rows.append({"call_id": cid, "qa_score": call_quality(answers), **{f"qa_{k}": v for k, v in answers.items()},
                     "ai_outcome": (a.get("outcome") or {}).get("label"),
                     "ai_sentiment": (a.get("sentiment") or {}).get("label"),
                     "model": ep.model, "prompt_version": PROMPT_VERSION})
        print(f"  QA {len(rows)}/{len(world.transcripts)}", flush=True)
    done = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done.to_csv(out_csv, index=False)
    print(f"{len(done)} of {len(world.transcripts)} calls scored ({ep.model} / {PROMPT_VERSION}): "
          f"{counts['new_llm_requests']} new LLM requests, {counts['from_cache']} from cache, {counts['failed']} failed"
          + (f"; {counts['not_done']} left, run again later" if counts["not_done"] else ""))
    return done
