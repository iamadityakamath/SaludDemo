"""All calls for "Agent Call Reviews", from two places, read-only:

- Salud's final package (database schema `final`): the 84 real calls with Salud's call records and transcripts. Their
  call-level sentiment and QA score come from local Phase 2 files (`Phase 2/reports/`), when present.
- Phase 1 calls (schema `phase1`): 10 earlier real calls, 50 synthetic calls and calls pasted into the old dashboard,
  with stored sentiment (call and turn level) and AI reviews.

8 of the 10 earlier real Phase 1 calls are also in the final package. They appear once, as the real call (Salud's
transcript), with their Phase 1 AI review attached. Their stored turn-level sentiment was scored on the Phase 1 copy of
the transcript, whose lines differ, so only call-level sentiment is shown for them.
"""

from __future__ import annotations

import time

import pandas as pd

from .. import db
from ..config import get_settings
from ..schemas import AiReview, CallDetail, CallSummary, ChecklistItem, SentimentSummary, Turn
from ..transcripts import parse_turns

PHASE1_SOURCE = {"real": "real", "synthetic": "synthetic", "new": "pasted"}
SENTIMENT_FIELDS = list(SentimentSummary.model_fields)

Q_FINAL = '''
SELECT ch."salud_gphpvcontacthistoryId" AS contact_id, ch."salud_DateandTimeofContact" AS contacted_at,
       ch."salud_ConnectionType" AS connection, ch."salud_Contact_Type" AS contact_type,
       ch."salud_CallDisposition" AS disposition, ch."salud_CollectionResult" AS collection_result,
       ch."salud_ActionsTaken" AS actions, ch."salud_PaymentAmount" AS payment_amount,
       ch."salud_AgentCallTalkDuration" AS talk_seconds, ch."salud_PreferredLanguage" AS language,
       t.transcript, t.phase1_call_id
FROM final.contact_history ch
LEFT JOIN final.transcripts t ON t."salud_TCNRecordID" = ch."salud_TCNRecordID"
'''
Q_PHASE1 = "SELECT call_id, source, transcript FROM phase1.calls"
Q_PHASE1_SENTIMENT = "SELECT call_id, summary FROM phase1.call_sentiment"
Q_AI = ("SELECT call_id, model, prompt_version, summary, outcome, sentiment, quality_score "
        "FROM phase1.ai_analyses")

_cache: dict = {"at": 0.0, "records": None}


def _num(v) -> float | None:
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _local_csv(path, key="call_id") -> dict[str, dict]:
    """Call-level rows from a local Phase 2 file, keyed by Salud's contact id ({} if the file isn't there)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(r[key]): r for r in df.to_dict("records")}


def _sentiment(d: dict | None) -> SentimentSummary | None:
    if not d:
        return None
    vals = {f: d.get(f) for f in SENTIMENT_FIELDS}
    vals = {k: (v if isinstance(v, str) else _num(v)) for k, v in vals.items()}
    return SentimentSummary(**vals) if any(v is not None for v in vals.values()) else None


def load_records(force: bool = False) -> list[dict]:
    """One dict per call with everything the list view needs. Cached for `cache_seconds`."""
    s = get_settings()
    if not force and _cache["records"] is not None and time.time() - _cache["at"] < s.cache_seconds:
        return _cache["records"]
    final_rows = db.fetch_all(Q_FINAL)
    p1_rows = db.fetch_all(Q_PHASE1)
    p1_sent = {r["call_id"]: r["summary"] for r in db.fetch_all(Q_PHASE1_SENTIMENT)}
    ai = {r["call_id"]: r for r in db.fetch_all(Q_AI)}
    real_sent = _local_csv(s.real_sentiment_file)
    real_qa = _local_csv(s.real_qa_file)

    records, linked = [], set()
    for r in final_rows:
        p1 = r["phase1_call_id"]
        if p1:
            linked.add(p1)
        sent = _sentiment(real_sent.get(r["contact_id"])) or _sentiment(p1_sent.get(p1))
        qa = _num(real_qa.get(r["contact_id"], {}).get("qa_score"))
        records.append({
            "key": f"final-{r['contact_id']}", "source": "real", "dataset": "final", "db_id": r["contact_id"],
            "phase1_call_id": p1, "contacted_at": r["contacted_at"], "outcome": r["collection_result"],
            "transcript_text": r["transcript"], "sentiment": sent, "qa_score": qa,
            "ai": ai.get(p1) if p1 else None,
            "facts": {k: r[k] for k in ("connection", "contact_type", "disposition", "collection_result", "actions",
                                        "payment_amount", "talk_seconds", "language")},
        })
    for r in p1_rows:
        if r["call_id"] in linked:
            continue
        a = ai.get(r["call_id"])
        records.append({
            "key": f"p1-{r['call_id']}", "source": PHASE1_SOURCE.get(r["source"], "pasted"), "dataset": "phase1",
            "db_id": r["call_id"], "phase1_call_id": r["call_id"], "contacted_at": None,
            "outcome": a["outcome"] if a else None, "transcript_text": r["transcript"],
            "sentiment": _sentiment(p1_sent.get(r["call_id"])),
            "qa_score": _num(a["quality_score"]) if a else None, "ai": a, "facts": {},
        })
    _cache.update(at=time.time(), records=records)
    return records


def to_summary(rec: dict) -> CallSummary:
    sent: SentimentSummary | None = rec["sentiment"]
    return CallSummary(
        key=rec["key"], source=rec["source"], dataset=rec["dataset"], contacted_at=rec["contacted_at"],
        outcome=rec["outcome"], caller_end_sentiment=sent.caller_end if sent else None,
        caller_end_label=sent.caller_end_label if sent else None, qa_score=rec["qa_score"],
        has_ai_review=rec["ai"] is not None, turns=len(parse_turns(rec["transcript_text"])),
    )


def list_calls(source: str | None = None, search: str | None = None, limit: int = 50, offset: int = 0
               ) -> tuple[int, list[CallSummary]]:
    recs = load_records()
    if source:
        recs = [r for r in recs if r["source"] == source]
    if search:
        q = search.lower()
        recs = [r for r in recs if q in (r["transcript_text"] or "").lower() or q in (r["outcome"] or "").lower()]
    dated = sorted((r for r in recs if r["contacted_at"]), key=lambda r: r["contacted_at"], reverse=True)
    undated = sorted((r for r in recs if not r["contacted_at"]), key=lambda r: r["key"])
    recs = dated + undated                    # newest real calls first, then Phase 1 calls (no call date)
    return len(recs), [to_summary(r) for r in recs[offset: offset + limit]]


def _turn_sentiment(call_id: str | None) -> list[dict]:
    """Stored turn scores, in order. They cover the people in the call (agent and caller), not automated-system lines."""
    if not call_id:
        return []
    return db.fetch_all("SELECT turn, role, score, label FROM phase1.turn_sentiment WHERE call_id = %s ORDER BY turn",
                        (call_id,))


def attach_turn_sentiment(turns: list[dict], scores: list[dict]) -> list[dict]:
    """Give each non-system turn its stored score, in order, but only when counts and speaker roles line up exactly;
    otherwise return the turns without scores (never risk showing a score on the wrong line)."""
    people = [t for t in turns if t["role"] != "system"]
    if not scores or len(people) != len(scores) or any(
            t["role"] != s["role"] and t["role"] != "unknown" for t, s in zip(people, scores)):
        return turns
    for t, s in zip(people, scores):
        t["sentiment"], t["sentiment_label"] = _num(s["score"]), s["label"]
    return turns


def _ai_review(rec: dict) -> AiReview | None:
    a, cid = rec["ai"], rec["phase1_call_id"]
    if not a or not cid:
        return None
    checklist = db.fetch_all("SELECT item, answer, reason FROM phase1.ai_checklist WHERE call_id = %s ORDER BY item",
                             (cid,))
    coaching = db.fetch_all("SELECT tip FROM phase1.ai_coaching WHERE call_id = %s ORDER BY tip_no", (cid,))
    return AiReview(model=a["model"], prompt_version=a["prompt_version"], summary=a["summary"], outcome=a["outcome"],
                    sentiment=a["sentiment"], quality_score=_num(a["quality_score"]),
                    checklist=[ChecklistItem(**c) for c in checklist], coaching=[c["tip"] for c in coaching])


def get_call(key: str) -> CallDetail | None:
    rec = next((r for r in load_records() if r["key"] == key), None)
    if rec is None:
        return None
    turns = parse_turns(rec["transcript_text"])
    # turn scores belong to the Phase 1 copy of the transcript: attach them only when that is the transcript shown
    if rec["dataset"] == "phase1":
        turns = attach_turn_sentiment(turns, _turn_sentiment(rec["phase1_call_id"]))
    facts = {k: (_num(v) if k in ("payment_amount", "talk_seconds") else v) for k, v in rec["facts"].items()}
    return CallDetail(
        **to_summary(rec).model_dump(),
        transcript=[Turn(**t) for t in turns],
        sentiment=rec["sentiment"], ai_review=_ai_review(rec), call_facts=facts,
    )
