"""The Phase 1 tables in the team database (schema `phase1`), shared with the team dashboard website:
calls (with transcripts), call and turn sentiment, AI analyses with their checklist and coaching rows.

What this app writes, and nothing else:
- a new call, when a transcript is uploaded (source 'new', call_id 'new/<time>-<id>');
- a call's sentiment (replaces that call's stored scores);
- a call's AI review (added if none exists; Re-analyze replaces that one call's review).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from .. import db
from .checklist import TRANSCRIPT_ITEMS, call_quality


def analysis_list() -> list[dict]:
    """Every call, newest uploads first, with what has been computed for it."""
    return db.fetch_all(
        "SELECT c.call_id, c.source, c.file_id, c.added_by, c.added_at, "
        "a.call_id IS NOT NULL AS analyzed, s.call_id IS NOT NULL AS scored, "
        "s.summary->>'caller_end_label' AS end_label "
        "FROM phase1.calls c LEFT JOIN phase1.ai_analyses a USING (call_id) "
        "LEFT JOIN phase1.call_sentiment s USING (call_id) "
        "ORDER BY c.source = 'new' DESC, c.added_at DESC NULLS LAST, c.source = 'synthetic', c.file_id")


def pending_calls(need_review: bool = True) -> list[str]:
    missing = "s.call_id IS NULL" + (" OR a.call_id IS NULL" if need_review else "")
    rows = db.fetch_all(
        "SELECT c.call_id FROM phase1.calls c LEFT JOIN phase1.ai_analyses a USING (call_id) "
        "LEFT JOIN phase1.call_sentiment s USING (call_id) WHERE " + missing +
        " ORDER BY c.source = 'new' DESC, c.added_at DESC NULLS LAST, c.call_id")
    return [r["call_id"] for r in rows]


def add_new_call(transcript: str, added_by: str) -> str:
    if not transcript.strip():
        raise ValueError("the transcript is empty")
    now = datetime.now(timezone.utc)
    file_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    with db.pool().connection() as c:
        c.execute("INSERT INTO phase1.calls (call_id, source, file_id, transcript, added_by, added_at) "
                  "VALUES (%s, 'new', %s, %s, %s, %s)", (f"new/{file_id}", file_id, transcript.strip(), added_by[:100], now))
    return f"new/{file_id}"


def transcript(call_id: str) -> str | None:
    rows = db.fetch_all("SELECT transcript FROM phase1.calls WHERE call_id = %s", (call_id,))
    return rows[0]["transcript"] if rows else None


def ai_analysis(call_id: str) -> dict | None:
    rows = db.fetch_all("SELECT raw FROM phase1.ai_analyses WHERE call_id = %s", (call_id,))
    return rows[0]["raw"] if rows else None


def add_ai_analysis(call_id: str, analysis: dict, model: str, prompt_version: str) -> bool:
    """Insert one review with its checklist and coaching rows. False if the call already has one."""
    cl = analysis.get("checklist") or {}
    with db.pool().connection() as c, c.transaction():
        inserted = c.execute(
            "INSERT INTO phase1.ai_analyses (call_id, model, prompt_version, analyzed_at, sentiment, outcome, "
            "quality_score, summary, quotes_dropped, raw) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (call_id) DO NOTHING RETURNING call_id",
            (call_id, model, prompt_version, analysis.get("analyzed_at"),
             (analysis.get("sentiment") or {}).get("label"), (analysis.get("outcome") or {}).get("label"),
             call_quality({k: (cl.get(k) or {}).get("answer") for k in TRANSCRIPT_ITEMS}),
             analysis.get("summary"), (analysis.get("evidence_check") or {}).get("dropped"), Jsonb(analysis))).fetchone()
        if not inserted:
            return False
        for k in TRANSCRIPT_ITEMS:
            item = cl.get(k) or {}
            ev = item.get("evidence") or []
            c.execute("INSERT INTO phase1.ai_checklist (call_id, item, answer, reason, evidence, needs_review) "
                      "VALUES (%s,%s,%s,%s,%s,%s)",
                      (call_id, k, item.get("answer"), item.get("reason"), Jsonb(ev), (not item.get("answer")) or not ev))
        for i, t in enumerate(analysis.get("coaching") or [], 1):
            c.execute("INSERT INTO phase1.ai_coaching (call_id, tip_no, item, tip, turn) VALUES (%s,%s,%s,%s,%s)",
                      (call_id, i, t.get("item"), t.get("tip"), t.get("turn")))
    return True


def replace_ai_analysis(call_id: str, analysis: dict, model: str, prompt_version: str) -> None:
    """Re-analyze: drop this one call's review (with its checklist and coaching rows) and add the new one."""
    with db.pool().connection() as c, c.transaction():
        for table in ("ai_coaching", "ai_checklist", "ai_analyses"):
            c.execute(f"DELETE FROM phase1.{table} WHERE call_id = %s", (call_id,))
    add_ai_analysis(call_id, analysis, model, prompt_version)


def save_sentiment(call_id: str, turns: list[dict], summary: dict, model: str) -> None:
    """Replace one call's turn scores and summary."""
    cols = ["turn", "speaker", "role", "words", "weight", "score", "magnitude", "label", "p_negative", "p_neutral", "p_positive"]
    with db.pool().connection() as c, c.transaction():
        c.execute("DELETE FROM phase1.turn_sentiment WHERE call_id = %s", (call_id,))
        with c.cursor() as cur:
            cur.executemany(f"INSERT INTO phase1.turn_sentiment (call_id, {', '.join(cols)}) "
                            f"VALUES (%s, {', '.join(['%s'] * len(cols))})", [[call_id] + [t[k] for k in cols] for t in turns])
        c.execute("INSERT INTO phase1.call_sentiment (call_id, model, summary) VALUES (%s, %s, %s) "
                  "ON CONFLICT (call_id) DO UPDATE SET model = EXCLUDED.model, summary = EXCLUDED.summary, scored_at = now()",
                  (call_id, model, Jsonb(summary)))


def sentiment(call_id: str) -> dict | None:
    """{'model', 'call': summary, 'turns': [{turn, role, score, magnitude, label}]}, or None if never scored."""
    head = db.fetch_all("SELECT model, summary FROM phase1.call_sentiment WHERE call_id = %s", (call_id,))
    if not head:
        return None
    turns = db.fetch_all("SELECT turn, role, score, magnitude, label FROM phase1.turn_sentiment "
                         "WHERE call_id = %s ORDER BY turn", (call_id,))
    return {"model": head[0]["model"], "call": head[0]["summary"], "turns": turns}
