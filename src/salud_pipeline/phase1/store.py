"""The team's shared Phase 1 data in Neon: calls with transcripts, AI analyses, who tags which call, and every
rater's tags.

The dashboard, `analyze` and `compare` all go through this class, so every teammate works on the same
data from their own laptop. Connection: DATABASE_URL (pooled) from .env.

Rules the database code enforces:
- AI analyses are only ever added (insert ... on conflict do nothing), except that Re-analyze on the agent
  dashboard replaces one call's analysis (replace_ai_analysis).
- A rater's tag for a call is one row; saving again replaces that rater's own row only.
- Calls pasted on the /analyze page (source 'new') stay out of tagging, assignments and comparisons:
  calls(), ai_info() and ai_analyses() leave them out unless asked.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pandas as pd
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ..llm_client import load_dotenv
from .checklist import OUTCOMES, SENTIMENTS, TARGETS, TRANSCRIPT_ITEMS

TAG_COLUMNS = ["call_id", "rater"] + TARGETS + ["notes", "llm_model", "overrides", "saved_at"]


def database_url(direct: bool = False) -> str:
    load_dotenv()
    keys = ["DATABASE_URL_UNPOOLED", "DATABASE_URL"] if direct else ["DATABASE_URL", "DATABASE_URL_UNPOOLED"]
    url = next((os.environ[k] for k in keys if os.environ.get(k)), None)
    if not url:
        raise RuntimeError("No database configured: add DATABASE_URL to .env (ask the team lead for it)")
    return url


class Store:
    def __init__(self, url: str | None = None, max_size: int = 4):
        # prepare_threshold=None: no server-side prepared statements, which PgBouncer pooling doesn't keep.
        self.pool = ConnectionPool(url or database_url(), min_size=0, max_size=max_size, open=True,
                                   kwargs={"row_factory": dict_row, "prepare_threshold": None,
                                           "autocommit": True},
                                   check=ConnectionPool.check_connection)

    def close(self) -> None:
        self.pool.close()

    def _all(self, sql: str, params=()) -> list[dict]:
        with self.pool.connection() as c:
            return c.execute(sql, params).fetchall()

    # ---- calls ----
    def calls(self, include_new: bool = False) -> list[dict]:
        where = "" if include_new else "WHERE source <> 'new' "
        return self._all("SELECT call_id, source, file_id FROM phase1.calls " + where +
                         "ORDER BY source = 'synthetic', file_id")

    def analysis_list(self) -> list[dict]:
        """Every call for the /analyze page, newest pasted calls first, with what has been computed for it."""
        return self._all(
            "SELECT c.call_id, c.source, c.file_id, c.added_by, c.added_at, "
            "a.call_id IS NOT NULL AS analyzed, s.call_id IS NOT NULL AS scored, "
            "s.summary->>'caller_end_label' AS end_label "
            "FROM phase1.calls c LEFT JOIN phase1.ai_analyses a USING (call_id) "
            "LEFT JOIN phase1.call_sentiment s USING (call_id) "
            "ORDER BY c.source = 'new' DESC, c.added_at DESC NULLS LAST, c.source = 'synthetic', c.file_id")

    def add_new_call(self, transcript: str, added_by: str) -> str:
        """Save a transcript pasted on /analyze as call 'new/<time>-<id>'. Returns its call_id."""
        if not transcript.strip():
            raise ValueError("the transcript is empty")
        now = datetime.now(timezone.utc)
        file_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        with self.pool.connection() as c:
            c.execute("INSERT INTO phase1.calls (call_id, source, file_id, transcript, added_by, added_at) "
                      "VALUES (%s, 'new', %s, %s, %s, %s)",
                      (f"new/{file_id}", file_id, transcript.strip(), added_by[:100], now))
        return f"new/{file_id}"

    def transcript(self, call_id: str) -> str | None:
        rows = self._all("SELECT transcript FROM phase1.calls WHERE call_id = %s", (call_id,))
        return rows[0]["transcript"] if rows else None

    def planted(self) -> pd.DataFrame:
        rows = self._all("SELECT call_id, planted FROM phase1.calls WHERE planted IS NOT NULL")
        return pd.DataFrame([{"call_id": r["call_id"], **r["planted"]} for r in rows],
                            columns=["call_id"] + TARGETS)

    # ---- AI analyses ----
    def ai_info(self) -> dict:
        r = self._all("SELECT count(*) AS analyzed, min(model) AS model, max(model) AS model_max, "
                      "min(prompt_version) AS prompt_version, max(prompt_version) AS pv_max FROM phase1.ai_analyses "
                      "WHERE NOT starts_with(call_id, 'new/')")[0]
        return {"analyzed": r["analyzed"], "model": r["model"], "prompt_version": r["prompt_version"],
                "mixed": (r["model"] != r["model_max"]) or (r["prompt_version"] != r["pv_max"])}

    def ai_analysis(self, call_id: str) -> dict | None:
        rows = self._all("SELECT raw FROM phase1.ai_analyses WHERE call_id = %s", (call_id,))
        return rows[0]["raw"] if rows else None

    def ai_analyses(self) -> dict[str, dict]:
        return {r["call_id"]: r["raw"] for r in self._all("SELECT call_id, raw FROM phase1.ai_analyses "
                                                          "WHERE NOT starts_with(call_id, 'new/')")}

    def add_ai_analysis(self, call_id: str, analysis: dict, model: str, prompt_version: str) -> bool:
        """Insert one analysis with its checklist and coaching rows. Returns False if one already exists."""
        from .compare import call_quality

        cl = analysis.get("checklist") or {}
        with self.pool.connection() as c, c.transaction():
            inserted = c.execute(
                "INSERT INTO phase1.ai_analyses (call_id, model, prompt_version, analyzed_at, sentiment, outcome, "
                "quality_score, summary, quotes_dropped, raw) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (call_id) DO NOTHING RETURNING call_id",
                (call_id, model, prompt_version, analysis.get("analyzed_at"),
                 (analysis.get("sentiment") or {}).get("label"), (analysis.get("outcome") or {}).get("label"),
                 call_quality({k: (cl.get(k) or {}).get("answer") for k in TRANSCRIPT_ITEMS}),
                 analysis.get("summary"), (analysis.get("evidence_check") or {}).get("dropped"),
                 Jsonb(analysis))).fetchone()
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

    def replace_ai_analysis(self, call_id: str, analysis: dict, model: str, prompt_version: str) -> None:
        """Re-analyze on the agent dashboard: drop this call's stored analysis (with its checklist and coaching
        rows) and add the new one. The only place an analysis is ever changed, and only for this one call."""
        with self.pool.connection() as c, c.transaction():
            for table in ("ai_coaching", "ai_checklist", "ai_analyses"):
                c.execute(f"DELETE FROM phase1.{table} WHERE call_id = %s", (call_id,))
        self.add_ai_analysis(call_id, analysis, model, prompt_version)

    # ---- turn-level sentiment (salud-sentiment) ----
    def save_sentiment(self, call_id: str, turns: list[dict], summary: dict, model: str) -> None:
        """Replace one call's turn scores and summary (a rescore overwrites the old one)."""
        cols = ["turn", "speaker", "role", "words", "weight", "score", "magnitude", "label",
                "p_negative", "p_neutral", "p_positive"]
        with self.pool.connection() as c, c.transaction():
            c.execute("DELETE FROM phase1.turn_sentiment WHERE call_id = %s", (call_id,))
            with c.cursor() as cur:
                cur.executemany(f"INSERT INTO phase1.turn_sentiment (call_id, {', '.join(cols)}) "
                                f"VALUES (%s, {', '.join(['%s'] * len(cols))})",
                                [[call_id] + [t[k] for k in cols] for t in turns])
            c.execute("INSERT INTO phase1.call_sentiment (call_id, model, summary) VALUES (%s, %s, %s) "
                      "ON CONFLICT (call_id) DO UPDATE SET model = EXCLUDED.model, summary = EXCLUDED.summary, "
                      "scored_at = now()", (call_id, model, Jsonb(summary)))

    def sentiment(self, call_id: str) -> dict | None:
        """{'model', 'call': summary, 'turns': [{turn, role, score, magnitude, label}]}, or None if never scored."""
        head = self._all("SELECT model, summary FROM phase1.call_sentiment WHERE call_id = %s", (call_id,))
        if not head:
            return None
        turns = self._all("SELECT turn, role, score, magnitude, label FROM phase1.turn_sentiment "
                          "WHERE call_id = %s ORDER BY turn", (call_id,))
        return {"model": head[0]["model"], "call": head[0]["summary"], "turns": turns}

    def sentiment_summaries(self) -> dict[str, dict]:
        """call_id -> summary, for every scored call."""
        return {r["call_id"]: r["summary"] for r in self._all("SELECT call_id, summary FROM phase1.call_sentiment")}

    # ---- reviewed tags ----
    def tags(self, rater: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM phase1.reviewed_tags" + (" WHERE rater = %s" if rater else "")
        rows = self._all(sql, (rater,) if rater else ())
        df = pd.DataFrame(rows, columns=TAG_COLUMNS)
        df["llm_model"] = df["llm_model"].fillna("")
        return df

    # ---- assignments ----
    def assignments(self) -> dict[str, list[str]]:
        """call_id -> raters assigned to it."""
        out: dict[str, list[str]] = {}
        for r in self._all("SELECT call_id, rater FROM phase1.assignments ORDER BY call_id, rater"):
            out.setdefault(r["call_id"], []).append(r["rater"])
        return out

    def assign(self, pairs: list[tuple[str, str]]) -> int:
        """Add (call_id, rater) assignments; ones that already exist are left alone. Returns how many were added."""
        n = 0
        with self.pool.connection() as c:
            for call_id, rater in pairs:
                n += c.execute("INSERT INTO phase1.assignments (call_id, rater) VALUES (%s, %s) "
                               "ON CONFLICT DO NOTHING RETURNING 1", (call_id, rater)).fetchone() is not None
        return n

    def save_tag(self, rater: str, call_id: str, values: dict, llm: dict | None = None) -> None:
        if not rater.strip():
            raise ValueError("rater is required")
        if not self._all("SELECT 1 FROM phase1.calls WHERE call_id = %s", (call_id,)):
            raise ValueError(f"unknown call {call_id}")
        row = {}
        for t in TRANSCRIPT_ITEMS:
            v = values.get(t, "")
            if v not in ("Yes", "No", "N/A", ""):
                raise ValueError(f"{t}: {v!r}")
            row[t] = v or None
        for t, allowed in (("sentiment", SENTIMENTS), ("outcome", OUTCOMES)):
            v = values.get(t, "")
            if v and v not in allowed:
                raise ValueError(f"{t}: {v!r}")
            row[t] = v or None
        suggested = (llm or {}).get("answers") or {}
        model = str((llm or {}).get("model") or "")[:100] if suggested else None
        overrides = [t for t in TARGETS if suggested.get(t) and row[t] and row[t] != suggested[t]]
        cols = ["call_id", "rater"] + TARGETS + ["notes", "llm_model", "overrides"]
        vals = [call_id, rater] + [row[t] for t in TARGETS] + [str(values.get("notes", ""))[:2000] or None,
                                                               model, overrides]
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols[2:]) + ", saved_at = now()"
        with self.pool.connection() as c:
            c.execute(f"INSERT INTO phase1.reviewed_tags ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
                      f"ON CONFLICT (call_id, rater) DO UPDATE SET {updates}", vals)


def as_label_frame(tags: pd.DataFrame) -> pd.DataFrame:
    """Tags in the shape compare.build expects (overrides as ';'-joined text)."""
    out = tags.copy()
    out["overrides"] = out["overrides"].map(lambda v: ";".join(v) if isinstance(v, list) else (v or ""))
    for t in TARGETS:
        out[t] = out[t].where(out[t].notna(), None)
    return out
