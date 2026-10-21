"""The team's shared Phase 1 data in Neon: calls with transcripts, AI analyses, who tags which call, and every
rater's tags.

The dashboard, `analyze` and `compare` all go through this class, so every teammate works on the same
data from their own laptop. Connection: DATABASE_URL (pooled) from .env.

Rules the database code enforces:
- AI analyses are only ever added, never changed (insert ... on conflict do nothing).
- A rater's tag for a call is one row; saving again replaces that rater's own row only.
"""

from __future__ import annotations

import os

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
    def calls(self) -> list[dict]:
        return self._all("SELECT call_id, source, file_id FROM phase1.calls "
                         "ORDER BY source = 'synthetic', file_id")

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
                      "min(prompt_version) AS prompt_version, max(prompt_version) AS pv_max FROM phase1.ai_analyses")[0]
        return {"analyzed": r["analyzed"], "model": r["model"], "prompt_version": r["prompt_version"],
                "mixed": (r["model"] != r["model_max"]) or (r["prompt_version"] != r["pv_max"])}

    def ai_analysis(self, call_id: str) -> dict | None:
        rows = self._all("SELECT raw FROM phase1.ai_analyses WHERE call_id = %s", (call_id,))
        return rows[0]["raw"] if rows else None

    def ai_analyses(self) -> dict[str, dict]:
        return {r["call_id"]: r["raw"] for r in self._all("SELECT call_id, raw FROM phase1.ai_analyses")}

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
