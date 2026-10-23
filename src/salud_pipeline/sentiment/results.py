"""Score calls into two tables, save them locally, and read them back for the dashboard.

    reports/sentiment/turns.csv   one row per scored turn: call_id, turn, speaker, role, words, weight,
                                  score, magnitude, label, p_negative, p_neutral, p_positive
    reports/sentiment/calls.csv   one row per call: caller_* and agent_* summaries (aggregate.summarize)
    reports/sentiment/meta.json   model, revision, when it ran

Both CSVs hold numbers and labels only, never transcript text.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..paths import SENTIMENT_REPORTS
from ..phase1.prompt import split_turns
from .aggregate import summarize
from .turns import scorable_turns

ROLES = ("caller", "agent")
TURN_COLUMNS = ["call_id", "turn", "speaker", "role", "words", "weight", "score", "magnitude", "label",
                "p_negative", "p_neutral", "p_positive"]


def score_calls(transcripts: dict[str, str], model) -> tuple[pd.DataFrame, pd.DataFrame]:
    """transcripts: call_id -> text. model: anything with .score(list[str]) -> list[dict] (model.py)."""
    per_call = {cid: scorable_turns(text) for cid, text in transcripts.items()}
    flat = [(cid, t) for cid, ts in per_call.items() for t in ts]
    scores = model.score([t["text"] for _, t in flat]) if flat else []
    rows = [{"call_id": cid, "turn": t["n"], "speaker": t["speaker"], "role": t["role"], "words": t["words"],
             "weight": t["weight"], **s} for (cid, t), s in zip(flat, scores)]
    turns = pd.DataFrame(rows, columns=TURN_COLUMNS)
    calls = []
    for cid, text in transcripts.items():
        mine = [r for r in rows if r["call_id"] == cid]
        total = len(split_turns(text))
        row = {"call_id": cid, "source": cid.split("/", 1)[0], "call_turns": total}
        for role in ROLES:
            s = summarize([dict(r, n=r["turn"]) for r in mine if r["role"] == role], total)
            row |= {f"{role}_{k}": v for k, v in s.items()}
        calls.append(row)
    return turns, pd.DataFrame(calls)


def save(turns: pd.DataFrame, calls: pd.DataFrame, meta: dict, out: Path = SENTIMENT_REPORTS) -> None:
    out.mkdir(parents=True, exist_ok=True)
    turns.round(4).to_csv(out / "turns.csv", index=False)
    calls.to_csv(out / "calls.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta | {"ran_at": datetime.now(timezone.utc).isoformat()}, indent=2))


class Results:
    """The saved results, reloaded whenever `salud-sentiment run` rewrites them. Empty if never run."""

    def __init__(self, folder: Path = SENTIMENT_REPORTS):
        self.folder = folder
        self._stamp = None
        self._lock = threading.Lock()
        self._turns: dict[str, list[dict]] = {}
        self._calls: dict[str, dict] = {}
        self.meta: dict = {}

    def _refresh(self) -> None:
        files = [self.folder / n for n in ("turns.csv", "calls.csv")]
        stamp = tuple(os.path.getmtime(f) if f.exists() else None for f in files)
        if stamp == self._stamp:
            return
        self._stamp = stamp
        self._turns, self._calls, self.meta = {}, {}, {}
        if None in stamp:
            return
        turns, calls = pd.read_csv(files[0]), pd.read_csv(files[1])
        for r in turns.to_dict("records"):
            self._turns.setdefault(r["call_id"], []).append(
                {k: r[k] for k in ("turn", "role", "score", "magnitude", "label")})
        self._calls = {r["call_id"]: {k: (None if pd.isna(v) else v) for k, v in r.items()}
                       for r in calls.to_dict("records")}
        meta = self.folder / "meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}

    def for_call(self, call_id: str) -> dict | None:
        with self._lock:
            self._refresh()
            if call_id not in self._calls:
                return None
            return {"model": self.meta.get("model"), "call": self._calls[call_id], "turns": self._turns.get(call_id, [])}
