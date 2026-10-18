"""Batch QA scoring with the same one-prompt review the review page uses (qa/assist.py).

Batch results and the page share one prompt and one cache (reports/qa/assist_cache), so a call
scored in either place is never sent to the LLM twice and both show identical answers.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..llm_client import LLMEndpoint
from .assist import CallAssistant
from .targets import TRANSCRIPT_ITEMS


class LLMScorer:
    def __init__(self, endpoint: LLMEndpoint, cache_dir: Path):
        self.assistant = CallAssistant(endpoint, cache_dir)
        self.model = endpoint.model

    def predict(self, df: pd.DataFrame, progress: bool = True) -> pd.DataFrame:
        """One row per call: predicted targets plus evidence counts. Failed calls keep None predictions."""
        rows, t0, done_new = [], time.time(), 0
        for i, (cid, text) in enumerate(zip(df["contact_id"], df["text"])):
            cached = self.assistant.cached(text) is not None
            try:
                r = self.assistant.review(text)
                err = None if r.get("ok", True) else "unparseable response"
            except (RuntimeError, ValueError) as e:
                r, err = {}, str(e)
            cl = r.get("checklist", {})
            row = {"contact_id": cid, "error": err}
            for k in TRANSCRIPT_ITEMS:
                row[k] = (cl.get(k) or {}).get("answer")
                row[f"{k}_n_evidence"] = len((cl.get(k) or {}).get("evidence", []))
            row["sentiment"] = (r.get("sentiment") or {}).get("label")
            row["outcome"] = (r.get("outcome") or {}).get("label")
            row["quotes_dropped"] = (r.get("evidence_check") or {}).get("dropped")
            row["n_coaching_tips"] = len(r.get("coaching", []))
            rows.append(row)
            if not cached:
                done_new += 1
            if progress and ((i + 1) % 5 == 0 or i + 1 == len(df)):
                rate = (time.time() - t0) / max(done_new, 1)
                left = sum(1 for t in df["text"].iloc[i + 1:] if self.assistant.cached(t) is None)
                print(f"  scored {i + 1}/{len(df)}  (new LLM calls: {done_new}, ~{left * rate / 60:.0f} min left)",
                      flush=True)
        return pd.DataFrame(rows)
