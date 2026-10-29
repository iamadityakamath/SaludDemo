"""Analysis of stored transcripts: fill in whatever a call is missing, then return everything.
Re-analyze (redo=True) recomputes both parts and replaces what was stored.

- Sentiment: scored with the int8 ONNX model in this process (sentiment/model.py) and saved to Neon.
- AI review: one LLM call (prompt.py, same prompt version as the stored analyses) and saved to Neon, only if
  the call has none yet and an LLM is configured (GROQ_API_KEY, or a local server via SALUD_LLM_BASE_URL).
Without redo, existing results are reused, so analyzed calls load instantly and cost nothing.

Uploading a call saves it and starts its analysis right away; the list shows it as processing until both parts
are stored (call_status). process_pending() catches calls added any other way: a background thread in the
laptop dashboard, a daily Vercel cron on the website (/api/cron/process), and `salud-phase1 process`.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..llm_client import LLMEndpoint
from ..paths import LLM_CACHE
from .prompt import PROMPT_VERSION, CallAssistant
from .store import Store

MAX_TRANSCRIPT_CHARS = 60_000
MAX_UPLOAD_CALLS = 20
PROCESSING_MINUTES = 10   # a call added this recently and still missing results is being analyzed


def _llm_cache() -> Path:
    return Path(tempfile.gettempdir()) / "salud-llm-cache" if os.environ.get("VERCEL") else LLM_CACHE


def llm_ready() -> bool:
    ep = LLMEndpoint()
    return bool(ep.model and (ep.api_key or ep.is_local))


_call_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(call_id: str) -> threading.Lock:
    with _locks_guard:
        return _call_locks.setdefault(call_id, threading.Lock())


def call_status(call: dict, review: bool, now: datetime | None = None) -> str:
    """'done' (everything this server can compute is stored), 'processing' (added in the last few minutes and
    still missing a part) or 'incomplete' (missing a part; opening the call or the next sweep fills it in)."""
    if call["scored"] and (call["analyzed"] or not review):
        return "done"
    added = call.get("added_at")
    if added and (now or datetime.now(timezone.utc)) - added < timedelta(minutes=PROCESSING_MINUTES):
        return "processing"
    return "incomplete"


def results(store: Store, call_id: str) -> dict:
    """What is stored for a call, without computing anything. Same shape as analyze()."""
    if store.transcript(call_id) is None:
        raise ValueError("unknown call")
    return {"sentiment": store.sentiment(call_id) or {"none": True},
            "analysis": store.ai_analysis(call_id) or {"none": True}, "notes": []}


def process_pending(store: Store, limit: int | None = None, seconds: float | None = None, log=None) -> dict:
    """Analyze stored calls that are missing sentiment or an AI review, newest pasted first.
    Stops at `limit` calls, after `seconds`, or at the first failed AI review (e.g. the provider's daily cap)."""
    start, review = time.monotonic(), llm_ready()
    pending = store.pending_calls(need_review=review)
    done, failed = [], None
    for cid in pending[:limit]:
        if seconds is not None and time.monotonic() - start > seconds:
            break
        r = analyze(store, cid)
        done.append(cid)
        if log:
            log(cid, r)
        if review and r["analysis"].get("none"):
            failed = (r["notes"] or ["the AI review failed"])[0]
            break
    return {"pending": len(pending), "done": len(done), "left": len(pending) - len(done), "stopped": failed}


def analyze(store: Store, call_id: str, redo: bool = False) -> dict:
    """{'sentiment', 'analysis', 'notes'}: each part is the stored result or {'none': True}."""
    with _lock(call_id):   # the background sweep and a click on the same call never score it twice at once
        return _analyze(store, call_id, redo)


def _analyze(store: Store, call_id: str, redo: bool) -> dict:
    text = store.transcript(call_id)
    if text is None:
        raise ValueError("unknown call")
    notes = []
    if redo or store.sentiment(call_id) is None:
        from ..sentiment.model import MODEL_NAME, shared_model
        from ..sentiment.results import score_and_save

        score_and_save(store, call_id, text, shared_model(), MODEL_NAME)   # replaces any stored scores
    had_review = store.ai_analysis(call_id) is not None
    if redo or not had_review:
        if not llm_ready():
            notes.append("The AI review isn't set up on this site (no LLM API key), so "
                         + ("the stored review was kept." if had_review else "only sentiment is shown."))
        else:
            ep = LLMEndpoint()
            try:
                if redo:   # an empty cache folder, so the LLM is really asked again instead of replaying its last reply
                    with tempfile.TemporaryDirectory(prefix="salud-redo-") as tmp:
                        result = CallAssistant(ep, Path(tmp)).review(text)
                else:
                    result = CallAssistant(ep, _llm_cache()).review(text)
            except (RuntimeError, ValueError) as e:   # provider error, e.g. the daily request cap
                notes.append(f"The AI review failed: {str(e)[:160]}"
                             + (" The stored review was kept." if had_review else ""))
            else:
                if not result.get("ok"):
                    notes.append("The AI's reply couldn't be read. Click Re-analyze to try again.")
                elif had_review:
                    store.replace_ai_analysis(call_id, result, ep.model, PROMPT_VERSION)
                else:
                    store.add_ai_analysis(call_id, result, ep.model, PROMPT_VERSION)
    return {"sentiment": store.sentiment(call_id) or {"none": True},
            "analysis": store.ai_analysis(call_id) or {"none": True}, "notes": notes}
