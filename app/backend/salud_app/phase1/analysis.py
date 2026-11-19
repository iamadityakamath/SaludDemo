"""Analysis of stored transcripts: fill in whatever a call is missing, then return everything. Re-analyze (redo=True)
recomputes both parts and replaces what was stored.

- Sentiment: scored with the local Hugging Face model (sentiment/) and saved.
- AI review: one Groq call (prompt.py, assist-v2) and saved, only if the call has none yet and an LLM is configured.
Without redo, stored results are reused, so analyzed calls load instantly and cost nothing.

Uploads are analyzed in a background thread right away; a sweep (every 15 s while the app runs) catches calls added
any other way. The list shows a call as processing until both parts are stored (call_status).
"""

from __future__ import annotations

import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import get_settings
from ..sentiment.model import MODEL_NAME, shared_model
from ..sentiment.scoring import score_call
from . import store
from .llm import LLMEndpoint
from .prompt import PROMPT_VERSION, CallAssistant

MAX_TRANSCRIPT_CHARS = 60_000
MAX_UPLOAD_CALLS = 20
PROCESSING_MINUTES = 10   # a call added this recently and still missing results is being analyzed

_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def _lock(call_id: str) -> threading.Lock:
    with _guard:
        return _locks.setdefault(call_id, threading.Lock())


def llm_ready() -> bool:
    return LLMEndpoint().ready


def call_status(call: dict, review: bool, now: datetime | None = None) -> str:
    """'done' (everything this server can compute is stored), 'processing' (added in the last few minutes and still
    missing a part) or 'incomplete' (missing a part; opening the call or the next sweep fills it in)."""
    if call["scored"] and (call["analyzed"] or not review):
        return "done"
    added = call.get("added_at")
    if added and (now or datetime.now(timezone.utc)) - added < timedelta(minutes=PROCESSING_MINUTES):
        return "processing"
    return "incomplete"


def results(call_id: str) -> dict:
    """What is stored for a call, without computing anything."""
    if store.transcript(call_id) is None:
        raise LookupError("unknown call")
    return {"sentiment": store.sentiment(call_id) or {"none": True},
            "analysis": store.ai_analysis(call_id) or {"none": True}, "notes": []}


def analyze(call_id: str, redo: bool = False) -> dict:
    """{'sentiment', 'analysis', 'notes'}: each part is the stored result or {'none': True}."""
    with _lock(call_id):   # the background sweep and a click on the same call never score it twice at once
        return _analyze(call_id, redo)


def _analyze(call_id: str, redo: bool) -> dict:
    text = store.transcript(call_id)
    if text is None:
        raise LookupError("unknown call")
    notes = []
    if redo or store.sentiment(call_id) is None:
        rows, summary = score_call(text, shared_model())
        store.save_sentiment(call_id, rows, summary, MODEL_NAME)
    had_review = store.ai_analysis(call_id) is not None
    if redo or not had_review:
        ep = LLMEndpoint()
        if not ep.ready:
            notes.append("The AI review isn't set up on this site (no LLM API key), so "
                         + ("the stored review was kept." if had_review else "only sentiment is shown."))
        else:
            try:
                if redo:   # an empty cache folder, so the LLM is really asked again instead of replaying its last reply
                    with tempfile.TemporaryDirectory(prefix="salud-redo-") as tmp:
                        result = CallAssistant(ep, Path(tmp)).review(text)
                else:
                    result = CallAssistant(ep, get_settings().llm_cache).review(text)
            except (RuntimeError, ValueError) as e:   # provider error, e.g. the daily request cap
                notes.append(f"The AI review failed: {str(e)[:160]}" + (" The stored review was kept." if had_review else ""))
            else:
                if not result.get("ok"):
                    notes.append("The AI's reply couldn't be read. Click Re-analyze to try again.")
                elif had_review:
                    store.replace_ai_analysis(call_id, result, ep.model, PROMPT_VERSION)
                else:
                    store.add_ai_analysis(call_id, result, ep.model, PROMPT_VERSION)
    return {"sentiment": store.sentiment(call_id) or {"none": True},
            "analysis": store.ai_analysis(call_id) or {"none": True}, "notes": notes}


def process_pending(limit: int | None = None, seconds: float | None = None, log=None) -> dict:
    """Analyze stored calls missing sentiment or an AI review, newest uploads first. Stops at `limit` calls, after
    `seconds`, or at the first failed AI review (e.g. the provider's daily cap)."""
    start, review = time.monotonic(), llm_ready()
    pending = store.pending_calls(need_review=review)
    done, failed = [], None
    for cid in pending[:limit]:
        if seconds is not None and time.monotonic() - start > seconds:
            break
        r = analyze(cid)
        done.append(cid)
        if log:
            log(cid, r)
        if review and r["analysis"].get("none"):
            failed = (r["notes"] or ["the AI review failed"])[0]
            break
    return {"pending": len(pending), "done": len(done), "left": len(pending) - len(done), "stopped": failed}


def analyze_in_background(ids: list[str]) -> None:
    """Uploaded calls: analyze them one at a time in a thread (the page's own request then just waits)."""
    def run():
        for cid in ids:
            try:
                analyze(cid)
            except (RuntimeError, ValueError, LookupError):   # the sweep or the page retries it
                pass
    threading.Thread(target=run, name="salud-upload", daemon=True).start()


def start_sweeper(every: float = 15, log=print) -> threading.Event:
    """Analyze new transcripts in the database now and then every `every` seconds. Set the returned event to stop."""
    stop = threading.Event()

    def run():
        while not stop.is_set():
            try:
                r = process_pending(log=lambda cid, _: log(f"  analyzed {cid}"))
                if r["stopped"]:
                    log(f"  automatic analysis paused: {r['stopped'][:160]} ({r['left']} calls left)")
            except Exception as e:  # noqa: BLE001 - database unreachable etc.: try again next round
                log(f"  automatic analysis skipped: {str(e)[:160]}")
            stop.wait(every)

    threading.Thread(target=run, name="salud-sweeper", daemon=True).start()
    return stop
