"""The agent dashboard's Analyze button: fill in whatever a call is missing, then return everything.
Re-analyze (redo=True) recomputes both parts and replaces what was stored.

- Sentiment: scored with the int8 ONNX model in this process (sentiment/model.py) and saved to Neon.
- AI review: one LLM call (prompt.py, same prompt version as the stored analyses) and saved to Neon, only if
  the call has none yet and an LLM is configured (GROQ_API_KEY, or a local server via SALUD_LLM_BASE_URL).
Without redo, existing results are reused, so the 60 Phase 1 calls load instantly and cost nothing.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..llm_client import LLMEndpoint
from ..paths import LLM_CACHE
from .prompt import PROMPT_VERSION, CallAssistant
from .store import Store

MAX_TRANSCRIPT_CHARS = 60_000


def _llm_cache() -> Path:
    return Path(tempfile.gettempdir()) / "salud-llm-cache" if os.environ.get("VERCEL") else LLM_CACHE


def llm_ready() -> bool:
    ep = LLMEndpoint()
    return bool(ep.model and (ep.api_key or ep.is_local))


def analyze(store: Store, call_id: str, redo: bool = False) -> dict:
    """{'sentiment', 'analysis', 'notes'}: each part is the stored result or {'none': True}."""
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
                    notes.append("The AI's reply couldn't be read. Click Analyze to try again.")
                elif had_review:
                    store.replace_ai_analysis(call_id, result, ep.model, PROMPT_VERSION)
                else:
                    store.add_ai_analysis(call_id, result, ep.model, PROMPT_VERSION)
    return {"sentiment": store.sentiment(call_id) or {"none": True},
            "analysis": store.ai_analysis(call_id) or {"none": True}, "notes": notes}
