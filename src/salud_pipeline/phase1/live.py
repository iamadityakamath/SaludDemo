"""The /analyze page's Analyze button: fill in whatever a call is missing, then return everything.

- Sentiment: scored with the int8 ONNX model in this process (sentiment/model.py) and saved to Neon.
- AI review: one LLM call (prompt.py, same prompt version as the stored analyses) and saved to Neon, only if
  the call has none yet and an LLM is configured (GROQ_API_KEY, or a local server via SALUD_LLM_BASE_URL).
Existing results are never redone, so the 60 Phase 1 calls load instantly and cost nothing.
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


def analyze(store: Store, call_id: str) -> dict:
    """{'sentiment', 'analysis', 'notes'}: each part is the stored result or {'none': True}."""
    text = store.transcript(call_id)
    if text is None:
        raise ValueError("unknown call")
    notes = []
    if store.sentiment(call_id) is None:
        from ..sentiment.model import MODEL_NAME, shared_model
        from ..sentiment.results import score_and_save

        score_and_save(store, call_id, text, shared_model(), MODEL_NAME)
    if store.ai_analysis(call_id) is None:
        if not llm_ready():
            notes.append("The AI review isn't set up on this site (no LLM API key), so only sentiment is shown.")
        else:
            ep = LLMEndpoint()
            try:
                result = CallAssistant(ep, _llm_cache()).review(text)
            except (RuntimeError, ValueError) as e:   # provider error, e.g. the daily request cap
                notes.append(f"The AI review failed: {str(e)[:160]}")
            else:
                if result.get("ok"):
                    store.add_ai_analysis(call_id, result, ep.model, PROMPT_VERSION)
                else:
                    notes.append("The AI's reply couldn't be read. Click Analyze to try again.")
    return {"sentiment": store.sentiment(call_id) or {"none": True},
            "analysis": store.ai_analysis(call_id) or {"none": True}, "notes": notes}
