"""Phase 1 route 2, step 1: run the LLM on every Phase 1 call and add the results to the team database.

    salud-phase1 analyze             # every call without an AI analysis yet
    salud-phase1 analyze --limit 20  # at most 20 new LLM requests this run (Groq free tier: ~60 a day)

Reads each call's transcript from Neon and adds its analysis to phase1.ai_analyses (+ ai_checklist and
ai_coaching): checklist answers with reasons and verified transcript quotes, caller sentiment, call outcome,
a summary and coaching tips (see prompt.py). Existing analyses are never changed, and a run refuses to start
if the model or prompt version differs from the analyses already stored, so results are never mixed.
Calls analyzed before on this laptop come from the local LLM cache at no cost.
"""

from __future__ import annotations

from .prompt import PROMPT_VERSION, CallAssistant
from .store import Store


def run(store: Store, assistant: CallAssistant, limit: int | None = None, progress: bool = True) -> dict:
    """Analyze every call that has no analysis. Stops early on a provider error (e.g. Groq's daily cap)."""
    model = assistant.endpoint.model
    info = store.ai_info()
    if info["analyzed"] and (info["mixed"] or (info["model"], info["prompt_version"]) != (model, PROMPT_VERSION)):
        raise RuntimeError(
            f"The database holds {info['analyzed']} analyses made with {info['model']} / {info['prompt_version']}, "
            f"but the current settings are {model} / {PROMPT_VERSION}. Nothing was changed.")

    done = set(store.ai_analyses())
    calls = store.calls()
    counts = {"calls": len(calls), "already_analyzed": 0, "from_cache": 0, "new_llm_requests": 0,
              "failed": 0, "not_done": 0, "stopped_early": None}
    for i, c in enumerate(calls, 1):
        call_id = c["call_id"]
        if call_id in done:
            counts["already_analyzed"] += 1
            continue
        text = store.transcript(call_id)
        analysis = assistant.cached(text)
        if analysis is not None:
            counts["from_cache"] += 1
        elif counts["stopped_early"] or (limit is not None and counts["new_llm_requests"] >= limit):
            counts["not_done"] += 1
            continue
        else:
            try:
                analysis = assistant.review(text)
                counts["new_llm_requests"] += 1
            except (RuntimeError, ValueError) as e:
                counts["failed"] += 1
                counts["stopped_early"] = str(e)[:200]
                counts["not_done"] += 1
                if progress:
                    print(f"  stopping: {counts['stopped_early']}")
                continue
            if not analysis.get("ok"):
                counts["failed"] += 1
                if progress:
                    print(f"  {call_id}: the model's reply could not be parsed; will retry next run")
                continue
        store.add_ai_analysis(call_id, analysis, model, PROMPT_VERSION)
        if progress:
            print(f"  [{i}/{len(calls)}] {call_id}", flush=True)
    counts["analyzed_total"] = store.ai_info()["analyzed"]
    return counts
