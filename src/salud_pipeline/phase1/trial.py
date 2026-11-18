"""Try another prompt version beside the one the dashboard uses, and score both against the same tags.

    salud-phase1 trial --version assist-v3 # synthetic calls only (default)
    salud-phase1 trial --source all        # real calls too (they are sent to the LLM provider)
    salud-phase1 trial --limit 20          # at most 20 new LLM requests this run

Each new analysis is added to phase1.ai_analysis_versions (never to phase1.ai_analyses, so the dashboard and
existing tags are untouched) and to reports/phase1/ai_versions/<prompt version>.json. Calls already stored
for this model and prompt version are skipped, so a run can be repeated after Groq's daily cap.

Tags were made by correcting the dashboard version's pre-filled answers, so they lean towards that version.
Read a trial's agreement as "how often it lands on the reviewers' final answer", not as an unbiased score.
"""

from __future__ import annotations

import json
from pathlib import Path

from .prompt import CallAssistant
from .store import Store


def run(store: Store, assistant: CallAssistant, out_json: Path, source: str = "synthetic",
        limit: int | None = None, progress: bool = True) -> dict:
    model, version = assistant.endpoint.model, assistant.version
    calls = [c for c in store.calls() if source == "all" or c["source"] == source]
    done = store.ai_versions(model, version)
    counts = {"calls": len(calls), "already_stored": 0, "from_cache": 0, "new_llm_requests": 0,
              "failed": 0, "not_done": 0, "stopped_early": None}
    for i, c in enumerate(calls, 1):
        call_id = c["call_id"]
        if call_id in done:
            counts["already_stored"] += 1
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
        store.add_ai_version(call_id, analysis, model, version)   # keeps an existing row if one appeared meanwhile
        if progress:
            print(f"  [{i}/{len(calls)}] {call_id}", flush=True)

    # Same shape as ai_analyses.json, so a version can be imported later. Re-read so the file matches the database.
    done = store.ai_versions(model, version)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"model": model, "prompt_version": version,
                                    "calls": {cid: {"analysis": a} for cid, a in sorted(done.items())}}, indent=1))
    counts["stored_total"] = len(done)
    return counts
