"""Phase 1 route 2, step 1: run the LLM on every Phase 1 call and save the results in one JSON file.

    salud-phase1 analyze             # all 60 calls; only adds calls missing from the file
    salud-phase1 analyze --limit 20  # at most 20 new LLM requests this run (Groq free tier: ~60 a day)

Output: reports/phase1/ai_analyses.json

    {"model": ..., "prompt_version": ..., "updated_at": ...,
     "calls": {"real/<file>.txt": {"source": "real", "file_id": "<file>.txt", "analysis": {...}}, ...}}

Each analysis holds the checklist answers with reasons and verified transcript quotes, caller sentiment,
call outcome, a summary and coaching tips (see prompt.py). Existing entries are never changed: a run only
adds calls that are missing, and it refuses to run if the model or prompt version no longer matches the
file. The dashboard reads this file but never writes to it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .calls import CallSource
from .checklist import clean_transcript
from .prompt import PROMPT_VERSION, CallAssistant


def load(path: Path) -> dict:
    if not path.exists():
        return {"calls": {}}
    return json.loads(path.read_text())


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def run(sources: list[CallSource], assistant: CallAssistant, out: Path, limit: int | None = None,
        progress: bool = True) -> dict:
    """Analyze every call not yet in `out`. Stops early on a provider error (e.g. Groq's daily cap)."""
    data = load(out)
    model = assistant.endpoint.model
    if data.get("calls") and (data.get("model"), data.get("prompt_version")) != (model, PROMPT_VERSION):
        # Never overwrite or mix analyses: the existing file is the AI side of the comparison.
        raise RuntimeError(
            f"{out.name} holds {len(data['calls'])} analyses made with {data.get('model')} / "
            f"{data.get('prompt_version')}, but the current settings are {model} / {PROMPT_VERSION}. "
            f"Nothing was changed. To analyze with the new settings, move {out.name} aside first.")
    data.update({"model": model, "prompt_version": PROMPT_VERSION})

    calls = [(s, p) for s in sources for p in s.files()]
    counts = {"calls": len(calls), "already_in_file": 0, "from_cache": 0, "new_llm_requests": 0,
              "failed": 0, "not_done": 0, "stopped_early": None}
    for i, (src, path) in enumerate(calls, 1):
        call_id = f"{src.key}/{path.name}"
        if call_id in data["calls"]:
            counts["already_in_file"] += 1
            continue
        text = clean_transcript(path.read_text())
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
        data["calls"][call_id] = {"source": src.key, "file_id": path.name, "analysis": analysis}
        _save(data, out)
        if progress:
            print(f"  [{i}/{len(calls)}] {call_id}", flush=True)
    _save(data, out)
    counts["in_file"] = len(data["calls"])
    return counts
