"""Zero-shot QA scoring with a self-hosted LLM behind an OpenAI-compatible endpoint
(team-hosted vLLM/Ollama/llama.cpp; see llm_client.py for configuration).

Responses are cached on disk keyed by (model, prompt version, transcript), so re-running an
evaluation never re-queries the model for a call it has already scored.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from ..llm_client import LLMEndpoint
from .targets import ITEM_DEFINITIONS, OUTCOMES, SENTIMENTS, TRANSCRIPT_ITEMS

PROMPT_VERSION = "v1"

SYSTEM = ("You are a quality-assurance reviewer for a hospital patient billing call center. "
          "You score calls strictly against the checklist, using only evidence in the transcript.")


def build_prompt(transcript: str) -> str:
    items = "\n".join(f'- "{k}": {d}' for k, d in ITEM_DEFINITIONS.items())
    return f"""Score this call transcript.

Checklist items (answer "Yes" or "No" for each; when a transcript is too garbled to judge, answer
"No" only if the required behavior is clearly absent, otherwise "Yes"):
{items}

Also classify:
- "sentiment": the caller's overall sentiment, one of {SENTIMENTS}
- "outcome": how the call ended, one of {OUTCOMES}

Return only a JSON object with this shape:
{{"greeting": {{"answer": "Yes", "evidence": "<short quote>"}}, ... one entry per checklist item ...,
  "sentiment": "neutral", "outcome": "non_committal"}}

Transcript:
\"\"\"
{transcript}
\"\"\""""


class LLMScorer:
    def __init__(self, endpoint: LLMEndpoint, cache_path: Path, temperature: float = 0.0):
        self.endpoint, self.model, self.temperature = endpoint, endpoint.model, temperature
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec["response"]

    def _key(self, transcript: str) -> str:
        return hashlib.sha256(f"{self.model}|{PROMPT_VERSION}|{transcript}".encode()).hexdigest()

    def _call(self, prompt: str) -> str:
        return self.endpoint.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                  temperature=self.temperature, json_mode=True)

    def score_one(self, transcript: str) -> dict:
        key = self._key(transcript)
        if key not in self.cache:
            raw = self._call(build_prompt(transcript))
            self.cache[key] = parse_response(raw)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as fh:
                fh.write(json.dumps({"key": key, "model": self.model, "response": self.cache[key]}) + "\n")
        return self.cache[key]

    def predict(self, df: pd.DataFrame, progress: bool = True) -> pd.DataFrame:
        rows = []
        for i, (cid, text) in enumerate(zip(df["contact_id"], df["text"])):
            r = self.score_one(text)
            rows.append({"contact_id": cid, **{k: r.get(k) for k in TRANSCRIPT_ITEMS},
                         "sentiment": r.get("sentiment"), "outcome": r.get("outcome"),
                         **{f"{k}_evidence": r.get(f"{k}_evidence") for k in TRANSCRIPT_ITEMS}})
            if progress and (i + 1) % 10 == 0:
                print(f"  scored {i + 1}/{len(df)}", flush=True)
        return pd.DataFrame(rows)


def parse_response(raw: str) -> dict:
    """Normalize model output; anything unparseable becomes None (counted as an error, not a guess)."""
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    out: dict = {"_raw_ok": bool(data)}
    for k in TRANSCRIPT_ITEMS:
        v = data.get(k)
        ans = v.get("answer") if isinstance(v, dict) else v
        ans = str(ans).strip().capitalize() if ans is not None else None
        out[k] = ans if ans in ("Yes", "No") else None
        out[f"{k}_evidence"] = v.get("evidence") if isinstance(v, dict) else None
    s = str(data.get("sentiment", "")).strip().lower()
    out["sentiment"] = s if s in SENTIMENTS else None
    o = str(data.get("outcome", "")).strip().lower()
    out["outcome"] = o if o in OUTCOMES else None
    return out
