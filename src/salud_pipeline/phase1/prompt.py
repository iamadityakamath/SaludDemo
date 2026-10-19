"""One-prompt LLM review of a call: checklist answers with cited evidence, sentiment, outcome,
a one-line summary and agent coaching tips.

The transcript is sent as numbered turns ([T1], [T2], ...). The model must cite evidence as
(turn, exact quote). Every citation is checked against the transcript; quotes that do not
appear in the cited turn are dropped and counted, so the page never highlights invented text.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from ..llm_client import LLMEndpoint
from .checklist import ITEM_DEFINITIONS, OUTCOME_DEFINITIONS, OUTCOME_PRECEDENCE, OUTCOMES, SENTIMENTS, TRANSCRIPT_ITEMS

PROMPT_VERSION = "assist-v2"  # v2: precise outcome definitions

SYSTEM = (
    "You are a senior quality-assurance reviewer for a hospital patient billing call center that "
    "practices ethical, patient-first collections. You score calls strictly against the checklist "
    "using only evidence in the transcript, and you coach agents toward empathy, accuracy and privacy "
    "compliance, never toward pressure tactics."
)

_TURN_RE = re.compile(r"^([^:\n]{1,45}):\s*([\s\S]*)$")


def split_turns(text: str) -> list[dict]:
    """Paragraph blocks -> [{'n': 1, 'speaker': 'Agent', 'text': '...'}]. Unlabeled blocks are 'Note'."""
    turns = []
    for block in [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]:
        m = _TURN_RE.match(block)
        speaker, body = (m.group(1).strip(), m.group(2).strip()) if m else ("Note", block)
        turns.append({"n": len(turns) + 1, "speaker": speaker, "text": body})
    return turns


def build_prompt(turns: list[dict]) -> str:
    items = "\n".join(f'  "{k}": {d}' for k, d in ITEM_DEFINITIONS.items())
    outcomes = "\n".join(f'  "{k}": {OUTCOME_DEFINITIONS[k]}' for k in OUTCOME_PRECEDENCE)
    numbered = "\n".join(f"[T{t['n']}] {t['speaker']}: {t['text']}" for t in turns)
    return f"""Review this call and return ONE JSON object.

CHECKLIST (answer each "Yes", "No" or "N/A"; use "N/A" only when the item cannot apply to this call,
e.g. no balance was discussed, or the caller is an insurance representative):
{items}

For every checklist item give:
- "answer"
- "reason": one short sentence
- "evidence": the turns that justify the answer, as a list of {{"turn": <number>, "quote": "<exact words copied from that turn>"}}.
  For "Yes", quote what the agent said or did. For "No", quote the turn where the behavior was missing or
  went wrong (e.g. the balance disclosed before verification). Quotes must be copied exactly, 3-25 words.

Also return:
- "sentiment": {{"label": one of {SENTIMENTS}, "evidence": [...same format, caller turns...]}}
- "outcome":   {{"label": <one outcome key below>, "evidence": [...turns showing how the call ended...]}}
- "summary": one sentence describing the call
- "coaching": 1-4 tips for the agent, each {{"item": <checklist key or "general">, "tip": "<specific, actionable
  advice in second person>", "turn": <turn number it refers to>}}. Base tips on items answered "No" first.
  If the call was excellent, give one tip that reinforces what went well. Never recommend pressuring the caller.

OUTCOME (how the call ended; if more than one fits, choose the one listed first):
{outcomes}

Shape:
{{"checklist": {{"greeting": {{"answer": "Yes", "reason": "...", "evidence": [{{"turn": 2, "quote": "..."}}]}}, ...all 7 items...}},
 "sentiment": {{...}}, "outcome": {{...}}, "summary": "...", "coaching": [...]}}

TRANSCRIPT
{numbered}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9$*\[\]]+", " ", s.lower()).strip()


def _check_evidence(raw, turns: list[dict], stats: dict) -> list[dict]:
    out = []
    by_n = {t["n"]: t for t in turns}
    for ev in raw if isinstance(raw, list) else []:
        if not isinstance(ev, dict):
            continue
        try:
            n = int(str(ev.get("turn", "")).lstrip("Tt"))
        except ValueError:
            stats["dropped"] += 1
            continue
        quote = str(ev.get("quote", "")).strip().strip('"')
        turn = by_n.get(n)
        if turn is None or not quote:
            stats["dropped"] += 1
            continue
        if _norm(quote) not in _norm(turn["text"]):
            # Accept the quote if it appears verbatim in another turn (wrong turn number).
            alt = next((t for t in turns if _norm(quote) and _norm(quote) in _norm(t["text"])), None)
            if alt is None:
                stats["dropped"] += 1
                continue
            n = alt["n"]
        stats["kept"] += 1
        out.append({"turn": n, "quote": quote})
    return out


def parse(raw: str, turns: list[dict]) -> dict:
    """Validate the model's JSON into the shape the tagging page expects."""
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    stats = {"kept": 0, "dropped": 0}
    checklist = {}
    src = data.get("checklist", {}) if isinstance(data.get("checklist"), dict) else {}
    for k in TRANSCRIPT_ITEMS:
        v = src.get(k) if isinstance(src.get(k), dict) else {}
        ans = str(v.get("answer", "")).strip()
        ans = {"yes": "Yes", "no": "No", "n/a": "N/A", "na": "N/A"}.get(ans.lower())
        checklist[k] = {"answer": ans, "reason": str(v.get("reason", ""))[:400],
                        "evidence": _check_evidence(v.get("evidence"), turns, stats)}

    def labelled(key, allowed):
        v = data.get(key) if isinstance(data.get(key), dict) else {"label": data.get(key)}
        label = str(v.get("label", "")).strip().lower()
        return {"label": label if label in allowed else None,
                "evidence": _check_evidence(v.get("evidence"), turns, stats)}

    coaching = []
    for c in data.get("coaching", []) if isinstance(data.get("coaching"), list) else []:
        if not isinstance(c, dict) or not str(c.get("tip", "")).strip():
            continue
        item = str(c.get("item", "general"))
        try:
            turn = int(str(c.get("turn", "")).lstrip("Tt"))
        except ValueError:
            turn = None
        coaching.append({"item": item if item in TRANSCRIPT_ITEMS else "general",
                         "tip": str(c["tip"]).strip()[:600],
                         "turn": turn if turn and 1 <= turn <= len(turns) else None})
    return {
        "ok": bool(data),
        "checklist": checklist,
        "sentiment": labelled("sentiment", SENTIMENTS),
        "outcome": labelled("outcome", OUTCOMES),
        "summary": str(data.get("summary", ""))[:400],
        "coaching": coaching[:4],
        "evidence_check": stats,
    }


class CallAssistant:
    """Runs the one-prompt review with an on-disk cache keyed by (model, prompt version, transcript)."""

    def __init__(self, endpoint: LLMEndpoint, cache_dir: Path):
        self.endpoint = endpoint
        self.cache_dir = cache_dir

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.endpoint.model}|{PROMPT_VERSION}|{text}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{key}.json"

    def cached(self, text: str) -> dict | None:
        """A previous review of this transcript, without calling the LLM."""
        path = self._path(text)
        return json.loads(path.read_text()) if path.exists() else None

    def review(self, text: str, force: bool = False) -> dict:
        """Cached review if there is one (unless force), otherwise one LLM call."""
        path = self._path(text)
        if not force and path.exists():
            return json.loads(path.read_text())
        turns = split_turns(text)
        raw = self.endpoint.chat([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": build_prompt(turns)}],
                                 temperature=0.0, json_mode=True, max_tokens=3500)
        result = parse(raw, turns) | {"model": self.endpoint.model, "provider": self.endpoint.provider,
                                      "analyzed_at": datetime.now().isoformat(timespec="seconds")}
        if result["ok"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
        return result
