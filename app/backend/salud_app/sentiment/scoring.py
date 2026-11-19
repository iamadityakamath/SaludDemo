"""Which turns are scored, how much each counts, and how turn scores roll up into one call's numbers.

Roles: 'agent' (Agent, Agent (inferred)), 'skip' (notes, automated systems, recordings) or 'caller' (everyone else
on the line). Call numbers per role: overall, start and end (first / last 30% of that role's turns), shift, the
lowest and highest turn. Labels follow CCAI: |score| >= 0.25 is positive or negative; nearer zero it is 'mixed' when
the turns carried strong feeling (magnitude >= 0.5), else 'neutral'.
"""

from __future__ import annotations

import math
import re

from ..phase1.prompt import split_turns

_SKIP = re.compile(r"note|automated|system|\bivr\b|recording|voicemail|music", re.I)
_AGENT = re.compile(r"^\s*agent\b", re.I)
FILLER_WORDS = 3        # turns shorter than this ("ok", "mm-hm") barely count
FILLER_WEIGHT = 0.2
MAX_WEIGHT_WORDS = 40   # longer turns don't count more than a 40-word turn
WINDOW = 0.30
POS, NEG = 0.25, -0.25
MIXED_MAGNITUDE = 0.5
ROLES = ("caller", "agent")


def role(speaker: str) -> str:
    if speaker == "Note" or _SKIP.search(speaker):
        return "skip"
    return "agent" if _AGENT.search(speaker) else "caller"


def weight(text: str) -> float:
    n = len(text.split())
    return min(n, MAX_WEIGHT_WORDS) * (FILLER_WEIGHT if n < FILLER_WORDS else 1.0)


def scorable_turns(transcript: str) -> list[dict]:
    out = []
    for t in split_turns(transcript):
        r = role(t["speaker"])
        if r != "skip" and t["text"].strip():
            out.append(t | {"role": r, "words": len(t["text"].split()), "weight": weight(t["text"])})
    return out


def label(score: float, magnitude: float) -> str:
    if score >= POS:
        return "positive"
    if score <= NEG:
        return "negative"
    return "mixed" if magnitude >= MIXED_MAGNITUDE else "neutral"


def _wavg(turns: list[dict], key: str) -> float:
    total = sum(t["weight"] for t in turns)
    return sum(t[key] for t in turns) / len(turns) if total == 0 else sum(t["weight"] * t[key] for t in turns) / total


def _window(turns: list[dict], head: bool) -> list[dict]:
    k = max(1, math.ceil(len(turns) * WINDOW))
    return turns[:k] if head else turns[-k:]


def summarize(turns: list[dict], total_turns: int) -> dict:
    if not turns:
        return {"turns": 0}
    overall, magnitude = _wavg(turns, "score"), _wavg(turns, "magnitude")
    start, end = _wavg(_window(turns, True), "score"), _wavg(_window(turns, False), "score")
    real = [t for t in turns if t["words"] >= FILLER_WORDS] or turns   # don't let "ok" be the low point
    low, high = min(real, key=lambda t: t["score"]), max(real, key=lambda t: t["score"])
    mean = sum(t["score"] for t in turns) / len(turns)
    return {
        "turns": len(turns), "overall": round(overall, 4), "magnitude": round(magnitude, 4),
        "label": label(overall, magnitude), "start": round(start, 4), "end": round(end, 4),
        "shift": round(end - start, 4), "end_label": label(end, _wavg(_window(turns, False), "magnitude")),
        "min_score": round(low["score"], 4), "min_turn": low["n"],
        "min_position": round((low["n"] - 1) / max(total_turns - 1, 1), 3),
        "max_score": round(high["score"], 4), "max_turn": high["n"],
        "volatility": round(math.sqrt(sum((t["score"] - mean) ** 2 for t in turns) / len(turns)), 4),
    }


def score_call(transcript: str, model) -> tuple[list[dict], dict]:
    """model: anything with .score(list[str]) -> list[dict]. Returns (turn rows, call summary)."""
    turns = scorable_turns(transcript)
    scores = model.score([t["text"] for t in turns]) if turns else []
    rows = [{"turn": t["n"], "speaker": t["speaker"], "role": t["role"], "words": t["words"], "weight": t["weight"],
             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}} for t, s in zip(turns, scores)]
    total = len(split_turns(transcript))
    summary: dict = {"call_turns": total}
    for r in ROLES:
        s = summarize([dict(x, n=x["turn"]) for x in rows if x["role"] == r], total)
        summary |= {f"{r}_{k}": v for k, v in s.items()}
    return rows, summary
