"""Turn scores -> one call's numbers, per role.

Report start, end and shift, not just an average: most calls average out near neutral even when the
caller arrives upset and leaves calm (the "sentiment arc"), and end sentiment is the best sign the
issue was handled. Window = first / last 30% of that role's turns, at least one turn.

Labels follow CCAI: a clear score is positive or negative; a near-zero score is 'mixed' when the turns
carried strong feeling in both directions, else 'neutral'. The thresholds are first guesses, to be tuned
against the raters' tags with `salud-sentiment eval`.
"""

from __future__ import annotations

import math

from .turns import FILLER_WORDS

WINDOW = 0.30
POS, NEG = 0.25, -0.25     # |score| beyond this is a clear label
MIXED_MAGNITUDE = 0.5      # near-zero score with at least this much feeling is 'mixed'

LABELS = ["negative", "mixed", "neutral", "positive"]


def label(score: float, magnitude: float) -> str:
    if score >= POS:
        return "positive"
    if score <= NEG:
        return "negative"
    return "mixed" if magnitude >= MIXED_MAGNITUDE else "neutral"


def as_tag(lbl: str) -> str:
    """Map onto the raters' three-way tag: 'mixed' counts as neutral."""
    return "neutral" if lbl == "mixed" else lbl


def _wavg(turns: list[dict], key: str) -> float:
    total = sum(t["weight"] for t in turns)
    if total == 0:
        return sum(t[key] for t in turns) / len(turns)
    return sum(t["weight"] * t[key] for t in turns) / total


def _window(turns: list[dict], head: bool) -> list[dict]:
    k = max(1, math.ceil(len(turns) * WINDOW))
    return turns[:k] if head else turns[-k:]


def summarize(turns: list[dict], total_turns: int) -> dict:
    """turns: one role's scored turns in call order (n, score, magnitude, weight, words).
    total_turns: turns in the whole call, to place the lowest turn on the call's timeline (0 = start, 1 = end)."""
    if not turns:
        return {"turns": 0}
    overall, magnitude = _wavg(turns, "score"), _wavg(turns, "magnitude")
    start, end = _wavg(_window(turns, True), "score"), _wavg(_window(turns, False), "score")
    real = [t for t in turns if t["words"] >= FILLER_WORDS] or turns   # don't let "ok" be the low point
    low, high = min(real, key=lambda t: t["score"]), max(real, key=lambda t: t["score"])
    mean = sum(t["score"] for t in turns) / len(turns)
    return {
        "turns": len(turns),
        "overall": round(overall, 4),
        "magnitude": round(magnitude, 4),
        "label": label(overall, magnitude),
        "start": round(start, 4),
        "end": round(end, 4),
        "shift": round(end - start, 4),
        "end_label": label(end, _wavg(_window(turns, False), "magnitude")),
        "min_score": round(low["score"], 4),
        "min_turn": low["n"],
        "min_position": round((low["n"] - 1) / max(total_turns - 1, 1), 3),
        "max_score": round(high["score"], 4),
        "max_turn": high["n"],
        "volatility": round(math.sqrt(sum((t["score"] - mean) ** 2 for t in turns) / len(turns)), 4),
    }
