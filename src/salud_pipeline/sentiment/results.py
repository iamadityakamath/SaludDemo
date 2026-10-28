"""Score calls and keep the results in the team database (phase1.turn_sentiment, phase1.call_sentiment).

score_call() is pure: transcript -> (turn rows, call summary). score_and_save() adds the database write, and is
what both `salud-sentiment run` and the website's Analyze button use. Turn rows hold numbers and labels only,
never transcript text.
"""

from __future__ import annotations

import pandas as pd

from ..phase1.prompt import split_turns
from .aggregate import summarize
from .turns import scorable_turns

ROLES = ("caller", "agent")


def score_call(transcript: str, model) -> tuple[list[dict], dict]:
    """model: anything with .score(list[str]) -> list[dict] (model.py). Returns (turn rows, summary)."""
    turns = scorable_turns(transcript)
    scores = model.score([t["text"] for t in turns]) if turns else []
    rows = [{"turn": t["n"], "speaker": t["speaker"], "role": t["role"], "words": t["words"], "weight": t["weight"],
             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}}
            for t, s in zip(turns, scores)]
    total = len(split_turns(transcript))
    summary = {"call_turns": total}
    for role in ROLES:
        s = summarize([dict(r, n=r["turn"]) for r in rows if r["role"] == role], total)
        summary |= {f"{role}_{k}": v for k, v in s.items()}
    return rows, summary


def score_and_save(store, call_id: str, transcript: str, model, model_name: str) -> dict:
    rows, summary = score_call(transcript, model)
    store.save_sentiment(call_id, rows, summary, model_name)
    return summary


def score_calls(transcripts: dict[str, str], model) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Many calls at once, as tables (call_id on every row). Used for checks and exports."""
    turns, calls = [], []
    for cid, text in transcripts.items():
        rows, summary = score_call(text, model)
        turns += [{"call_id": cid} | r for r in rows]
        calls.append({"call_id": cid, "source": cid.split("/", 1)[0]} | summary)
    return pd.DataFrame(turns), pd.DataFrame(calls)
