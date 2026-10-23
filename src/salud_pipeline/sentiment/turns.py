"""Which turns get scored, and how much each one counts.

Speakers are mapped to a role: 'agent' (Agent, Agent (inferred)), 'skip' (editorial notes, automated
systems, unlabeled blocks) or 'caller' (everyone else on the line: the patient, a representative,
an account holder, an insurance representative). Turn numbers are the dashboard's, from split_turns.
"""

from __future__ import annotations

import re

from ..phase1.prompt import split_turns

_SKIP = re.compile(r"note|automated|system|\bivr\b|recording|voicemail|music", re.I)
_AGENT = re.compile(r"^\s*agent\b", re.I)

FILLER_WORDS = 3      # turns shorter than this ("ok", "mm-hm") barely count
FILLER_WEIGHT = 0.2
MAX_WEIGHT_WORDS = 40  # longer turns don't count more than a 40-word turn


def role(speaker: str) -> str:
    if speaker == "Note" or _SKIP.search(speaker):
        return "skip"
    return "agent" if _AGENT.search(speaker) else "caller"


def words(text: str) -> int:
    return len(text.split())


def weight(text: str) -> float:
    """How much a turn counts in call-level averages: its length, capped, with filler turns down-weighted."""
    n = words(text)
    return min(n, MAX_WEIGHT_WORDS) * (FILLER_WEIGHT if n < FILLER_WORDS else 1.0)


def scorable_turns(transcript: str) -> list[dict]:
    """[{'n', 'speaker', 'role', 'text', 'words', 'weight'}] for the caller and agent turns of a call."""
    out = []
    for t in split_turns(transcript):
        r = role(t["speaker"])
        if r != "skip" and t["text"].strip():
            out.append(t | {"role": r, "words": words(t["text"]), "weight": weight(t["text"])})
    return out
