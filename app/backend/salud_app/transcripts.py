"""Split a transcript into speaker turns.

Transcripts are plain text, one turn per line or paragraph, starting with a speaker label: `Agent: ...`,
`Caller: ...`, `Patient: ...`, `Patient Representative: ...`, `Insurance Representative: ...`, `Agent (inferred): ...`,
`Patient / Account Holder: ...`, `Automated System: ...`, `Unknown Speaker: ...`. Lines without a label continue the
previous turn. An `Editorial note: ...` line (on LLM-written synthetic calls) is not part of the conversation and is
skipped.
"""

from __future__ import annotations

import re

LABEL = re.compile(r"^\s*([A-Za-z][A-Za-z ()/&.-]{1,40}?):\s*(.*)$")
SKIP = ("editorial note",)


def role_of(speaker: str) -> str | None:
    """agent / caller / system / unknown for a speaker label; None if the text before ':' is not a speaker."""
    s = speaker.strip().lower()
    if any(s.startswith(k) for k in SKIP):
        return "skip"
    if "agent" in s:
        return "agent"
    if any(k in s for k in ("automated", "ivr", "system", "voicemail")):
        return "system"
    if any(k in s for k in ("patient", "caller", "customer", "representative", "account holder", "guarantor",
                            "insurance", "family", "spouse")):
        return "caller"
    if "unknown" in s or "speaker" in s:
        return "unknown"
    return None


def parse_turns(text: str | None) -> list[dict]:
    """[{turn, speaker, role, text}], turns numbered from 1 in order."""
    turns: list[dict] = []
    skipping = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LABEL.match(line)
        role = role_of(m.group(1)) if m else None
        if role == "skip":
            skipping = True
        elif role:
            skipping = False
            turns.append({"turn": len(turns) + 1, "speaker": m.group(1).strip(), "role": role, "text": m.group(2)})
        elif turns and not skipping:
            turns[-1]["text"] = (turns[-1]["text"] + " " + line).strip()
    return turns
