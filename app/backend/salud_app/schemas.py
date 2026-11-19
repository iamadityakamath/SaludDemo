"""Response shapes of the API (also the contract for the frontend's `src/api/types.ts`)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Source = Literal["real", "synthetic", "pasted"]


class SentimentSummary(BaseModel):
    """Call-level sentiment from the local sentiment model: scores from -1 (negative) to 1 (positive)."""
    caller_overall: float | None = None
    caller_start: float | None = None
    caller_end: float | None = None
    caller_shift: float | None = None
    caller_label: str | None = None
    caller_end_label: str | None = None
    agent_overall: float | None = None
    agent_label: str | None = None


class CallSummary(BaseModel):
    key: str                                  # "final-<contact id>" (Salud's package) or "p1-<call id>" (Phase 1)
    source: Source
    dataset: Literal["final", "phase1"]
    contacted_at: datetime | None = None
    outcome: str | None = None                # Salud's collection result, or the AI's outcome for Phase 1 calls
    caller_end_sentiment: float | None = None
    caller_end_label: str | None = None
    qa_score: float | None = None
    has_ai_review: bool = False
    turns: int = 0


class CallList(BaseModel):
    total: int
    items: list[CallSummary]


class Turn(BaseModel):
    turn: int
    speaker: str
    role: Literal["agent", "caller", "system", "unknown"]
    text: str
    sentiment: float | None = None            # turn-level score, when the call has turn sentiment
    sentiment_label: str | None = None


class ChecklistItem(BaseModel):
    item: str
    answer: str | None = None
    reason: str | None = None


class AiReview(BaseModel):
    model: str | None = None
    prompt_version: str | None = None
    summary: str | None = None
    outcome: str | None = None
    sentiment: str | None = None
    quality_score: float | None = None
    checklist: list[ChecklistItem] = []
    coaching: list[str] = []


class CallDetail(CallSummary):
    transcript: list[Turn]
    sentiment: SentimentSummary | None = None
    ai_review: AiReview | None = None
    call_facts: dict[str, str | float | None] = {}   # Salud's call record fields (real calls only)


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    phase2_master_file: bool


class Phase2Summary(BaseModel):
    available: bool
    rows: int = 0
    by_part: list[dict] = []                  # rows, patients, usable rows and pay rates per data part
    date_range: dict[str, str | None] = {}
    feature_sets: dict[str, dict[str, list[str]]] = {}
