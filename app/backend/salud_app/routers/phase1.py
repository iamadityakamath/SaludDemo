"""Agent Call Reviews: the call list, a call's transcript and stored results, analysis (fill in missing parts or
Re-analyze) and uploads. Same paths and response shapes as the team dashboard website."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import get_settings
from ..phase1 import analysis, store
from ..phase1.checklist import ITEM_DEFINITIONS, TRANSCRIPT_ITEMS
from ..phase1.prompt import split_turns

router = APIRouter(tags=["agent call reviews"])


@router.get("/analyze/list")
def call_list() -> dict:
    llm = analysis.llm_ready()
    calls = [c | {"status": analysis.call_status(c, llm)} for c in store.analysis_list()]
    return {"me": get_settings().salud_app_user, "website": False, "llm": llm, "items": TRANSCRIPT_ITEMS,
            "definitions": ITEM_DEFINITIONS, "calls": calls}


@router.get("/call")
def call(id: str = Query(..., max_length=200)) -> dict:
    text = store.transcript(id)
    if text is None:
        raise HTTPException(404, "unknown call")
    return {"text": text, "turns": split_turns(text)}


@router.get("/result")
def result(id: str = Query(..., max_length=200)) -> dict:
    try:
        return analysis.results(id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from None


class RunBody(BaseModel):
    id: str = Field(max_length=200)
    redo: bool = False


@router.post("/analyze/run")
def run(body: RunBody) -> dict:
    """Fill in what the call is missing (redo false) or recompute and replace both parts (redo true)."""
    try:
        return analysis.analyze(body.id, redo=body.redo)
    except LookupError as e:
        raise HTTPException(404, str(e)) from None


class UploadBody(BaseModel):
    transcripts: list[str] = Field(min_length=1, max_length=analysis.MAX_UPLOAD_CALLS)


@router.post("/analyze/new")
def upload(body: UploadBody) -> dict:
    """Save uploaded transcripts as new calls and start their analysis in the background."""
    if any(len(t) > analysis.MAX_TRANSCRIPT_CHARS for t in body.transcripts):
        raise HTTPException(400, f"a transcript is too long (max {analysis.MAX_TRANSCRIPT_CHARS:,} characters)")
    if any(not t.strip() for t in body.transcripts):
        raise HTTPException(400, "a transcript is empty")
    ids = [store.add_new_call(t, get_settings().salud_app_user) for t in body.transcripts]
    analysis.analyze_in_background(ids)
    return {"ids": ids}

