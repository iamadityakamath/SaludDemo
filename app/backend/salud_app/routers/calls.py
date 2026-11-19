"""Agent Call Reviews: every call (real and synthetic) with its transcript, sentiment and AI review."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from ..repositories import calls as repo
from ..schemas import CallDetail, CallList

router = APIRouter(prefix="/calls", tags=["calls"])


@router.get("", response_model=CallList)
def list_calls(source: Literal["real", "synthetic", "pasted"] | None = None,
               search: str | None = Query(None, max_length=100),
               limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)) -> CallList:
    total, items = repo.list_calls(source=source, search=search, limit=limit, offset=offset)
    return CallList(total=total, items=items)


@router.get("/{key:path}", response_model=CallDetail)
def get_call(key: str) -> CallDetail:
    call = repo.get_call(key)
    if call is None:
        raise HTTPException(status_code=404, detail="call not found")
    return call
