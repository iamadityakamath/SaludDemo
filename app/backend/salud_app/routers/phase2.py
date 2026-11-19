"""Phase 2 (payment propensity): an overview of the master training data and the selected feature sets."""

from fastapi import APIRouter

from ..repositories import phase2 as repo
from ..schemas import Phase2Summary

router = APIRouter(prefix="/phase2", tags=["phase2"])


@router.get("/summary", response_model=Phase2Summary)
def summary() -> Phase2Summary:
    return repo.summary()
