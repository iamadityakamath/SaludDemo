"""Response shapes of the API (also the contract for the frontend's `src/api/types.ts`)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
