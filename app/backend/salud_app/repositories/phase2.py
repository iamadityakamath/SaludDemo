"""Phase 2 (payment propensity) overview from local files, aggregates only.

Reads `Phase 2/reports/pre_training_data.xlsx` (the master file: synthetic + Salud's real calls) and the saved feature
sets. Never returns individual rows: the master file includes Salud's real calls.
"""

from __future__ import annotations

import json

import pandas as pd

from ..config import get_settings
from ..schemas import Phase2Summary

_cache: dict = {"mtime": None, "summary": None}


def _rate(s: pd.Series) -> float | None:
    s = pd.to_numeric(s.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}), errors="coerce")
    return round(float(s.mean()), 3) if s.notna().any() else None


def summary() -> Phase2Summary:
    s = get_settings()
    if not s.master_file.exists():
        return Phase2Summary(available=False)
    mtime = s.master_file.stat().st_mtime
    if _cache["mtime"] == mtime:
        return _cache["summary"]
    m = pd.read_excel(s.master_file)
    usable = m["label_usable"].astype(str).str.lower() == "true"
    post = m["post_call_row"].astype(str).str.lower() == "true"
    parts = []
    for (dtype, split), g in m.groupby(["data_type", "split"]):
        gu = usable.loc[g.index]
        parts.append({
            "data_type": dtype, "split": split, "calls": int(len(g)), "patients": int(g["guarantor_id"].nunique()),
            "usable": int(gu.sum()), "pay_rate_pre_call": _rate(g.loc[gu, "paid_30d"]),
            "pay_rate_post_call": _rate(g.loc[gu & post.loc[g.index], "paid_after_call_30d"]),
        })
    as_of = pd.to_datetime(m["as_of"], errors="coerce")
    sets = json.loads(s.feature_sets_file.read_text())["sets"] if s.feature_sets_file.exists() else {}
    out = Phase2Summary(available=True, rows=int(len(m)), by_part=parts,
                        date_range={"first_call": str(as_of.min().date()) if as_of.notna().any() else None,
                                    "last_call": str(as_of.max().date()) if as_of.notna().any() else None},
                        feature_sets=sets)
    _cache.update(mtime=mtime, summary=out)
    return out
