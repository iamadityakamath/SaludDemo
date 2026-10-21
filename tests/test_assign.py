from collections import Counter

import pytest

from salud_pipeline.phase1.assign import plan

TEAM = ["avni", "sharayu", "aditya", "ibrahim", "ananya", "rudrakshi"]
CALLS = [{"call_id": f"real/r{i}.txt", "source": "real"} for i in range(10)] + \
        [{"call_id": f"synthetic/s{i}.txt", "source": "synthetic"} for i in range(50)]


def test_real_calls_get_two_raters_and_work_is_even():
    pairs = plan(CALLS, TEAM, {})
    per_call = Counter(c for c, _ in pairs)
    assert all(per_call[c["call_id"]] == (2 if c["source"] == "real" else 1) for c in CALLS)
    assert len(set(pairs)) == len(pairs) == 70                       # never the same rater twice on a call
    load = Counter(r for _, r in pairs)
    real = Counter(r for c, r in pairs if c.startswith("real/"))
    assert max(load.values()) - min(load.values()) <= 1
    assert max(real.values()) - min(real.values()) <= 1


def test_rerun_only_fills_gaps_and_keeps_existing():
    existing = {"real/r0.txt": ["avni"], "synthetic/s0.txt": ["ibrahim"]}
    pairs = plan(CALLS, [" Avni", "SHARAYU", "aditya"], existing)
    assert ("synthetic/s0.txt", "avni") not in pairs and not any(c == "synthetic/s0.txt" for c, _ in pairs)
    added_r0 = [r for c, r in pairs if c == "real/r0.txt"]
    assert len(added_r0) == 1 and added_r0 != ["avni"]               # one more rater, not avni again
    assert {r for _, r in pairs} <= {"avni", "sharayu", "aditya"}  # names normalized to lower case
    full = {c: list(rs) for c, rs in existing.items()}
    for c, r in pairs:
        full.setdefault(c, []).append(r)
    assert plan(CALLS, ["avni", "sharayu", "aditya"], full) == []   # nothing left to do


def test_needs_raters():
    with pytest.raises(ValueError):
        plan(CALLS, [" ", ""], {})
