"""Split the Phase 1 calls across the team's raters.

    salud-phase1 assign --raters avni,sharayu,aditya,ibrahim,ananya,rudrakshi

Each real call gets two raters (so rater agreement can be measured on Salud's own calls) and each
synthetic call one. Work is spread evenly: real calls first, to whoever has the fewest real calls,
then synthetic calls to whoever has the fewest calls overall. Running it again only fills calls that
are short of raters; existing assignments are never moved.
"""

from __future__ import annotations


def rater_name(name: str) -> str:
    """Rater names are stored lower case, so 'Avni' and 'avni ' are the same person."""
    return name.strip().lower()


def plan(calls: list[dict], raters: list[str], existing: dict[str, list[str]],
         real_per_call: int = 2, synthetic_per_call: int = 1) -> list[tuple[str, str]]:
    """New (call_id, rater) pairs that bring every call up to its number of raters.

    calls: [{call_id, source}]; existing: call_id -> raters already assigned (any rater, kept as is).
    """
    raters = list(dict.fromkeys(rater_name(r) for r in raters if r.strip()))
    if not raters:
        raise ValueError("no raters given")
    load = {r: 0 for r in raters}
    real_load = {r: 0 for r in raters}
    source = {c["call_id"]: c["source"] for c in calls}
    for cid, rs in existing.items():
        for r in rs:
            if r in load:
                load[r] += 1
                real_load[r] += source.get(cid) == "real"

    new = []
    for c in sorted(calls, key=lambda c: c["source"] != "real"):  # real calls first
        real = c["source"] == "real"
        have = set(existing.get(c["call_id"], []))
        need = min((real_per_call if real else synthetic_per_call), len(raters)) - len(have)
        for _ in range(max(0, need)):
            free = [r for r in raters if r not in have]
            if not free:
                break
            pick = min(free, key=lambda r: ((real_load[r], load[r]) if real else (load[r],), raters.index(r)))
            have.add(pick)
            load[pick] += 1
            real_load[pick] += real
            new.append((c["call_id"], pick))
    return new
