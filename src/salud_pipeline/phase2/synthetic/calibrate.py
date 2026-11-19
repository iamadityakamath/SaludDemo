"""Compare a synthetic world with Salud's real package, statistic by statistic.

    salud-synth-calibrate --world "Phase 2/data/synthetic_world_salud"      # report vs the real package (Neon `final`)

Every statistic is computed with the same code on both worlds (sources.World -> snapshots.build), on the same
population: *recorded calls* (agent calls, phone-menu calls, messages left) and one row per patient as of each call.
For the real data each statistic gets a 95% bootstrap interval (resampling calls); a synthetic statistic "matches"
when it falls inside that interval. With 84 real calls the intervals are wide — matching them is necessary, not
sufficient, for a realistic world.

Groups:
    call mix       how calls split by connection, disposition, contact type, result, actions, payment on the call
    patients       accounts, balances, debt age, payment and insurance history at the call date
    outcome        paid within 30 days; paid in the 30 days after the call
    relationships  each feature's single-feature AUC against paid-within-30-days (> 0.5 = higher value -> pays)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..propensity.snapshots import build
from ..propensity.sources import World, load_real, load_synthetic

RESULT_GROUPS = {
    "paid (any)": lambda r: r.str.startswith("Paid", na=False),
    "promised to pay": lambda r: r.eq("Promised to Pay"),
    "non-committal": lambda r: r.eq("Non-Committal"),
    "transferred": lambda r: r.eq("N/A, Transferred"),
    "pursued insurance": lambda r: r.eq("Pursued Insurance Payment"),
    "contested / would not pay": lambda r: r.isin(["Contested Balance", "Said Would Not Pay"]),
    "blank": lambda r: r.isna(),
}
RELATIONSHIP_FEATURES = ["n_payments_90d", "n_payments_365d", "days_since_last_payment", "days_since_first_placement",
                         "days_since_last_placement", "n_accounts", "balance_at_t", "n_adjustments_365d",
                         "n_insurance_payments_365d", "inbound", "spoke_with_agent"]


def _auc(x: pd.Series, y: pd.Series, never_fill: bool = False) -> float:
    x = x.astype(float)
    x = x.fillna(x.max() + 1 if never_fill else 0)
    if y.nunique() < 2 or x.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y.astype(int), x))


def statistics(calls: pd.DataFrame, snap: pd.DataFrame, accounts: pd.DataFrame) -> dict[tuple[str, str], float]:
    """All statistics for one world (or one bootstrap resample). calls and snap rows are aligned by call_id."""
    s: dict[tuple[str, str], float] = {}
    n = len(calls)
    conn = calls["connection"].fillna("(blank)")
    for k in ["Inbound Call", "Outbound Call, Campaign", "Outbound Call, Manual", "Outbound Call, No Connection"]:
        s[("call mix", f"connection: {k}")] = (conn == k).mean()
    disp = calls["disposition"].fillna("(blank)")
    s[("call mix", "disposition: Spoke with CSR")] = (disp == "Spoke with CSR").mean()
    s[("call mix", "disposition: phone menu (IVR)")] = disp.str.contains("IVR").mean()
    s[("call mix", "disposition: message left")] = disp.isin(["Left Message With Person", "Voicemail Did Leave Message"]).mean()
    ctype = calls["contact_type"].fillna("(blank)")
    for k in ["Patient", "(blank)", "Other", "Non-patient Guarantor", "Insurance"]:
        s[("call mix", f"contact type: {k}")] = (ctype == k).mean()
    res = calls["collection_result"]
    for k, f in RESULT_GROUPS.items():
        s[("call mix", f"result: {k}")] = f(res).mean()
    s[("call mix", "payment taken on the call")] = (calls["payment_on_call"].fillna(0) > 0).mean()
    acts = calls["actions"].fillna("")
    for k, pat in [("shared balance", "Verbally Shared"), ("payment plan", "Payment Plan"),
                   ("financial assistance", "FA |Financial Assistance"), ("statement / itemized bill", "Statement|Itemized")]:
        s[("call mix", f"action: {k}")] = acts.str.contains(pat).mean()
    acts_l = acts.str.lower()
    s[("call mix", "action: insurance")] = (acts_l.str.contains("insurance") & ~acts_l.str.contains("coinsurance")).mean()
    agent = calls["disposition"].eq("Spoke with CSR")
    s[("call mix", "asked about financial assistance")] = calls["asked_fa"].eq("Yes").mean()
    s[("call mix", "collection attempt blank, agent calls")] = calls.loc[agent, "collection_attempt"].isna().mean() \
        if agent.any() else np.nan
    s[("call mix", "collection attempt yes, phone-menu calls")] = \
        calls.loc[calls["disposition"].fillna("").str.contains("IVR"), "collection_attempt"].eq("Yes").mean()
    s[("call mix", "talk seconds, agent calls (median)")] = calls.loc[agent, "talk_seconds"].median() if agent.any() else np.nan

    rel = snap[snap["label_reliable"]]
    s[("patients", "accounts per patient (median)")] = rel["n_accounts"].median()
    s[("patients", "accounts per patient (mean)")] = rel["n_accounts"].mean()
    s[("patients", "share with more than 5 accounts")] = (rel["n_accounts"] > 5).mean()
    s[("patients", "hospital share of accounts")] = (accounts["hbpb"] == "Hospital Account").mean()
    for t in ("Hospital Account", "Physician Account"):
        s[("patients", f"placement balance per account, {t.split()[0].lower()} (median)")] = \
            accounts.loc[accounts["hbpb"] == t, "placement_balance"].median()
    s[("patients", "balance at call (median)")] = rel["balance_at_t"].median()
    s[("patients", "days since first placement (median)")] = rel["days_since_first_placement"].median()
    s[("patients", "days since last placement (median)")] = rel["days_since_last_placement"].median()
    s[("patients", "paid before the call (share)")] = (rel["n_payments_ever"] > 0).mean()
    s[("patients", "payments in the last 365 days (mean)")] = rel["n_payments_365d"].mean()
    s[("patients", "insurance payment in last 365 days (share)")] = (rel["n_insurance_payments_365d"] > 0).mean()
    s[("patients", "adjustment in last 365 days (share)")] = (rel["n_adjustments_365d"] > 0).mean()

    use = rel[rel["window_complete"]]
    s[("outcome", "paid within 30 days")] = use["paid_30d"].mean()
    post = use[~use["paid_on_call"].astype(bool)]
    s[("outcome", "paid in 30 days after the call (no payment on call)")] = post["paid_after_call_30d"].mean()

    for f in RELATIONSHIP_FEATURES:
        s[("relationships", f"AUC {f}")] = _auc(use[f], use["paid_30d"], never_fill=f.startswith("days_since_last_pay"))
    return s


def profile(world: World) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snap = build(world)
    calls = world.calls.set_index("call_id").loc[snap["call_id"]].reset_index()
    called = set(snap.loc[snap["label_reliable"], "guarantor_id"])
    accounts = world.accounts[world.accounts["guarantor_id"].isin(called)]
    return calls, snap, accounts


def compare(real: World, syn: World, n_boot: int = 500, seed: int = 7) -> pd.DataFrame:
    rc, rs, ra = profile(real)
    sc, ss, sa = profile(syn)
    point = statistics(rc, rs, ra)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(rc), len(rc))
        bc, bs = rc.iloc[i].reset_index(drop=True), rs.iloc[i].reset_index(drop=True)
        pats = set(bs.loc[bs["label_reliable"], "guarantor_id"])
        boots.append(statistics(bc, bs, ra[ra["guarantor_id"].isin(pats)]))
    boot = pd.DataFrame(boots)
    synth = statistics(sc, ss, sa)
    rows = []
    for key, v in point.items():
        lo, hi = np.nanpercentile(boot[key], [2.5, 97.5]) if boot[key].notna().any() else (np.nan, np.nan)
        sv = synth.get(key, np.nan)
        rows.append({"group": key[0], "statistic": key[1], "real": v, "real 95% low": lo, "real 95% high": hi,
                     "synthetic": sv, "match": bool(lo <= sv <= hi) if pd.notna(sv) and pd.notna(lo) else False})
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby("group", sort=False)["match"].agg(matched="sum", total="size")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="salud-synth-calibrate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=Path, required=True, help="a synthetic world folder (salud-synth output)")
    ap.add_argument("--out", type=Path, help="write the full table to this CSV")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args(argv)
    table = compare(load_real(), load_synthetic(args.world), n_boot=args.boot)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(table.round(3).to_string(index=False))
    print("\n", summary(table).to_string(), f"\n\nmatched {int(table.match.sum())}/{len(table)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
