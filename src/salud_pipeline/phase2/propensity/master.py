"""The master modeling file: one row per recorded call, synthetic and real, for both models.

    salud-propensity master      # -> Phase 2/reports/pre_training_data.xlsx (sheet "Model rows")

Synthetic rows come from the master world (configs/synthetic_world_master.json: Salud's time frame, accounts from
mid-2023, calls up to 2026-11-18, 55/55 calibration statistics). Real rows are Salud's 84 calls (Neon `final`).

Columns added on top of the snapshot columns (dictionary.py):
    data_type        'synthetic' or 'real'
    split            'train' / 'test' for synthetic (random 80/20 by patient: all of a patient's calls on one side);
                     'real_test' for real rows (never trained on)
    label_usable     the 30-day label can be trusted: the patient has account data and the 30 days are inside the data
    post_call_row    no payment was taken on the call (the rows the post-call model uses)
    sentiment_source / qa_source   how the call's sentiment and QA score were made

Synthetic sentiment and QA have no transcripts behind them. Each synthetic call gets the five scores (caller overall,
end, shift; agent overall; QA) drawn from the real calls' distribution for the same call outcome: group means shrunk
towards the overall mean (few real calls per group), the real within-group covariance, clipped to the real range;
QA is then snapped to the values real QA takes (items passed / items applicable).
Pre-call model: pre-call columns, target paid_30d. Post-call model: post_call_row rows, all columns, target
paid_after_call_30d. Both: label_usable rows, split for train/test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...paths import PHASE2_REPORTS, PHASE2_WORLDS
from .snapshots import build, score_sentiment
from .sources import load_real, load_synthetic

MASTER_XLSX = PHASE2_REPORTS / "pre_training_data.xlsx"
SENTIMENT_CACHE = PHASE2_REPORTS / "sentiment_cache"
REAL_QA = PHASE2_REPORTS / "qa_real.csv"
SCORES = ["caller_sentiment_overall", "caller_sentiment_end", "caller_sentiment_shift", "agent_sentiment_overall",
          "qa_score"]
SHRINK = 8          # pseudo-calls pulling a small outcome group's mean towards the overall mean
TEST_SHARE = 0.2
SEED = 42


def outcome_group(df: pd.DataFrame) -> pd.Series:
    """How the call ended, in the groups the sentiment sampler conditions on."""
    g = pd.Series("other", index=df.index)
    for col, name in [("pursued_insurance", "insurance"), ("contested_or_refused", "contested"),
                      ("transferred", "transferred"), ("non_committal", "non_committal"),
                      ("promised_to_pay", "promised")]:
        g[df[col].fillna(False).astype(bool)] = name
    g[df["paid_on_call"].fillna(False).astype(bool)] = "paid"
    g[~df["spoke_with_agent"].fillna(False).astype(bool)] = "no_agent"
    return g


def fit_sampler(real: pd.DataFrame) -> dict:
    """Per-group means (shrunk), pooled within-group covariance and range, from the real calls."""
    r = real.assign(_g=outcome_group(real)).dropna(subset=SCORES)
    overall = r[SCORES].mean()
    means = {}
    for g, d in r.groupby("_g"):
        n = len(d)
        means[g] = (n * d[SCORES].mean() + SHRINK * overall) / (n + SHRINK)
    resid = r[SCORES] - r["_g"].map(lambda g: means[g]).apply(pd.Series)[SCORES].to_numpy()
    return {"means": means, "overall": overall, "cov": np.cov(resid.to_numpy(float).T),
            "lo": r[SCORES].min(), "hi": r[SCORES].max(),
            "caller_missing": float(real["caller_sentiment_overall"].isna().mean()),
            # real QA = items passed / items applicable, so it takes few values; synthetic QA snaps to them
            "qa_grid": np.sort(r["qa_score"].round(4).unique())}


def sample_scores(rows: pd.DataFrame, sampler: dict, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = outcome_group(rows)
    mu = np.vstack([sampler["means"].get(g, sampler["overall"])[SCORES].to_numpy(float) for g in groups])
    draw = mu + rng.multivariate_normal(np.zeros(len(SCORES)), sampler["cov"], size=len(rows))
    out = pd.DataFrame(draw, columns=SCORES, index=rows.index).clip(sampler["lo"], sampler["hi"], axis=1).round(4)
    grid = sampler["qa_grid"]
    out["qa_score"] = grid[np.abs(out["qa_score"].to_numpy()[:, None] - grid[None, :]).argmin(axis=1)]
    no_caller = rng.random(len(rows)) < sampler["caller_missing"]   # calls with no caller speech
    out.loc[no_caller, ["caller_sentiment_overall", "caller_sentiment_end", "caller_sentiment_shift"]] = np.nan
    return out


def patient_split(guarantors: pd.Series, test_share: float = TEST_SHARE, seed: int = SEED) -> pd.Series:
    """Random train/test by patient, so no patient is on both sides."""
    ids = np.sort(guarantors.dropna().unique())
    test = set(np.random.default_rng(seed).choice(ids, size=round(test_share * len(ids)), replace=False))
    return guarantors.map(lambda g: "test" if g in test else "train")


def finish(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "data_type", data_type)
    df.insert(1, "split", patient_split(df["guarantor_id"]) if data_type == "synthetic" else "real_test")
    df.insert(2, "label_usable", df["label_reliable"] & df["window_complete"])
    df.insert(3, "post_call_row", ~df["paid_on_call"].astype(bool))
    return df


def build_master(world: str = "master") -> pd.DataFrame:
    real_world = load_real()
    sent = score_sentiment(real_world, SENTIMENT_CACHE / "real.csv")
    qa = pd.read_csv(REAL_QA).set_index("call_id")["qa_score"] if REAL_QA.exists() else None
    real = finish(build(real_world, sentiment=sent, qa=qa), "real")
    real["sentiment_source"] = "local sentiment model on the transcript"
    real["qa_source"] = "Stage 1 AI review of the transcript" if qa is not None else "missing"

    syn = build(load_synthetic(PHASE2_WORLDS[world]))
    syn[SCORES] = sample_scores(syn, fit_sampler(real))
    syn = finish(syn, "synthetic")
    syn["sentiment_source"] = "sampled from real calls with the same outcome"
    syn["qa_source"] = "sampled from real calls with the same outcome"
    return pd.concat([syn, real], ignore_index=True)


def write_master(df: pd.DataFrame, path=MASTER_XLSX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet = df.copy()
    sheet["as_of"] = pd.to_datetime(sheet["as_of"]).dt.strftime("%Y-%m-%d %H:%M")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sheet.to_excel(xw, sheet_name="Model rows", index=False)
        ws = xw.book["Model rows"]
        ws.freeze_panes = "E2"
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in list(col)[:50])
            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 40)


def audit(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Integrity checks on the master file: (check, passed, detail). Run after every build."""
    from . import dictionary as D

    out: list[tuple[str, bool, str]] = []

    def check(name, bad, detail=""):
        n = int(bad.sum()) if hasattr(bad, "sum") else int(bad)
        out.append((name, n == 0, f"{n} rows" + (f"; {detail}" if detail else "")))

    num = lambda c: pd.to_numeric(df[c].astype(float) if df[c].dtype == bool else df[c], errors="coerce")  # noqa: E731
    syn, real = df[df.data_type == "synthetic"], df[df.data_type == "real"]
    check("call_id unique", df["call_id"].duplicated())
    check("data_type is synthetic or real", ~df["data_type"].isin(["synthetic", "real"]))
    check("synthetic split is train/test", ~syn["split"].isin(["train", "test"]))
    check("real split is real_test", real["split"] != "real_test")
    shared = set(syn.loc[syn.split == "train", "guarantor_id"]) & set(syn.loc[syn.split == "test", "guarantor_id"])
    check("no patient in both train and test", len(shared))
    check("test share 15-25% of synthetic patients",
          not 0.15 <= syn.loc[syn.split == "test", "guarantor_id"].nunique() / max(syn["guarantor_id"].nunique(), 1) <= 0.25)
    check("label_usable = label_reliable & window_complete",
          df["label_usable"] != (df["label_reliable"].astype(bool) & df["window_complete"].astype(bool)))
    check("post_call_row = not paid_on_call", df["post_call_row"] == df["paid_on_call"].astype(bool))
    u = df[df["label_usable"]]
    check("paid after the call implies paid within 30 days", u["paid_after_call_30d"].astype(bool) & ~u["paid_30d"].astype(bool))
    d1 = pd.to_numeric(u["days_to_first_payment"], errors="coerce")
    check("days_to_first_payment set exactly when paid_30d", d1.notna() != u["paid_30d"].astype(bool))
    check("days_to_first_payment within 0-30", (d1 < 0) | (d1 > 30))
    check("accounts = hospital + physician", num("n_accounts") != num("n_hospital_accounts") + num("n_physician_accounts"))
    check("payments 90d <= payments 365d", num("n_payments_90d") > num("n_payments_365d"))
    neg = sum((num(c) < 0) for c in D.MODEL_PRE if c in df)
    check("no negative pre-call field", neg > 0)
    check("last placement not before first", num("days_since_last_placement") > num("days_since_first_placement"))
    check("days_since_last_payment empty exactly when never paid",
          num("days_since_last_payment").isna() != (num("n_payments_ever") == 0))
    check("days_since_last_payment within 365 days when paid in the last year",
          (num("n_payments_365d") > 0) & (num("days_since_last_payment") > 365))
    for c in ["caller_sentiment_overall", "caller_sentiment_end", "agent_sentiment_overall"]:
        check(f"{c} within -1..1", num(c).abs() > 1)
    check("qa_score within 0..1", (num("qa_score") < 0) | (num("qa_score") > 1))
    check("paid_share_so_far within 0..1", (num("paid_share_so_far") < 0) | (num("paid_share_so_far") > 1))
    check("last payment amount 0 exactly when never paid",
          (num("last_payment_amount") == 0) != (num("n_payments_ever") == 0))
    real_qa = set(num("qa_score")[df.data_type == "real"].dropna().round(4))
    check("synthetic QA only takes real QA values", ~num("qa_score")[df.data_type == "synthetic"].dropna().round(4).isin(real_qa))
    # leakage smell: no single model field should almost determine the target
    from sklearn.metrics import roc_auc_score
    su = u[u.data_type == "synthetic"]
    for target, fields, rows in [("paid_30d", D.MODEL_PRE, su),
                                 ("paid_after_call_30d", D.MODEL_PRE + D.MODEL_POST_EXTRA, su[su.post_call_row])]:
        y = rows[target].astype(int)
        worst = max(abs(roc_auc_score(y, pd.to_numeric(rows[c].astype(float) if rows[c].dtype == bool else rows[c],
                                                         errors="coerce").fillna(-1)) - 0.5) + 0.5 for c in fields)
        out.append((f"no single field near-perfect for {target} (leak check)", worst < 0.9, f"max single-field AUC {worst:.3f}"))
    return out
