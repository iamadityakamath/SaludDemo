"""Metrics for QA scorers against synthetic truth or human ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from .targets import TARGETS


def score_table(truth: pd.DataFrame, pred: pd.DataFrame, targets=TARGETS) -> pd.DataFrame:
    """Per-target accuracy, macro-F1, Cohen's kappa and coverage (share with a valid prediction).

    Rows where the human marked N/A are dropped for that target. Missing predictions count as wrong.
    """
    m = truth.merge(pred, on="contact_id", suffixes=("_true", "_pred"))
    rows = []
    for t in targets:
        y = m[f"{t}_true"]
        p = m[f"{t}_pred"]
        keep = y.notna() & (y != "N/A")
        y, p = y[keep].astype(str), p[keep]
        covered = p.notna()
        p = p.fillna("<missing>").astype(str)
        rows.append({
            "target": t, "n": int(keep.sum()),
            "accuracy": accuracy_score(y, p) if len(y) else np.nan,
            "macro_f1": f1_score(y, p, average="macro", zero_division=0) if len(y) else np.nan,
            "kappa": cohen_kappa_score(y, p) if len(set(y)) > 1 else np.nan,
            "coverage": float(covered.mean()) if len(y) else np.nan,
            "majority_baseline": float(y.value_counts(normalize=True).iloc[0]) if len(y) else np.nan,
        })
    return pd.DataFrame(rows)


def bootstrap_accuracy_ci(truth: pd.DataFrame, pred: pd.DataFrame, target: str, n_boot: int = 1000,
                          seed: int = 0) -> tuple[float, float]:
    m = truth.merge(pred, on="contact_id", suffixes=("_true", "_pred"))
    m = m[m[f"{target}_true"].notna() & (m[f"{target}_true"] != "N/A")]
    hits = (m[f"{target}_true"].astype(str) == m[f"{target}_pred"].astype(str)).to_numpy()
    if len(hits) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(hits, size=(n_boot, len(hits)), replace=True).mean(axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def inter_rater(gt: pd.DataFrame, targets=TARGETS) -> pd.DataFrame:
    """Cohen's kappa for every pair of raters who tagged the same calls."""
    raters = sorted(gt["rater"].dropna().unique())
    rows = []
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            m = gt[gt["rater"] == a].merge(gt[gt["rater"] == b], on="contact_id", suffixes=("_a", "_b"))
            for t in targets:
                pair = m[[f"{t}_a", f"{t}_b"]].dropna()
                pair = pair[(pair != "N/A").all(axis=1)]
                rows.append({"rater_a": a, "rater_b": b, "target": t, "n": len(pair),
                             "agreement": float((pair.iloc[:, 0] == pair.iloc[:, 1]).mean()) if len(pair) else np.nan,
                             "kappa": cohen_kappa_score(pair.iloc[:, 0], pair.iloc[:, 1])
                             if len(pair) and pair.stack().nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def consensus(gt: pd.DataFrame, targets=TARGETS) -> pd.DataFrame:
    """One label per call: the majority across raters (ties go to the first rater's label)."""
    def mode(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else None
    agg = gt.groupby("contact_id").agg({**{t: mode for t in targets}, "text": "first"})
    return agg.reset_index()
