"""Agreement metrics: AI vs human tags, and rater vs rater. Plain numpy, no ML dependencies.

N/A is a real answer here: if a rater says N/A and the AI says Yes, that is a disagreement.
Calls the rater left blank for a target are excluded for that target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .checklist import TARGETS


def cohen_kappa(a, b) -> float:
    """Agreement beyond chance for two label sequences (any number of categories)."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if len(a) == 0:
        return np.nan
    labels = sorted(set(a) | set(b), key=str)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == lab)) * float(np.mean(b == lab)) for lab in labels)
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_ci(hits: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(hits) == 0:
        return (np.nan, np.nan)
    boots = np.random.default_rng(seed).choice(hits, size=(n_boot, len(hits)), replace=True).mean(axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def agreement_table(truth: pd.DataFrame, pred: pd.DataFrame, key: str = "call_id",
                    targets=TARGETS) -> pd.DataFrame:
    """Per target: how often `pred` matches `truth`, with a 95% CI and Cohen's kappa.

    A missing prediction counts as a disagreement and is also reported in `pred_missing`.
    """
    m = truth.merge(pred, on=key, suffixes=("_true", "_pred"))
    rows = []
    for t in targets:
        y = m[f"{t}_true"]
        keep = y.notna() & (y.astype(str) != "")
        y = y[keep].astype(str).to_numpy()
        p_raw = m.loc[keep, f"{t}_pred"]
        p = p_raw.fillna("<missing>").astype(str).to_numpy()
        hits = (y == p).astype(float)
        lo, hi = bootstrap_ci(hits)
        rows.append({"target": t, "n": len(y),
                     "agreement": float(hits.mean()) if len(y) else np.nan, "ci_low": lo, "ci_high": hi,
                     "kappa": cohen_kappa(y, p) if len(set(y)) > 1 else np.nan,
                     "pred_missing": int(p_raw.isna().sum())})
    return pd.DataFrame(rows)


def inter_rater(labels: pd.DataFrame, key: str = "call_id", targets=TARGETS) -> pd.DataFrame:
    """Agreement and Cohen's kappa for every pair of raters who tagged the same calls."""
    raters = sorted(labels["rater"].dropna().unique())
    rows = []
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            m = labels[labels["rater"] == a].merge(labels[labels["rater"] == b], on=key, suffixes=("_a", "_b"))
            for t in targets:
                pair = m[[f"{t}_a", f"{t}_b"]].replace("", np.nan).dropna()
                rows.append({"rater_a": a, "rater_b": b, "target": t, "n": len(pair),
                             "agreement": float((pair.iloc[:, 0] == pair.iloc[:, 1]).mean()) if len(pair) else np.nan,
                             "kappa": cohen_kappa(pair.iloc[:, 0], pair.iloc[:, 1])
                             if len(pair) and pair.stack().nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def consensus(labels: pd.DataFrame, key: str = "call_id", targets=TARGETS) -> pd.DataFrame:
    """One label per call: the answer most raters gave. A tie means no consensus (None), so that call is
    left out of AI-vs-human agreement for that item; the split shows up in inter-rater agreement instead."""
    def mode(s):
        counts = s.replace("", np.nan).dropna().value_counts()
        if counts.empty or (len(counts) > 1 and counts.iloc[0] == counts.iloc[1]):
            return None
        return counts.index[0]
    return labels.groupby(key).agg({t: mode for t in targets}).reset_index()
