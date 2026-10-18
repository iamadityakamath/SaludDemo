"""LLM vs traditional ML on call QA scoring (roadmap Phase 4 comparison study).

Test set: the first N calls of an LLM-written synthetic run (llm_order.txt). Every approach is scored
on exactly those calls:
  - LLM:            zero-shot one-prompt review (same prompt and cache as the review page)
  - ML (LLM calls):  TF-IDF + logistic regression trained on the remaining LLM-written calls,
                    at several training sizes (learning curve)
  - ML (template):   the same model trained on template transcripts, excluding the test calls
                    (both runs share one synthetic world, so their labels would leak)
  - Majority:        always predict the most common label

Metrics: accuracy, macro-F1 and Cohen's kappa per target, with bootstrap 95% CIs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from .ml_scorer import MLScorer
from .targets import TARGETS, TRANSCRIPT_ITEMS, load_synthetic

METRICS = {
    "accuracy": lambda y, p: accuracy_score(y, p),
    "macro_f1": lambda y, p: f1_score(y, p, average="macro", zero_division=0),
    "kappa": lambda y, p: cohen_kappa_score(y, p) if len(set(y)) > 1 else np.nan,
}


def load_llm_split(run: Path, test_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_synthetic(run).set_index("contact_id")
    order = [c for c in (run / "llm_order.txt").read_text().split() if c in data.index]
    test_ids, train_ids = order[:test_n], order[test_n:]
    return data.loc[test_ids].reset_index(), data.loc[train_ids].reset_index()


def metric_with_ci(y: np.ndarray, p: np.ndarray, name: str, n_boot: int = 1000, seed: int = 0) -> tuple:
    fn = METRICS[name]
    point = fn(y, p)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if name == "kappa" and len(set(y[idx])) < 2:
            continue
        boots.append(fn(y[idx], p[idx]))
    lo, hi = (np.nanquantile(boots, 0.025), np.nanquantile(boots, 0.975)) if boots else (np.nan, np.nan)
    return point, lo, hi


def evaluate(truth: pd.DataFrame, pred: pd.DataFrame, approach: str) -> pd.DataFrame:
    m = truth.merge(pred, on="contact_id", suffixes=("_true", "_pred"))
    rows = []
    for t in TARGETS:
        y = m[f"{t}_true"].astype(str).to_numpy()
        p = m[f"{t}_pred"].fillna("<missing>").astype(str).to_numpy()
        row = {"approach": approach, "target": t, "n": len(y)}
        for name in METRICS:
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = metric_with_ci(y, p, name)
        rows.append(row)
    return pd.DataFrame(rows)


def majority(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = {"contact_id": test["contact_id"].to_numpy()}
    for t in TARGETS:
        out[t] = np.full(len(test), train[t].mode().iloc[0], dtype=object)
    return pd.DataFrame(out)


def learning_curve(train: pd.DataFrame, test: pd.DataFrame, sizes: list[int], reps: int) -> pd.DataFrame:
    rows = []
    for size in sizes:
        for rep in range(reps if size < len(train) else 1):
            sub = train.sample(n=size, random_state=rep)
            pred = MLScorer().fit(sub).predict(test)
            m = test.merge(pred, on="contact_id", suffixes=("_true", "_pred"))
            for t in TARGETS:
                rows.append({"train_size": size, "rep": rep, "target": t,
                             "accuracy": accuracy_score(m[f"{t}_true"].astype(str), m[f"{t}_pred"].astype(str))})
    return pd.DataFrame(rows).groupby(["train_size", "target"])["accuracy"].agg(["mean", "std"]).reset_index()


def summary_table(results: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Targets x approaches, 'point [lo, hi]' strings, plus a checklist-average row."""
    def fmt(r):
        return f"{r[metric]:.2f} [{r[metric + '_lo']:.2f}, {r[metric + '_hi']:.2f}]"
    table = results.assign(cell=results.apply(fmt, axis=1)).pivot(index="target", columns="approach", values="cell")
    table = table.reindex(TARGETS)
    avg = results[results["target"].isin(TRANSCRIPT_ITEMS)].groupby("approach")[metric].mean()
    table.loc["checklist average (7 items)"] = {a: f"{v:.2f}" for a, v in avg.items()}
    return table


def to_markdown(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.to_numpy()]) + " |")
    return "\n".join(lines)
