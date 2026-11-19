"""Shared machinery for the propensity notebooks (Phase 2/notebooks/): data, feature sets, preprocessing, selection,
cross-validation with logging, metrics and model comparison.

Every notebook runs experiments the same way:

    from salud_pipeline.phase2.propensity import modeling as M
    dev, holdout, real = M.load_data()
    pipe = M.pipeline(M.preprocess("log_z"), M.logreg(C=1.0))
    M.run_experiment("LR-03", pipe, dev, M.FS_ALL, notebook="02", description="log1p + z-score, no penalty")
    M.leaderboard()

Rules that keep the comparison honest:
- Everything that learns from data (imputation fill values, scaling, binning, feature selection, calibration)
  sits inside one sklearn Pipeline and is refitted on each training fold only.
- Folds are grouped by patient (one patient never sits on both sides) and stratified by the label.
- The holdout (latest 20% of synthetic rows by date) and the real rows are never used for choosing; only for
  the one-time final tests in notebook 11.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, RobustScaler, StandardScaler

from ...paths import DEFAULT_WORLD, PHASE2_REPORTS, PHASE2_WORLDS
from . import dictionary as D

def world_files(world: str = DEFAULT_WORLD) -> tuple:
    """(call-row CSV, fixed-date CSV) built by `salud-propensity build --world <world>`."""
    tag = "" if world == "original" else f"_{world}"
    return (PHASE2_REPORTS / f"synthetic_snapshots{tag}.csv", PHASE2_REPORTS / f"synthetic_fixed_date{tag}.csv")


SYNTHETIC_CSV, SYNTHETIC_FIXED_CSV = world_files(DEFAULT_WORLD)
REAL_XLSX = PHASE2_REPORTS / "propensity_dataset.xlsx"
RESULTS = PHASE2_REPORTS / "experiments" / "results.csv"
SEED = 42
TARGET_PRE, TARGET_POST = "paid_30d", "paid_after_call_30d"

# ---------------------------------------------------------------- feature sets (decided in notebook 00)

# Duplicates (Spearman |rho| > 0.9 on synthetic): keep the first of each cluster, drop the rest.
DUPLICATES = {
    "balance_at_t": ["placement_balance"],
    "days_since_last_placement": ["days_since_last_discharge"],
    "n_payments_365d": ["ever_paid", "n_payments_ever", "amount_paid_ever", "amount_paid_365d"],
    "n_payments_90d": ["amount_paid_90d"],
    "n_insurance_payments_365d": ["amount_insurance_365d"],
    "n_adjustments_365d": ["amount_adjustments_365d"],
    "action_shared_balance": ["reviewed_balance"],
}
DROPPED_DUPLICATES = [d for ds in DUPLICATES.values() for d in ds]
# Not present in the real extract (it holds only the 84 calls, no contact history): can't transfer.
NO_TRANSFER = ["n_prior_contacts_90d", "n_prior_calls", "days_since_last_contact"]
CONSTANT = ["has_caller_turns"]

FS_ALL = list(D.PRE_FEATURES)
FS_PRUNED = [f for f in FS_ALL if f not in DROPPED_DUPLICATES]
FS_TRANSFER = [f for f in FS_PRUNED if f not in NO_TRANSFER]
POST_EXTRA = [f for f in D.POST_FEATURES if f not in D.PRE_FEATURES and f not in CONSTANT + DROPPED_DUPLICATES]
FEATURE_SETS = {"FS_ALL": FS_ALL, "FS_PRUNED": FS_PRUNED, "FS_TRANSFER": FS_TRANSFER,
                "FS_TRANSFER+POST": FS_TRANSFER + POST_EXTRA}

# Column roles, for transforms that apply to some columns only.
RECENCY = [f for f in D.POST_FEATURES if f.startswith("days_since_last_payment") or f == "days_since_last_contact"]
SKEWED = ["placement_balance", "balance_at_t", "amount_paid_90d", "amount_paid_365d", "amount_paid_ever",
          "amount_insurance_365d", "amount_adjustments_365d", "n_payments_90d", "n_payments_365d", "n_payments_ever",
          "n_insurance_payments_365d", "n_adjustments_365d", "n_accounts", "n_hospital_accounts",
          "n_physician_accounts", "n_prior_contacts_90d", "n_prior_calls", "talk_seconds", "n_actions"]


# ---------------------------------------------------------------- data

def load_data(design: str = "call", holdout_frac: float = 0.2, world: str = DEFAULT_WORLD
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(dev, holdout, real). real: the usable real rows (reliable label, full 30-day window).

    design="call":  one synthetic row per call; holdout = latest 20% by date.
    design="fixed": one synthetic row per patient with an open balance on the world's cutoff date (2026-04-01);
                    all rows share one date, so the holdout is a stratified random 20% of patients."""
    if design == "fixed":
        syn = pd.read_csv(world_files(world)[1], parse_dates=["as_of"])
        syn = syn[syn["label_reliable"] & syn["window_complete"]].reset_index(drop=True)
        from sklearn.model_selection import train_test_split
        dev, holdout = (d.reset_index(drop=True) for d in train_test_split(
            syn, test_size=holdout_frac, stratify=syn[TARGET_PRE], random_state=SEED))
    else:
        syn = pd.read_csv(world_files(world)[0], parse_dates=["as_of"])
        syn = syn[syn["label_reliable"] & syn["window_complete"]].sort_values("as_of").reset_index(drop=True)
        cut = syn["as_of"].quantile(1 - holdout_frac)
        dev, holdout = syn[syn["as_of"] < cut].reset_index(drop=True), syn[syn["as_of"] >= cut].reset_index(drop=True)
    real = pd.read_excel(REAL_XLSX, sheet_name="Patient snapshots")
    real = real[real["label_reliable"] & real["window_complete"]].reset_index(drop=True)
    for df in (dev, holdout, real):
        bools = df.select_dtypes(include="bool").columns
        df[bools] = df[bools].astype(int)
    return dev, holdout, real


def post_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for the post-call (follow-through) model: no payment taken on the call."""
    return df[df["paid_on_call"] == 0].reset_index(drop=True)


def xy(df: pd.DataFrame, features: list[str], target: str = TARGET_PRE) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[features].apply(pd.to_numeric, errors="coerce").astype(float)
    return X, df[target].astype(int).to_numpy()


def oracle_traits(df: pd.DataFrame, world: str = DEFAULT_WORLD) -> pd.DataFrame:
    """The synthetic world's hidden traits per call (ability, willingness, hardship): never observable, used only
    for the 'oracle' baseline (what the hidden traits alone explain)."""
    truth = pd.read_csv(PHASE2_WORLDS[world] / "truth" / "truth_guarantors.csv",
                        usecols=["gid", "ability", "willingness", "hardship"])
    return df[["guarantor_id"]].merge(truth, left_on="guarantor_id", right_on="gid", how="left")[
        ["ability", "willingness", "hardship"]].astype(float)


# ---------------------------------------------------------------- preprocessing

class RecencyFill(BaseEstimator, TransformerMixin):
    """'Days since …' columns are empty when the event never happened. Fill with (train max + 1) — 'longer ago
    than anyone seen' — and add a 0/1 '<col>__never' flag, so models can tell 'never' from 'long ago'."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.cols_ = [c for c in (self.columns if self.columns is not None else RECENCY) if c in X.columns]
        self.fill_ = {c: (np.nanmax(X[c].to_numpy()) if X[c].notna().any() else 0.0) + 1 for c in self.cols_}
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            X[f"{c}__never"] = X[c].isna().astype(float)
            X[c] = X[c].fillna(self.fill_[c])
        return X.fillna(0.0)  # any other gap (rare) -> 0

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.feature_names_in_) + [f"{c}__never" for c in self.cols_])


class ZeroFlags(BaseEstimator, TransformerMixin):
    """For zero-heavy columns, add '<col>__any' (value > 0): separates 'none' from 'how much'."""

    def __init__(self, columns=None, min_zero_share: float = 0.5):
        self.columns, self.min_zero_share = columns, min_zero_share

    def fit(self, X, y=None):
        cand = self.columns if self.columns is not None else list(X.columns)
        self.cols_ = [c for c in cand if c in X.columns and X[c].nunique() > 2 and (X[c] == 0).mean() >= self.min_zero_share]
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            X[f"{c}__any"] = (X[c] > 0).astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.feature_names_in_) + [f"{c}__any" for c in self.cols_])


class WoE(BaseEstimator, TransformerMixin):
    """Weight-of-evidence binning (credit-scoring standard). Each feature -> quantile bins (missing = its own
    bin) -> ln(share of payers / share of non-payers) per bin, smoothed. monotonic=True merges neighbouring bins
    until WoE only rises or only falls. iv_ holds each feature's information value."""

    def __init__(self, n_bins: int = 6, monotonic: bool = False, smoothing: float = 0.5, binning: str = "smart",
                 min_bin_share: float = 0.03):
        self.n_bins, self.monotonic, self.smoothing = n_bins, monotonic, smoothing
        self.binning, self.min_bin_share = binning, min_bin_share

    def _edges(self, x):
        """Bin edges (a value v falls in bin = number of edges <= v).
        quantile: plain quantile cuts — collapses on zero-heavy or few-valued columns.
        smart:    few distinct values -> one bin per value; zero-heavy -> zero gets its own bin and quantiles are
                  cut over the non-zero values; then bins smaller than min_bin_share are merged into a neighbour."""
        ok = x[~np.isnan(x)]
        if not len(ok):
            return np.array([])
        if self.binning == "quantile":
            return np.unique(np.quantile(ok, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        vals = np.unique(ok)
        if len(vals) <= self.n_bins:
            edges = (vals[:-1] + vals[1:]) / 2
        else:
            zero_share = (ok == 0).mean()
            nz = ok[ok != 0]
            if zero_share >= 1 / self.n_bins and len(nz):
                q = np.quantile(nz, np.linspace(0, 1, self.n_bins)[1:-1])
                edges = np.unique(np.r_[np.min(nz) / 2 if np.min(nz) > 0 else 0, q])
            else:
                edges = np.unique(np.quantile(ok, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        while len(edges):
            counts = np.bincount(np.searchsorted(edges, ok, side="right"), minlength=len(edges) + 1) / len(ok)
            small = int(np.argmin(counts))
            if counts[small] >= self.min_bin_share:
                break
            edges = np.delete(edges, min(small, len(edges) - 1))  # merge the smallest bin into its neighbour
        return edges

    def _woe(self, x_bin, y, bins):
        pos, neg = y.sum(), len(y) - y.sum()
        out = {}
        for b in bins:
            m = x_bin == b
            p, n = y[m].sum() + self.smoothing, (m.sum() - y[m].sum()) + self.smoothing
            out[b] = (np.log((p / (pos + self.smoothing)) / (n / (neg + self.smoothing))),
                      (p / (pos + self.smoothing)) - (n / (neg + self.smoothing)))
        return out

    def fit(self, X, y):
        y = np.asarray(y)
        self.edges_, self.table_, self.iv_ = {}, {}, {}
        for c in X.columns:
            x = X[c].to_numpy(dtype=float)
            edges = self._edges(x)
            if self.monotonic:
                edges = self._monotone(x, y, edges)
            self.edges_[c] = edges
            xb = self._bin(x, edges)
            table = self._woe(xb, y, np.unique(xb))
            self.table_[c] = {b: w for b, (w, _) in table.items()}
            self.iv_[c] = float(sum(d * w for w, d in table.values()))
        self.feature_names_in_ = np.array(X.columns)
        return self

    def _monotone(self, x, y, edges):
        while len(edges) > 0:
            xb = self._bin(x, edges)
            ws = [self._woe(xb, y, [b])[b][0] for b in range(len(edges) + 1) if (xb == b).any()]
            d = np.diff(ws)
            if len(d) == 0 or (d >= 0).all() or (d <= 0).all():
                break
            edges = np.delete(edges, int(np.argmin(np.abs(d))))  # merge the two most similar neighbours
        return edges

    @staticmethod
    def _bin(x, edges):
        b = np.searchsorted(edges, x, side="right").astype(float)
        b[np.isnan(x)] = -1  # missing = own bin
        return b

    def transform(self, X):
        out = {}
        for c in self.feature_names_in_:
            xb = self._bin(X[c].to_numpy(dtype=float), self.edges_[c])
            out[c] = [self.table_[c].get(b, 0.0) for b in xb]  # unseen bin -> 0 (neutral)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class Quantile(QuantileTransformer):
    """QuantileTransformer with n_quantiles capped at the training size (no warning on small samples)."""

    def fit(self, X, y=None):
        self.n_quantiles = min(500, len(X))
        return super().fit(X, y)


def _log1p_cols(columns):
    def f(X):
        X = X.copy()
        for c in columns:
            if c in X.columns:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X
    return FunctionTransformer(f, feature_names_out="one-to-one")


PREPROCESSORS = {
    "none": "fill missing only (recency -> 'never' value + flag)",
    "z": "z-score",
    "log_z": "log1p on skewed money/count columns, then z-score",
    "log_robust": "log1p, then robust scaling (median / IQR)",
    "quantile": "quantile transform to a normal distribution",
    "woe": "weight-of-evidence bins",
    "woe_mono": "monotonic weight-of-evidence bins",
}


def preprocess(kind: str = "log_z", zero_flags: bool = False) -> list[tuple[str, object]]:
    """Pipeline steps for one preprocessing variant (see PREPROCESSORS)."""
    steps: list[tuple[str, object]] = []
    if kind in ("woe", "woe_mono"):
        return [("woe", WoE(monotonic=kind == "woe_mono"))]
    steps.append(("fill", RecencyFill()))
    if zero_flags:
        steps.append(("zero_flags", ZeroFlags()))
    if kind in ("log_z", "log_robust"):
        steps.append(("log1p", _log1p_cols(SKEWED + RECENCY)))
    scaler = {"none": None, "z": StandardScaler(), "log_z": StandardScaler(), "log_robust": RobustScaler(),
              "quantile": Quantile(output_distribution="normal", random_state=SEED)}[kind]
    if scaler is not None:
        steps.append(("scale", scaler))
    return steps


# ---------------------------------------------------------------- selection

def _auc_dir(x, y) -> float:
    try:
        a = roc_auc_score(y, x)
    except ValueError:
        return 0.5
    return max(a, 1 - a)


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop one of each pair with |Spearman rho| > threshold, keeping the one with the stronger single-feature AUC.
    Fitted per fold."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def fit(self, X, y):
        y = np.asarray(y)
        strength = {c: _auc_dir(X[c].to_numpy(), y) for c in X.columns}
        corr = X.rank().corr().abs().fillna(0).to_numpy()
        keep: list[str] = []
        for c in sorted(X.columns, key=lambda c: -strength[c]):
            i = list(X.columns).index(c)
            if all(corr[i, list(X.columns).index(k)] <= self.threshold for k in keep):
                keep.append(c)
        self.keep_ = [c for c in X.columns if c in keep]
        return self

    def transform(self, X):
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


class StabilitySelector(BaseEstimator, TransformerMixin):
    """L1 logistic on many half-samples; keep features chosen in >= threshold of them (Meinshausen & Buhlmann).
    Expects scaled input."""

    def __init__(self, n_boot: int = 100, C: float = 0.05, threshold: float = 0.7, random_state: int = SEED):
        self.n_boot, self.C, self.threshold, self.random_state = n_boot, C, threshold, random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        hits = np.zeros(X.shape[1])
        for _ in range(self.n_boot):
            idx = rng.choice(len(y), size=len(y) // 2, replace=False)
            m = LogisticRegression(l1_ratio=1, C=self.C, solver="liblinear").fit(X.iloc[idx], y[idx])
            hits += np.abs(m.coef_[0]) > 1e-8
        self.frequency_ = pd.Series(hits / self.n_boot, index=X.columns)
        self.keep_ = list(self.frequency_[self.frequency_ >= self.threshold].index) or [self.frequency_.idxmax()]
        return self

    def transform(self, X):
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


class IVFilter(BaseEstimator, TransformerMixin):
    """Keep WoE columns with information value above min_iv (0.02 = 'weak' in credit scoring). Place after WoE."""

    def __init__(self, min_iv: float = 0.02):
        self.min_iv = min_iv

    def fit(self, X, y):
        w = WoE().fit(X, y) if not hasattr(self, "_woe_step") else self._woe_step
        self.iv_ = pd.Series(w.iv_)
        self.keep_ = list(self.iv_[self.iv_ > self.min_iv].index) or [self.iv_.idxmax()]
        return self

    def transform(self, X):
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


class Columns(BaseEstimator, TransformerMixin):
    """Fixed column subset (for 'train only on these features' runs)."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[c for c in self.columns if c in X.columns]]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns)


def select_l1(C: float = 0.05) -> SelectFromModel:
    return SelectFromModel(LogisticRegression(l1_ratio=1, C=C, solver="liblinear"), threshold=1e-8)


def select_mi(k: int = 10) -> SelectKBest:
    return SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=SEED), k=k)


# ---------------------------------------------------------------- models

def logreg(C: float = np.inf, l1_ratio: float = 0.0, class_weight=None, max_iter: int = 5000) -> LogisticRegression:
    """C=inf: no penalty. l1_ratio 0 = L2 (ridge), 1 = L1 (lasso), between = elastic net."""
    solver = "lbfgs" if (l1_ratio == 0 or C == np.inf) else "saga"
    return LogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, class_weight=class_weight, max_iter=max_iter)


def pipeline(*steps) -> Pipeline:
    """Flatten preprocess()/selector/model pieces into one Pipeline that keeps pandas column names."""
    flat: list[tuple[str, object]] = []
    for s in steps:
        if isinstance(s, list):
            flat += s
        elif isinstance(s, tuple):
            flat.append(s)
        else:
            flat.append((type(s).__name__.lower(), s))
    p = Pipeline(flat)
    p.set_output(transform="pandas")
    return p


# ---------------------------------------------------------------- metrics

def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Slope: logistic regression of y on logit(p) (1 = well spread). Intercept: calibration-in-the-large,
    mean predicted minus observed rate (0 = right overall level)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        slope = float(LogisticRegression(C=np.inf).fit(lp, y).coef_[0][0])
    except ValueError:
        slope = np.nan
    return slope, float(p.mean() - np.mean(y))


def top_capture(y, p, frac: float = 0.2) -> float:
    """Share of all payers found in the top `frac` of scores."""
    y = np.asarray(y)
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-np.asarray(p))[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    slope, citl = calibration_slope_intercept(y, p)
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1]), "cal_slope": slope, "cal_in_large": citl,
            "top20_capture": top_capture(y, p)}


def bootstrap_ci(y, p, metric=roc_auc_score, n: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """(estimate, 2.5%, 97.5%) by resampling rows. For the small real test set."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(metric(y[i], p[i]))
    return float(metric(y, p)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- cross-validation + logging

@dataclass
class Split:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def cv_splits(df: pd.DataFrame, target: str = TARGET_PRE, k: int = 5, repeats: int = 5, grouped: bool = True,
              seed: int = SEED) -> list[Split]:
    """Repeated stratified K-fold, grouped by patient unless grouped=False (only to demonstrate leakage)."""
    y = df[target].astype(int).to_numpy()
    groups = df["guarantor_id"].fillna(df["snapshot_id"]).to_numpy() if grouped else np.arange(len(df))
    out = []
    for r in range(repeats):
        skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
            out.append(Split(r, f, tr, te))
    return out


def features_used(fitted, columns: list[str] | None = None) -> list[str]:
    """Columns the final model actually uses: after any selection step, and, for linear models, only those with a
    non-zero weight (so L1 shows how many features it really kept). Unwraps a tuned search to its best pipeline."""
    fitted = getattr(fitted, "best_estimator_", fitted)
    if not isinstance(fitted, Pipeline):
        return [getattr(fitted, "feature_")] if hasattr(fitted, "feature_") else list(columns or [])
    if len(fitted.steps) == 1:
        names = list(getattr(fitted[-1], "feature_names_in_", []))
    else:
        try:
            names = list(fitted[:-1].get_feature_names_out())
        except Exception:  # noqa: BLE001 - some steps can't report names
            names = list(getattr(fitted[-1], "feature_names_in_", []))
    coef = getattr(fitted[-1], "coef_", None)
    if coef is not None and np.ndim(coef) == 2 and coef.shape[1] == len(names):
        names = [n for n, w in zip(names, coef[0]) if abs(w) > 1e-8]
    return names


def run_experiment(run_id: str, pipe, df: pd.DataFrame, features: list[str] | str, *, notebook: str,
                   description: str, target: str = TARGET_PRE, splits: list[Split] | None = None,
                   log: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Fit `pipe` on each training fold, score the test fold, log one row per fold to results.csv
    (rerunning a run_id replaces its rows). Returns the fold table."""
    fs_name = features if isinstance(features, str) else "custom"
    cols = FEATURE_SETS[features] if isinstance(features, str) else list(features)
    X, y = xy(df, cols, target)
    splits = splits if splits is not None else cv_splits(df, target)
    rows = []
    for s in splits:
        t0 = time.perf_counter()
        m = clone(pipe).fit(X.iloc[s.train], y[s.train])
        p = m.predict_proba(X.iloc[s.test])[:, 1]
        used = features_used(m, cols)
        rows.append({"run_id": run_id, "notebook": notebook, "description": description, "target": target,
                     "feature_set": fs_name, "n_features_in": len(cols), "n_features_used": len(used),
                     "features_used": ";".join(map(str, used)), "repeat": s.repeat, "fold": s.fold,
                     "n_train": len(s.train), "n_test": len(s.test), **metrics(y[s.test], p),
                     "fit_seconds": round(time.perf_counter() - t0, 3),
                     "logged_at": datetime.now().isoformat(timespec="seconds")})
    folds = pd.DataFrame(rows)
    if log:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        old = read_results(RESULTS) if RESULTS.exists() else pd.DataFrame()
        if len(old):
            old = old[old["run_id"] != run_id]
        pd.concat([old, folds], ignore_index=True).to_csv(RESULTS, index=False)
    if verbose:
        print(f"{run_id:8s} AUC {folds.auc.mean():.3f} ± {folds.auc.std():.3f} | Brier {folds.brier.mean():.3f} | "
              f"features {folds.n_features_used.mean():.0f} | {description}")
    return folds


def read_results(path=RESULTS) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"notebook": str, "run_id": str})


def leaderboard(path=RESULTS, notebook: str | None = None) -> pd.DataFrame:
    r = read_results(path)
    if notebook:
        r = r[r["notebook"].astype(str) == str(notebook)]
    g = r.groupby(["run_id", "description", "feature_set", "target"], sort=False)
    out = g.agg(auc=("auc", "mean"), auc_sd=("auc", "std"), folds=("auc", "size"), pr_auc=("pr_auc", "mean"),
                brier=("brier", "mean"), log_loss=("log_loss", "mean"), cal_slope=("cal_slope", "mean"),
                top20=("top20_capture", "mean"), n_features=("n_features_used", "mean"),
                fit_s=("fit_seconds", "mean")).reset_index()
    out["auc_se"] = out["auc_sd"] / np.sqrt(out["folds"])
    return out.sort_values("auc", ascending=False).reset_index(drop=True)


def corrected_ttest(run_a: str, run_b: str, path=RESULTS, metric: str = "auc") -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired folds: the variance is inflated by
    n_test/n_train because CV training sets overlap. Returns mean difference (a - b) and the p-value."""
    r = read_results(path)
    a = r[r.run_id == run_a].set_index(["repeat", "fold"])
    b = r[r.run_id == run_b].set_index(["repeat", "fold"])
    d = (a[metric] - b[metric]).dropna()
    J = len(d)
    ratio = (a["n_test"] / a["n_train"]).mean()
    var = (1 / J + ratio) * d.var(ddof=1)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return {"a": run_a, "b": run_b, "mean_diff": float(d.mean()), "t": float(t), "p_value": float(p), "folds": J}


def one_se_pick(board: pd.DataFrame, complexity: str = "n_features") -> pd.Series:
    """Best AUC minus one standard error; among runs above that line, the simplest (fewest features), then the
    best Brier score."""
    best = board.loc[board["auc"].idxmax()]
    ok = board[board["auc"] >= best["auc"] - best["auc_se"]]
    return ok.sort_values([complexity, "brier"]).iloc[0]


# ---------------------------------------------------------------- tuning and importance-based selection

def tuned(pipe, params: dict, n_iter: int = 15, inner_k: int = 3, seed: int = SEED):
    """Nested CV: wrap a pipeline in a randomized search over `params` (keys are '<step>__<param>'), scored by AUC
    on an inner stratified K-fold of the *training fold only*. run_experiment then scores it on the outer folds.
    (Inner folds aren't grouped: fine for the fixed-date design, one row per patient.)"""
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    return RandomizedSearchCV(pipe, params, n_iter=n_iter, scoring="roc_auc", refit=True, random_state=seed,
                              cv=StratifiedKFold(inner_k, shuffle=True, random_state=seed), n_jobs=-1)


class TopKPermutation(BaseEstimator, TransformerMixin):
    """Keep the k features whose shuffling hurts a quick random forest most. Importance is measured on a 25% slice
    of the training fold the forest didn't see (never on the test fold)."""

    def __init__(self, k: int = 6, random_state: int = SEED):
        self.k, self.random_state = k, random_state

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split

        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, stratify=y, random_state=self.random_state)
        rf = RandomForestClassifier(200, min_samples_leaf=5, random_state=self.random_state, n_jobs=-1).fit(Xa, ya)
        imp = permutation_importance(rf, Xb, yb, scoring="roc_auc", n_repeats=5, random_state=self.random_state)
        self.importance_ = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
        self.keep_ = list(self.importance_.index[: self.k])
        return self

    def transform(self, X):
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


# Domain direction of effect (benchmarking doc + collections practice): +1 more -> likelier to pay, -1 less.
MONOTONE = {
    "n_payments_90d": 1, "n_payments_365d": 1, "amount_paid_90d": 1, "amount_paid_365d": 1, "ever_paid": 1,
    "n_payments_ever": 1, "amount_paid_ever": 1,
    "days_since_last_payment": -1, "balance_at_t": -1, "placement_balance": -1, "days_since_first_placement": -1,
}


def monotone_for(columns) -> dict:
    """{column: direction} for the columns a model will see (others unconstrained)."""
    return {c: MONOTONE[c] for c in columns if c in MONOTONE}



def filled_columns(features: list[str]) -> list[str]:
    """Column order after RecencyFill: the features, then a '__never' flag per recency column present."""
    return list(features) + [f"{c}__never" for c in RECENCY if c in features]


def lgbm(features: list[str] | None = None, monotone: bool = False, **params):
    """LightGBM classifier (after RecencyFill). monotone=True constrains each column per MONOTONE, which needs the
    feature list to know the column order."""
    import lightgbm as lgb

    if monotone:
        params["monotone_constraints"] = [MONOTONE.get(c, 0) for c in filled_columns(features)]
        params["monotone_constraints_method"] = "advanced"
    return lgb.LGBMClassifier(verbose=-1, random_state=SEED, n_jobs=1, **params)
