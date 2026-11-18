"""The shared modelling machinery: no leakage, grouped folds, reproducibility, WoE, pruning, feature sets."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from salud_pipeline.phase2.propensity import modeling as M  # noqa: E402


def _toy(n=600, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n // 3, n)                       # ~3 rows per patient
    x1 = rng.normal(size=n)
    df = pd.DataFrame({
        "guarantor_id": [f"G{i}" for i in g], "snapshot_id": [f"S{i}" for i in range(n)],
        "balance_at_t": rng.lognormal(6, 1.5, n), "n_payments_365d": rng.poisson(1, n).astype(float),
        "days_since_last_payment": np.where(rng.random(n) < 0.6, np.nan, rng.integers(1, 300, n)),
        "x1": x1, "x1_copy": x1 * 2 + rng.normal(scale=0.01, size=n),
    })
    logit = 1.5 * x1 if signal else 0
    df["paid_30d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


FEATS = ["balance_at_t", "n_payments_365d", "days_since_last_payment", "x1", "x1_copy"]


def test_shuffled_labels_give_chance_auc():
    """If scaling or selection leaked from the test fold, a model could beat chance on random labels."""
    df = _toy(signal=False)
    pipe = M.pipeline(M.preprocess("log_z"), M.CorrelationPruner(), M.select_l1(C=1.0), M.logreg(C=1.0))
    folds = M.run_experiment("T-SHUF", pipe, df, FEATS, notebook="test", description="shuffled", log=False,
                             verbose=False, splits=M.cv_splits(df, k=5, repeats=2))
    assert abs(folds.auc.mean() - 0.5) < 0.05


def test_real_signal_is_found():
    df = _toy()
    folds = M.run_experiment("T-SIG", M.pipeline(M.preprocess("log_z"), M.logreg(C=1.0)), df, FEATS,
                             notebook="test", description="signal", log=False, verbose=False,
                             splits=M.cv_splits(df, k=5, repeats=1))
    assert folds.auc.mean() > 0.7


def test_patients_never_cross_folds():
    df = _toy()
    for s in M.cv_splits(df, k=5, repeats=3):
        assert not set(df.guarantor_id.iloc[s.train]) & set(df.guarantor_id.iloc[s.test])


def test_ungrouped_splits_do_mix_patients():
    df = _toy()
    mixed = [bool(set(df.guarantor_id.iloc[s.train]) & set(df.guarantor_id.iloc[s.test]))
             for s in M.cv_splits(df, k=5, repeats=1, grouped=False)]
    assert all(mixed)


def test_same_seed_same_results():
    df = _toy()
    run = lambda: M.run_experiment("T-REP", M.pipeline(M.preprocess("quantile"), M.logreg(C=0.5)), df, FEATS,  # noqa: E731
                                   notebook="test", description="rep", log=False, verbose=False)
    pd.testing.assert_frame_equal(run().drop(columns=["fit_seconds", "logged_at"]),
                                  run().drop(columns=["fit_seconds", "logged_at"]))


def test_recency_fill_flags_never():
    X = pd.DataFrame({"days_since_last_payment": [10.0, np.nan, 30.0], "balance_at_t": [1.0, 2.0, 3.0]})
    out = M.RecencyFill().fit(X).transform(X)
    assert out["days_since_last_payment"].tolist() == [10.0, 31.0, 30.0]
    assert out["days_since_last_payment__never"].tolist() == [0.0, 1.0, 0.0]


def test_woe_handles_missing_and_unseen_values():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": np.r_[rng.normal(size=200), [np.nan] * 20]})
    y = (np.r_[X.a.iloc[:200] > 0, [True] * 20]).astype(int)
    w = M.WoE(n_bins=4).fit(X, y)
    out = w.transform(pd.DataFrame({"a": [-5.0, 0.1, 5.0, np.nan]}))
    assert out.a.notna().all() and out.a.iloc[0] < out.a.iloc[2]   # low values -> fewer payers
    assert w.iv_["a"] > 0.5


def test_monotonic_woe_is_monotonic():
    rng = np.random.default_rng(2)
    x = rng.normal(size=2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-(x + 0.8 * np.sin(4 * x))))).astype(int)
    w = M.WoE(n_bins=10, monotonic=True).fit(pd.DataFrame({"x": x}), y)
    vals = [w.table_["x"][b] for b in sorted(k for k in w.table_["x"] if k >= 0)]
    d = np.diff(vals)
    assert (d >= 0).all() or (d <= 0).all()


def test_correlation_pruner_keeps_one_per_cluster():
    df = _toy()
    X, y = M.xy(df, FEATS)
    keep = M.CorrelationPruner(0.9).fit(X.fillna(0), y).keep_
    assert ("x1" in keep) != ("x1_copy" in keep)


def test_feature_sets_are_consistent():
    assert set(M.FS_TRANSFER) < set(M.FS_PRUNED) < set(M.FS_ALL)
    assert not set(M.NO_TRANSFER) & set(M.FS_TRANSFER)
    assert not set(M.DROPPED_DUPLICATES) & set(M.FS_PRUNED)
    assert "preferred_language" not in M.FEATURE_SETS["FS_TRANSFER+POST"]
    assert "payment_on_call" not in M.FEATURE_SETS["FS_TRANSFER+POST"]


def test_corrected_ttest_is_more_conservative(tmp_path):
    rows = []
    rng = np.random.default_rng(3)
    for r in range(5):
        for f in range(5):
            base = 0.70 + rng.normal(scale=0.01)
            rows += [{"run_id": "A", "repeat": r, "fold": f, "auc": base + 0.005, "n_train": 800, "n_test": 200},
                     {"run_id": "B", "repeat": r, "fold": f, "auc": base, "n_train": 800, "n_test": 200}]
    path = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    res = M.corrected_ttest("A", "B", path=path)
    assert res["mean_diff"] == pytest.approx(0.005)
    assert res["folds"] == 25


def test_smart_woe_keeps_zero_heavy_and_few_valued_columns():
    """Plain quantile bins collapse a 95%-zero column into one bin; smart binning gives zero its own bin."""
    rng = np.random.default_rng(4)
    n = 2000
    x = np.where(rng.random(n) < 0.95, 0.0, rng.integers(1, 5, n)).astype(float)
    y = (rng.random(n) < np.where(x > 0, 0.7, 0.3)).astype(int)
    X = pd.DataFrame({"x": x, "count": rng.integers(0, 4, n).astype(float)})
    assert len(M.WoE(binning="quantile").fit(X, y).table_["x"]) == 1
    smart = M.WoE().fit(X, y)
    assert len(smart.table_["x"]) == 2 and smart.iv_["x"] > 0.05
    assert len(smart.table_["count"]) == 4          # one bin per distinct value


def test_dictionary_model_fields_match_the_models():
    from salud_pipeline.phase2.propensity import dictionary as D

    assert D.MODEL_PRE == M.FS_TRANSFER
    names = {c.name for c in D.COLUMNS}
    assert set(D.MODEL_PRE + D.MODEL_POST_EXTRA) <= names
    f = D.as_frame().set_index("name")["in_current_models"]
    assert (f == "pre-call + post-call").sum() == 11 and (f == "post-call only").sum() == 21
