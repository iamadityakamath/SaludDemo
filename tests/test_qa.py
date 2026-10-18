import pandas as pd
import pytest

from salud_pipeline.qa.evaluate import agreement_table, cohen_kappa, consensus, inter_rater
from salud_pipeline.qa.targets import TARGETS, load_synthetic
from salud_pipeline.synth.cli import main as synth_main


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    synth_main(["--out", str(out), "--n-guarantors", "250", "--seed", "5"])
    return load_synthetic(out)


def test_load_synthetic_has_every_target(synthetic):
    assert set(TARGETS) <= set(synthetic.columns)
    assert synthetic[TARGETS].notna().all().all()
    assert not synthetic["text"].str.contains("Editorial note").any()


def test_cohen_kappa_matches_hand_calculation():
    # 10 pairs: po = 0.8; a has 6 Yes/4 No, b has 6 Yes/4 No -> pe = .36 + .16 = .52 -> kappa = .28/.48
    a = ["Yes"] * 5 + ["No"] * 3 + ["Yes", "No"]
    b = ["Yes"] * 5 + ["No"] * 3 + ["No", "Yes"]
    assert cohen_kappa(a, b) == pytest.approx(0.28 / 0.48)
    assert cohen_kappa(["Yes", "No"], ["Yes", "No"]) == pytest.approx(1.0)


def test_agreement_counts_na_as_an_answer_and_missing_as_wrong():
    truth = pd.DataFrame({"call_id": ["a", "b", "c", "d"], "greeting": ["Yes", "N/A", "No", ""]})
    pred = pd.DataFrame({"call_id": ["a", "b", "c", "d"], "greeting": ["Yes", "Yes", None, "Yes"]})
    row = agreement_table(truth, pred, targets=["greeting"]).iloc[0]
    assert row["n"] == 3                      # blank human tag excluded
    assert row["agreement"] == pytest.approx(1 / 3)
    assert row["pred_missing"] == 1
    assert row["ci_low"] <= row["agreement"] <= row["ci_high"]


def test_inter_rater_and_consensus():
    gt = pd.DataFrame({"call_id": ["a", "a", "b", "b"], "rater": ["X", "Y", "X", "Y"],
                       "greeting": ["Yes", "Yes", "No", "Yes"]})
    irr = inter_rater(gt, targets=["greeting"]).iloc[0]
    assert irr["agreement"] == 0.5
    cons = consensus(gt, targets=["greeting"]).set_index("call_id")
    assert cons.loc["a", "greeting"] == "Yes"
    assert pd.isna(cons.loc["b", "greeting"])  # raters split: no consensus
