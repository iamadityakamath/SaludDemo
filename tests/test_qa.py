import json

import pandas as pd
import pytest

from salud_pipeline.qa.evaluate import consensus, inter_rater, score_table
from salud_pipeline.qa.llm_scorer import parse_response
from salud_pipeline.qa.ml_scorer import MLScorer
from salud_pipeline.qa.targets import TARGETS, TRANSCRIPT_ITEMS, load_synthetic
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


def test_ml_scorer_beats_majority_on_synthetic(synthetic):
    train, test = synthetic.iloc[:300], synthetic.iloc[300:]
    st = score_table(test, MLScorer().fit(train).predict(test)).set_index("target")
    assert (st.loc["compliance", "accuracy"] > st.loc["compliance", "majority_baseline"])


def test_parse_response_normalizes_and_rejects_bad_values():
    raw = "Sure! " + json.dumps({"greeting": {"answer": "yes", "evidence": "This is X"},
                                 "compliance": "NO", "culture": {"answer": "maybe"},
                                 "sentiment": "Positive", "outcome": "teleported"})
    r = parse_response(raw)
    assert r["greeting"] == "Yes" and r["greeting_evidence"] == "This is X"
    assert r["compliance"] == "No"
    assert r["culture"] is None and r["outcome"] is None
    assert r["sentiment"] == "positive"
    assert parse_response("not json")["_raw_ok"] is False


def test_score_table_skips_na_and_counts_missing_as_wrong():
    truth = pd.DataFrame({"contact_id": ["a", "b", "c"], "greeting": ["Yes", "N/A", "No"]})
    pred = pd.DataFrame({"contact_id": ["a", "b", "c"], "greeting": ["Yes", "No", None]})
    row = score_table(truth, pred, targets=["greeting"]).iloc[0]
    assert row["n"] == 2 and row["accuracy"] == 0.5 and row["coverage"] == 0.5


def test_inter_rater_and_consensus():
    gt = pd.DataFrame({"contact_id": ["a", "a", "b", "b"], "rater": ["X", "Y", "X", "Y"],
                       "text": ["t", "t", "u", "u"], "greeting": ["Yes", "Yes", "No", "Yes"]})
    irr = inter_rater(gt, targets=["greeting"]).iloc[0]
    assert irr["agreement"] == 0.5
    assert consensus(gt, targets=["greeting"]).set_index("contact_id").loc["a", "greeting"] == "Yes"
