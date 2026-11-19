"""The snapshot builder on a tiny hand-made world: point-in-time features and 30-day labels."""

import pandas as pd
import pytest

from salud_pipeline.phase2.propensity import dictionary as D
from salud_pipeline.phase2.propensity.snapshots import build, self_pay
from salud_pipeline.phase2.propensity.sources import World

T = pd.Timestamp("2026-07-10 14:00")


def _world(tx_rows, n_accounts=1, exclude_col=False):
    calls = pd.DataFrame([{
        "call_id": "C1", "guarantor_id": "G1", "t": T, "connection": "Outbound Call, Manual",
        "contact_type": "Patient", "disposition": "Spoke with CSR", "collection_attempt": "Yes",
        "collection_result": "Promised to Pay", "actions": "Verbally Shared Balance(s); Set Up Payment Plan",
        "payment_on_call": float("nan"), "review_balance": "Yes", "asked_fa": "No", "requested_callback": "No",
        "transfer_needed": "No", "language": "English", "talk_seconds": 300.0, "agent": "[Agent 001]",
        "tcn_record_id": "R1"}])
    accounts = pd.DataFrame([{"account_id": f"A{i}", "guarantor_id": "G1", "hbpb": "Hospital Account",
                              "placement_date": pd.Timestamp("2026-01-01"), "discharge_date": pd.Timestamp("2025-12-20"),
                              "placement_balance": 100.0} for i in range(n_accounts)]
                            + [{"account_id": "LATE", "guarantor_id": "G1", "hbpb": "Physician Account",
                                "placement_date": pd.Timestamp("2026-08-01"), "discharge_date": pd.NaT,
                                "placement_balance": 999.0}])
    links = pd.DataFrame({"account_id": [f"A{i}" for i in range(n_accounts)], "call_id": "C1"})
    tx = pd.DataFrame([{"tx_id": f"T{i}", "account_id": "A0", "guarantor_id": "G1", "post_date": pd.Timestamp(d),
                        "amount": a, "txn_type": tt, "payment_type": pt, "exclude": ex if exclude_col else None}
                       for i, (d, a, tt, pt, ex) in enumerate(tx_rows)],
                      columns=["tx_id", "account_id", "guarantor_id", "post_date", "amount", "txn_type",
                               "payment_type", "exclude"])
    return World("test", calls, calls.copy(), accounts, links, tx, pd.Timestamp("2026-11-02"), {})


PAY = ("Payment", "Patient Payment", "Include")


def test_payment_the_day_before_is_history_and_the_day_after_is_the_outcome():
    row = build(_world([("2026-07-09", 40.0, *PAY), ("2026-07-11", 25.0, *PAY)])).iloc[0]
    assert row["n_payments_ever"] == 1 and row["amount_paid_ever"] == 40.0 and row["ever_paid"]
    assert row["days_since_last_payment"] == 1
    assert row["paid_30d"] and row["paid_after_call_30d"] and row["paid_amount_30d"] == 25.0
    assert row["days_to_first_payment"] == 1


def test_payment_on_the_call_day_counts_for_pre_call_label_only():
    row = build(_world([("2026-07-10", 30.0, *PAY)])).iloc[0]
    assert row["n_payments_ever"] == 0
    assert row["paid_30d"] and not row["paid_after_call_30d"]


def test_window_ends_at_30_days():
    assert not build(_world([("2026-08-10", 30.0, *PAY)])).iloc[0]["paid_30d"]   # day 31
    assert build(_world([("2026-08-09", 30.0, *PAY)])).iloc[0]["paid_30d"]       # day 30


def test_nothing_after_t_reaches_the_features():
    row = build(_world([("2026-07-20", 500.0, *PAY), ("2026-07-20", 80.0, "Payment", "Insurance Payment", "Include")])).iloc[0]
    assert row["n_payments_ever"] == 0 and row["n_insurance_payments_365d"] == 0
    assert row["n_accounts"] == 1                       # the account placed in August is not there yet
    assert row["placement_balance"] == 100.0 and row["balance_at_t"] == 100.0


def test_balance_at_t_subtracts_earlier_patient_payments():
    row = build(_world([("2026-03-01", 30.0, *PAY)])).iloc[0]
    assert row["balance_at_t"] == 70.0


def test_many_accounts_give_one_row():
    df = build(_world([("2026-07-11", 25.0, *PAY)], n_accounts=39))
    assert len(df) == 1 and df.iloc[0]["n_accounts"] == 39 and df.iloc[0]["n_linked_accounts"] == 39


def test_exclude_flag_is_respected_when_the_column_exists():
    tx = pd.DataFrame({"txn_type": ["Payment"] * 3, "payment_type": ["Patient Payment"] * 3,
                       "exclude": ["Include", "Exclude", None]})
    assert self_pay(tx).tolist() == [True, False, False]
    tx["exclude"] = None                                 # column not provided at all (real package)
    assert self_pay(tx).all()


def test_call_fields_become_flags():
    row = build(_world([])).iloc[0]
    assert row["promised_to_pay"] and row["action_payment_plan"] and row["action_shared_balance"]
    assert row["n_actions"] == 2 and not row["paid_on_call"] and row["label_reliable"]


def test_feature_lists_hold_no_leaks_labels_or_audit_columns():
    forbidden = set(D.names(("label", "id", "audit"))) | {"payment_on_call", "collection_result"}
    assert not forbidden & set(D.PRE_FEATURES) and not forbidden & set(D.POST_FEATURES)
    assert not set(D.LEAKING_SNAPSHOT_FIELDS) & set(D.POST_FEATURES)
    assert "preferred_language" not in D.POST_FEATURES
    assert set(D.PRE_FEATURES) < set(D.POST_FEATURES)


def test_synthetic_codes_and_real_labels_match():
    from salud_pipeline.phase2.propensity.sources import HANDOFF_JSON, _decode, option_labels
    if not HANDOFF_JSON.exists():
        pytest.skip("schema handoff not on this machine")
    codes = option_labels()[("salud_gphpvcontacthistory", "salud_collectionresult")]
    out = _decode(pd.Series(["785570003", None]), codes)
    assert out[0] == "Promised to Pay" and pd.isna(out[1])
    acts = option_labels()[("salud_gphpvcontacthistory", "salud_actionstaken")]
    assert _decode(pd.Series(["785570000;785570006"]), acts, multi=True)[0] == \
        "Verbally Shared Balance(s); Set Up Payment Plan"


def test_score_qa_writes_one_row_per_call_and_resumes_from_cache(tmp_path, monkeypatch):
    """Real-call QA: a fake reviewer stands in for the LLM; the second run must use only the cache."""
    from types import SimpleNamespace

    import salud_pipeline.llm_client as llm_client
    import salud_pipeline.paths as paths
    from salud_pipeline.phase1 import prompt
    from salud_pipeline.phase1.checklist import TRANSCRIPT_ITEMS
    from salud_pipeline.phase2.propensity.snapshots import score_qa

    calls = {"n": 0}

    class FakeEndpoint:
        model, provider, max_retries = "fake-model", "fake", 0

        def chat(self, *a, **k):
            calls["n"] += 1
            answers = {k: {"answer": "Yes" if i % 2 == 0 else "No", "reason": "r",
                           "evidence": [{"turn": 1, "quote": "Hello this is billing"}]}
                       for i, k in enumerate(TRANSCRIPT_ITEMS)}
            import json
            return json.dumps({"checklist": answers, "sentiment": {"label": "neutral", "evidence": []},
                               "outcome": {"label": "non_committal", "evidence": []}, "summary": "s", "coaching": []})

    monkeypatch.setattr(llm_client, "LLMEndpoint", FakeEndpoint)
    monkeypatch.setattr(paths, "LLM_CACHE", tmp_path / "cache")
    world = SimpleNamespace(transcripts={"c1": "Agent: Hello this is billing.\n\nCaller: Hi.",
                                         "c2": "Agent: Hello this is billing again.\n\nCaller: Hello."})
    out = tmp_path / "qa.csv"
    first = score_qa(world, out)
    assert len(first) == 2 and calls["n"] == 2 and out.exists()
    yes = sum(i % 2 == 0 for i in range(len(TRANSCRIPT_ITEMS)))
    assert first["qa_score"].round(3).eq(round(yes / len(TRANSCRIPT_ITEMS), 3)).all()
    assert set(first["prompt_version"]) == {prompt.PROMPT_VERSION}
    second = score_qa(world, out)
    assert calls["n"] == 2 and len(second) == 2                       # served from the cache, no new requests


def test_master_split_sampler_and_groups():
    import numpy as np
    import pandas as pd
    from salud_pipeline.phase2.propensity.master import SCORES, fit_sampler, outcome_group, patient_split, sample_scores

    g = pd.Series([f"g{i % 50}" for i in range(400)])
    split = patient_split(g)
    assert not (set(g[split == "train"]) & set(g[split == "test"]))          # no patient on both sides
    assert 0.15 < (split == "test").mean() < 0.25

    rng = np.random.default_rng(0)
    n = 120
    real = pd.DataFrame({"spoke_with_agent": rng.random(n) < 0.8, "paid_on_call": rng.random(n) < 0.2,
                         **{c: rng.random(n) < 0.15 for c in ["promised_to_pay", "non_committal", "transferred",
                                                              "contested_or_refused", "pursued_insurance"]}})
    for c in SCORES:
        real[c] = rng.uniform(-0.5, 0.5, n) + 0.3 * real["paid_on_call"]
    grp = outcome_group(real)
    assert (grp[~real["spoke_with_agent"]] == "no_agent").all()
    assert (grp[real["spoke_with_agent"] & real["paid_on_call"]] == "paid").all()
    s = fit_sampler(real)
    out = sample_scores(pd.concat([real] * 20, ignore_index=True), s)
    for c in SCORES:   # inside the real range, and paid calls score higher, as in the real data
        assert out[c].min() >= real[c].min() - 1e-4 and out[c].max() <= real[c].max() + 1e-4   # 4-decimal rounding
    assert set(out["qa_score"].round(4)) <= set(real["qa_score"].round(4))   # QA only takes real values
    big = pd.concat([real] * 20, ignore_index=True)
    assert out["qa_score"][outcome_group(big) == "paid"].mean() > out["qa_score"][outcome_group(big) != "paid"].mean()


def test_insurance_flag_ignores_coinsurance():
    from salud_pipeline.phase2.propensity.snapshots import _flags

    assert not _flags("Verbally Shared Balance(s); Explained Coinsurance/Deductible/Copay")["action_insurance"]
    assert _flags("Requested FA Paperwork; Added Insurance Information")["action_insurance"]
    assert _flags("Requested Insurance Review")["action_insurance"]
