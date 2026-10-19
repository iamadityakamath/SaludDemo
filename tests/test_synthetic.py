import numpy as np
import pandas as pd
import pytest

from salud_pipeline.phase2 import schema as S
from salud_pipeline.phase2.synthetic.config import GeneratorConfig
from salud_pipeline.phase2.synthetic.transcripts import render_template
from salud_pipeline.phase2.synthetic.validate import validate
from salud_pipeline.phase2.synthetic.world import generate


@pytest.fixture(scope="module")
def small():
    cfg = GeneratorConfig(n_guarantors=300, n_agents=12, seed=11)
    return cfg, generate(cfg)


def test_all_validation_checks_pass(small):
    cfg, t = small
    failed = [c for c in validate(t, pd.Timestamp(cfg.prediction_cutoff), cfg.outcome_horizon_days) if not c.ok]
    assert not failed, failed


def test_same_seed_is_reproducible(small):
    cfg, t = small
    again = generate(GeneratorConfig(n_guarantors=300, n_agents=12, seed=11))
    for name in (S.ACCOUNTS, S.RECEIPTS, S.CONTACTS):
        pd.testing.assert_frame_equal(t[name], again[name])


def test_crm_tables_have_exact_schema_columns(small):
    _, t = small
    for name, cols in S.COLUMNS.items():
        assert list(t[name].columns) == cols


def test_compliance_failure_discloses_balance_before_verification(small):
    _, t = small
    calls = t["truth_calls"]
    rng = np.random.default_rng(0)
    for call in calls[(~calls["qa_compliance"]) & (calls["scenario"] != "third_party")
                      & (~calls["hung_up"])].head(10).to_dict("records"):
        text = render_template(call, rng, noise=0.0)
        assert text.index("balance") < text.index("date of birth")


def _recover_coefficients(acc):
    """Least-squares fit of the planted logit on the generator's standardized features."""
    acc = acc[acc["open_at_cutoff"]].copy()
    names = ["ability", "willingness", "prior_payer", "balance", "aging", "call_quality",
             "promised_to_pay", "hardship_fa", "on_plan"]
    X = acc[[f"feat_{n}" for n in names]].astype(float)
    for c in ("ability", "willingness", "balance", "aging", "call_quality"):
        col = X[f"feat_{c}"]
        X[f"feat_{c}"] = ((col - col.mean()) / col.std()).fillna(0.0)
    X.insert(0, "intercept", 1.0)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), acc["planted_logit"].to_numpy(), rcond=None)
    return dict(zip(["intercept", *names], beta))


@pytest.mark.parametrize("effect", [0.5, 0.0])
def test_planted_call_quality_effect_is_recoverable(effect):
    cfg = GeneratorConfig(n_guarantors=300, n_agents=12, seed=3)
    cfg.outcome.call_quality = effect
    coefs = _recover_coefficients(generate(cfg)["truth_accounts"])
    assert coefs["call_quality"] == pytest.approx(effect, abs=1e-6)
    assert coefs["ability"] == pytest.approx(cfg.outcome.ability, abs=1e-6)
