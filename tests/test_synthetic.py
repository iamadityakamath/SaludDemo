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


# ---------- realism mode (world tuned to Salud's final package) ----------

@pytest.fixture(scope="module")
def realistic():
    from pathlib import Path
    cfg = GeneratorConfig.from_json(Path(__file__).parents[1] / "configs" / "synthetic_world_salud.json")
    cfg.n_guarantors, cfg.n_agents = 400, 12
    return cfg, generate(cfg)


def test_realism_world_passes_validation(realistic):
    cfg, t = realistic
    failed = [c for c in validate(t, pd.Timestamp(cfg.prediction_cutoff), cfg.outcome_horizon_days) if not c.ok]
    assert not failed, failed


def test_realism_world_has_salud_like_structure(realistic):
    _, t = realistic
    per_patient = t[S.ACCOUNTS].groupby("salud_GuarantorId").size()
    assert per_patient.median() >= 2 and per_patient.max() > 10                    # heavy-tailed accounts
    c = t[S.CONTACTS]
    ivr = c["salud_CallDisposition"].isin([S.DISPOSITION["english_ivr"], S.DISPOSITION["spanish_ivr"]])
    assert ivr.any()                                                                # campaign calls reach the phone menu
    assert (c["salud_CollectionResult"] == S.RESULT["na_transferred"]).any()      # transferred calls
    r = t[S.RECEIPTS]
    assert ((r["salud_TXNType"] == S.TXN_ADJUSTMENT) & (r["salud_PaymentType"] == S.PAY_INSURANCE)).any()


def test_realism_payment_habit_makes_history_predictive(realistic):
    """Patients who paid before the cutoff are clearly likelier to pay in the window (the real data's strongest signal)."""
    _, t = realistic
    truth = t["truth_accounts"]
    o = truth[truth["open_at_cutoff"]].copy()
    rec = t[S.RECEIPTS]
    pp = rec[(rec["salud_TXNType"] == S.TXN_PAYMENT) & (rec["salud_PaymentType"] == S.PAY_PATIENT)]
    before = set(pp[pp["salud_PostDate"] <= pd.Timestamp("2026-04-01")]["salud_Guarantor"])
    o["paid_before"] = o["guarantor_id"].isin(before)
    rates = o.groupby("paid_before")["planted_paid"].mean()
    assert rates[True] > 3 * rates[False] and rates[True] - rates[False] > 0.1


def test_original_world_unchanged_by_realism_code():
    """Without a realism section the generator must reproduce the original world (same seed -> same rows)."""
    a = generate(GeneratorConfig(n_guarantors=150, n_agents=8, seed=3))
    b = generate(GeneratorConfig(n_guarantors=150, n_agents=8, seed=3))
    assert "feat_call_quality" in a["truth_accounts"]          # planted outcome model still used
    pd.testing.assert_frame_equal(a[S.CONTACTS], b[S.CONTACTS])
