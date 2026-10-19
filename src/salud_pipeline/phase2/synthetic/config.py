"""Generator configuration. Every knob that shapes the planted 'truth' lives here.

Conventions (documented for the handoff's open questions):
- Timestamps are naive local time, America/Chicago.
- Money is USD. Receipt amounts are positive for cash/credit received; reversals and
  refunds would be negative (none are generated in v1).
- ActionsTaken is a ';'-joined list of option codes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class OutcomeModel:
    """Logit coefficients for self-pay payment in the outcome window (the planted truth).

    Inputs are standardized inside the generator. Set call_quality to 0.0 to simulate a
    world where call quality carries no signal; the pipeline should then find none.
    """

    intercept: float = -0.6
    ability: float = 1.4          # latent ability to pay
    willingness: float = 1.1      # latent willingness / engagement
    prior_payer: float = 0.8      # paid anything before the cutoff
    balance: float = -0.35        # log open balance
    aging: float = -0.45          # days since placement
    call_quality: float = 0.5     # quality of the last connected call before the cutoff
    promised_to_pay: float = 0.6  # last call ended with a promise to pay
    hardship_fa: float = -1.0     # hardship patient routed to financial assistance
    on_plan: float = 0.7          # active payment plan at the cutoff


@dataclass
class GeneratorConfig:
    seed: int = 7
    n_guarantors: int = 2000
    n_agents: int = 40

    # Timeline
    history_start: str = "2025-07-01"
    prediction_cutoff: str = "2026-04-01"
    outcome_horizon_days: int = 90

    # Population
    mean_accounts_per_guarantor: float = 1.6
    hardship_rate: float = 0.18
    do_not_call_rate: float = 0.03
    spanish_rate: float = 0.12
    representative_caller_rate: float = 0.15   # caller is a non-patient guarantor / family member

    # Contact activity before the cutoff
    mean_contact_attempts: float = 5.0
    inbound_share_of_connected: float = 0.55   # sample transcripts are mostly inbound

    # QA
    qa_review_rate: float = 0.35          # share of connected calls with a completed QA review
    qa_reviewer_flip_rate: float = 0.04   # human-reviewer disagreement with the planted label

    # Transcripts
    transcript_noise: float = 0.15        # ASR-style fillers, masking and dropped words

    outcome: OutcomeModel = field(default_factory=OutcomeModel)

    @classmethod
    def from_json(cls, path: str | Path) -> "GeneratorConfig":
        raw = json.loads(Path(path).read_text())
        outcome = OutcomeModel(**raw.pop("outcome", {}))
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw, outcome=outcome)

    def to_dict(self) -> dict:
        return asdict(self)
