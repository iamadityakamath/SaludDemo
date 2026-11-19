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
class Realism:
    """Knobs that make the world look like Salud's final package (Nov 2026). Only used when GeneratorConfig.realism
    is set; without it the generator behaves exactly as before. Values are tuned with `salud-synth-calibrate`.

    Main differences from the original world:
    - accounts per patient are heavy-tailed (negative binomial), balances smaller, visits spread over time;
    - insurance processing: insurance payments and adjustments shortly after discharge (more likely for able payers);
    - payments follow a *habit* process instead of a planted outcome model: a patient who starts paying usually keeps
      paying monthly; starting becomes likelier as the debt ages (statements accumulate, insurance is settled). The
      same process runs before and after the cutoff, so every scoring date carries the same relationships;
    - calls: more inbound calls, outbound campaign calls mostly answered by the phone menu (IVR), a share of inbound
      calls transferred to another department, few messages left.
    """

    # accounts and balances
    accounts_extra_mean: float = 3.9       # accounts per patient = 1 + NegBin(mean, dispersion), capped
    accounts_dispersion: float = 0.45
    max_accounts: int = 40
    visit_spread_days: int = 420           # a patient's later visits fall up to this long after the first
    hb_share: float = 0.48
    hb_balance_mu: float = 5.25            # lognormal placement balance per account
    hb_balance_sd: float = 1.6
    pb_balance_mu: float = 4.45
    pb_balance_sd: float = 1.4
    # insurance processing (per account, around discharge + 55..110 days)
    insurance_payment_p: float = 0.08
    adjustment_p: float = 0.12
    insurance_ability_slope: float = 1.0   # insured / processed accounts are likelier for able payers
    # payment habit (monthly)
    start_intercept: float = -4.2
    start_ability: float = 2.2
    start_willingness: float = 1.2
    start_age: float = 0.55                # per log1p(months since first placement)
    continue_intercept: float = 0.6
    continue_ability: float = 1.8
    plan_fraction_low: float = 0.08        # monthly amount as a share of the open balance
    plan_fraction_high: float = 0.3
    call_to_habit: float = 0.6             # payment / plan / kept promise on a call -> starts the habit
    # calls
    inbound_scale: float = 1.0             # multiplies the willingness-driven inbound call rate
    campaign_share: float = 0.74           # connected outbound calls that come from a campaign
    campaign_ivr_share: float = 0.88       # campaign calls answered by the phone menu (no agent)
    message_left_share: float = 0.08       # unanswered calls where a message was left
    transfer_rate: float = 0.3             # inbound agent calls transferred to another department
    other_contact_rate: float = 0.08       # non-transferred agent calls with contact type 'Other'
    hardship_scale: float = 0.4            # weight of the hardship scenario
    dispute_itemized_p: float = 0.3        # disputes where an itemized bill is sent
    talk_mu: float = 5.35                  # lognormal agent talk time
    result_offsets: dict = field(default_factory=dict)   # added to the call-result logits, e.g. {"non_committal": -0.5}
    inbound_pay_offset: float = 0.0        # inbound callers: added to the pay / promise / plan result logits
    ptp_keep_scale: float = 1.0            # multiplies the chance a promise to pay is kept
    outbound_skip_paying: float = 0.0      # collections campaigns skip patients who are currently paying
    pay_by_phone: float = 0.0              # a paying patient makes the month's payment by calling in
    call_delay_days: float | None = None   # calls follow a random account's placement by Exp(mean) days;
                                           #   None = spread evenly over the whole history (as before)
    phone_pay_max_months: float | None = None  # pay by phone only within this many months of the latest placement
    plan_on_promise: float = 0.0           # promise-to-pay calls that set up a plan, first payment on the promised date
    insurance_review_p: float = 1.0        # share of insurance calls where "Requested Insurance Review" is recorded
    asked_fa_rate: float = 0.0             # agent calls (not hardship) where the patient asks about financial assistance
    salud_collection_attempt: bool = False # fill CollectionAttempt like Salud: blank on most agent calls, Yes on IVR calls
    same_day_posting: bool = False         # money taken on a call posts the same day (else 1-30 hours later)


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
    realism: Realism | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "GeneratorConfig":
        raw = json.loads(Path(path).read_text())
        outcome = OutcomeModel(**raw.pop("outcome", {}))
        realism = Realism(**raw.pop("realism")) if "realism" in raw else None
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw, outcome=outcome, realism=realism)

    def to_dict(self) -> dict:
        return asdict(self)
