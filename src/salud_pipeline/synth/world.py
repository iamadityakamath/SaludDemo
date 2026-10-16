"""Simulate a synthetic Salud worklist with planted, recoverable ground truth.

Pipeline:
  1. Latent traits: agents (per-QA-item skill) and guarantors (ability, willingness, reachability,
     hardship).
  2. Pre-cutoff history per guarantor, in time order: outreach attempts, connected calls (true QA
     items -> call outcome), spontaneous payments, promise-to-pay follow-through, payment plans,
     financial-assistance and insurance adjustments.
  3. Outcome window: each open account pays with probability sigmoid(OutcomeModel . features).
  4. Worklist snapshot as of the day after the outcome window (current state, so it leaks the
     outcome if used naively; features must be rebuilt from dated rows before the cutoff).

Everything the CRM would not know (latents, true call outcome, planted probability) goes to the
truth tables, never into the CRM-shaped tables.
"""

from __future__ import annotations

import heapq
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import schema as S
from .config import GeneratorConfig
from .text import agent_note, lessons_learned, qa_comment

# Base log-odds that an average agent passes each QA item (skill shifts all items).
QA_ITEM_BASE = {
    "greeting": 2.4, "culture": 1.8, "capturing_data": 1.4, "understanding_needs": 0.9,
    "correct_info": 1.5, "first_contact_resolution": 0.6, "compliance": 1.9,
    "call_statusing": 1.9, "grammar": 1.2,
}
FIN_CLASSES_STD = ["Commercial - Balance After Insurance", "Medicare - Balance After Insurance",
                   "Self Pay", "Commercial High Deductible"]
FIN_CLASSES_HARDSHIP = ["Self Pay - Uninsured", "Medicaid - Pending", "Self Pay"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Agent:
    name: str
    skill: float
    warmth: float
    item_p: dict[str, float]


@dataclass
class Guarantor:
    gid: str
    ability: float
    willingness: float
    reachability: float
    hardship: bool
    language: str
    caller_type: str
    do_not_call: bool


@dataclass
class Account:
    aid: str
    gid: str
    hbpb: int
    fin_class: str
    discharge: pd.Timestamp
    placement: pd.Timestamp
    last_placement: pd.Timestamp
    placement_balance: float
    balance: float
    receipts_total: float = 0.0
    adj_total: float = 0.0
    last_txn: pd.Timestamp | None = None
    last_patient_payment: pd.Timestamp | None = None
    zero_date: pd.Timestamp | None = None
    on_plan: bool = False
    plan_amount: float = 0.0
    plan_payments: int = 0
    plan_missed: int = 0
    plan_last_payment: pd.Timestamp | None = None
    fa_provided: bool = False
    paid_before_cutoff: bool = False
    last_attempt: pd.Timestamp | None = None
    last_connect: pd.Timestamp | None = None
    last_call_quality: float | None = None
    last_call_ptp: bool = False


@dataclass
class World:
    cfg: GeneratorConfig
    rng: np.random.Generator
    cutoff: pd.Timestamp
    window_end: pd.Timestamp
    snapshot: pd.Timestamp
    agents: list[Agent] = field(default_factory=list)
    guarantors: list[Guarantor] = field(default_factory=list)
    accounts: dict[str, Account] = field(default_factory=dict)
    contacts: list[dict] = field(default_factory=list)
    bridge: list[dict] = field(default_factory=list)
    receipts: list[dict] = field(default_factory=list)
    qa: list[dict] = field(default_factory=list)
    call_truth: list[dict] = field(default_factory=list)
    account_truth: list[dict] = field(default_factory=list)
    # guarantor id -> [(time, contact id)] of outbound calls, for receipts.LastOutboundCall
    outbound_calls: dict[str, list[tuple[pd.Timestamp, str]]] = field(default_factory=dict)

    # ---------- helpers ----------
    def uid(self) -> str:
        return str(uuid.UUID(bytes=self.rng.bytes(16), version=4))

    def rand_time(self, lo: pd.Timestamp, hi: pd.Timestamp, business_hours: bool = True) -> pd.Timestamp:
        days = max((hi - lo).days, 0)
        day = (lo + pd.Timedelta(days=int(self.rng.integers(0, days + 1)))).normalize()
        if business_hours:
            secs = int(self.rng.integers(8 * 3600, 18 * 3600))
        else:
            secs = int(self.rng.integers(0, 24 * 3600))
        return day + pd.Timedelta(seconds=secs)

    def post(self, acct: Account, when: pd.Timestamp, amount: float, txn: int | None, ptype: int | None,
             exclude: int | None = S.TXN_INCLUDE) -> None:
        """Append a receipt row and update the account's running state."""
        amount = round(float(amount), 2)
        if amount <= 0:
            return
        last_out = None
        for t, cid in reversed(self.outbound_calls.get(acct.gid, [])):
            if t <= when:
                last_out = cid
                break
        self.receipts.append({
            "salud_gphpvreceiptsId": self.uid(), "salud_ReceiptsId": acct.aid,
            "salud_Guarantor": acct.gid, "salud_PostDate": when, "salud_Amount": amount,
            "salud_TXNType": txn, "salud_PaymentType": ptype, "salud_ExcludeTransaction": exclude,
            "salud_HBPB": acct.hbpb, "salud_LastOutboundCall": last_out,
        })
        counts = exclude == S.TXN_INCLUDE
        if not counts:
            return
        if txn == S.TXN_PAYMENT and ptype == S.PAY_PATIENT:
            acct.receipts_total += amount
            acct.last_patient_payment = when
            if when <= self.cutoff:
                acct.paid_before_cutoff = True
        elif txn == S.TXN_ADJUSTMENT:
            acct.adj_total += amount
        acct.balance = round(acct.balance - amount, 2)
        acct.last_txn = when
        if acct.balance <= 0.005 and acct.zero_date is None:
            acct.balance = 0.0
            acct.zero_date = when


def build_agents(w: World) -> None:
    for i in range(w.cfg.n_agents):
        skill = float(w.rng.normal(0, 1))
        item_p = {k: float(_sigmoid(b + 1.1 * skill + w.rng.normal(0, 0.5))) for k, b in QA_ITEM_BASE.items()}
        w.agents.append(Agent(name=f"Agent {i + 1:03d}", skill=skill,
                              warmth=float(np.clip(0.5 + 0.2 * skill + w.rng.normal(0, 0.15), 0, 1)),
                              item_p=item_p))


def build_guarantors_and_accounts(w: World, history_start: pd.Timestamp) -> None:
    cfg, rng = w.cfg, w.rng
    latest_placement = w.cutoff - pd.Timedelta(days=14)
    for _ in range(cfg.n_guarantors):
        hardship = bool(rng.random() < cfg.hardship_rate)
        g = Guarantor(
            gid=w.uid(),
            ability=float(rng.beta(1.8, 5.0) if hardship else rng.beta(4.0, 2.5)),
            willingness=float(rng.beta(3.0, 2.0)),
            reachability=float(rng.beta(2.5, 2.0)),
            hardship=hardship,
            language="spanish" if rng.random() < cfg.spanish_rate else "english",
            caller_type="representative" if rng.random() < cfg.representative_caller_rate else "patient",
            do_not_call=bool(rng.random() < cfg.do_not_call_rate),
        )
        w.guarantors.append(g)
        n_acc = 1 + int(rng.poisson(max(cfg.mean_accounts_per_guarantor - 1, 0)))
        for _ in range(n_acc):
            hbpb = S.HB if rng.random() < 0.6 else S.PB
            discharge = w.rand_time(history_start - pd.Timedelta(days=120),
                                    latest_placement - pd.Timedelta(days=20), business_hours=False).normalize()
            placement = min(discharge + pd.Timedelta(days=int(rng.integers(20, 91))), latest_placement).normalize()
            last_placement = placement
            if rng.random() < 0.08:
                last_placement = min(placement + pd.Timedelta(days=int(rng.integers(30, 120))), latest_placement)
            mu, sd = (6.4, 1.0) if hbpb == S.HB else (5.1, 0.8)
            bal = round(float(np.clip(rng.lognormal(mu, sd), 25, 60000)), 2)
            pool = FIN_CLASSES_HARDSHIP if hardship else FIN_CLASSES_STD
            a = Account(aid=w.uid(), gid=g.gid, hbpb=hbpb, fin_class=pool[int(rng.integers(len(pool)))],
                        discharge=discharge, placement=placement, last_placement=last_placement,
                        placement_balance=bal, balance=bal)
            w.accounts[a.aid] = a


# ---------- calls ----------

def _pick_scenario(rng, g: Guarantor, inbound: bool) -> str:
    if inbound and rng.random() < 0.05:
        return "third_party"
    weights = {
        "wants_to_pay": 0.15 + 0.35 * g.willingness * g.ability,
        "billing_question": 0.30,
        "insurance_issue": 0.20,
        "hardship": 0.05 + (0.55 if g.hardship else 0.0),
        "dispute": 0.05 + 0.25 * (1 - g.willingness),
    }
    keys = list(weights)
    p = np.array([weights[k] for k in keys])
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _true_result(rng, g: Guarantor, scenario: str, qz: float, hung_up: bool) -> str:
    if scenario == "third_party":
        return "na_transferred"
    if hung_up:
        return "would_not_pay" if rng.random() < 0.4 else "non_committal"
    ab, wi = g.ability, g.willingness
    logits = {
        "paid_account_full": -2.2 + 2.5 * ab + 1.0 * wi + 0.6 * qz,
        "partial_payment": -1.6 + 1.2 * ab + 1.0 * wi + 0.6 * qz,
        "promised_to_pay": -0.6 + 0.6 * ab + 1.6 * wi + 0.8 * qz,
        "paid_plan_full": -1.6 + 1.6 * (1 - ab) + 1.0 * wi + 0.8 * qz,
        "non_committal": 0.2 - 0.2 * qz,
        "contested_balance": -1.2 - 1.5 * wi - 0.3 * qz,
        "would_not_pay": -2.2 - 2.0 * wi - 0.6 * qz,
    }
    if scenario == "insurance_issue":
        logits["pursued_insurance"] = 1.2
    if scenario == "dispute":
        logits["contested_balance"] += 1.5
    if scenario == "wants_to_pay":
        logits["paid_account_full"] += 1.2
        logits["partial_payment"] += 0.6
    if scenario == "hardship":
        logits["paid_plan_full"] += 1.0
        logits["paid_account_full"] -= 1.5
    keys = list(logits)
    z = np.array([logits[k] for k in keys])
    p = np.exp(z - z.max())
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _misstatus(rng, true_result: str) -> str | None:
    """Recorded result when Correct Call Statusing = No."""
    if rng.random() < 0.2:
        return None
    wrong = {"promised_to_pay": "non_committal", "non_committal": "promised_to_pay",
             "contested_balance": "would_not_pay", "would_not_pay": "non_committal",
             "pursued_insurance": "non_committal", "partial_payment": "paid_account_full",
             "paid_account_full": "partial_payment", "paid_plan_full": "promised_to_pay",
             "na_transferred": "non_committal"}
    return wrong.get(true_result, "non_committal")


def simulate_connected_call(w: World, g: Guarantor, when: pd.Timestamp, direction: str,
                            accts: list[Account], cid: str, events: list) -> dict:
    """Create the CRM fields for a connected call and the hidden truth row."""
    rng, cfg = w.rng, w.cfg
    agent = w.agents[int(rng.integers(len(w.agents)))]
    inbound = direction.startswith("inbound")
    scenario = _pick_scenario(rng, g, inbound)

    items = {k: bool(rng.random() < agent.item_p[k]) for k in S.QA_KEYS}
    # Context makes some items harder: disputes and insurance issues are harder to close on one call.
    if scenario in ("dispute", "insurance_issue", "third_party") and rng.random() < 0.35:
        items["first_contact_resolution"] = False
    transcript_items = [k for k in S.QA_KEYS if k not in ("call_statusing", "grammar")]
    quality = float(np.mean([items[k] for k in S.QA_KEYS]))
    qz = (quality - 0.75) / 0.18

    hung_up = bool(scenario != "third_party" and rng.random() < 0.10 * (1.5 - g.willingness) * (1.6 - quality))
    result = _true_result(rng, g, scenario, qz, hung_up)
    sentiment = float(np.tanh(1.6 * (g.willingness - 0.55) + 0.6 * qz
                              + (-0.8 if result in ("contested_balance", "would_not_pay") else 0.0)
                              + (0.5 if result.startswith("paid") else 0.0)
                              + rng.normal(0, 0.35)))
    sentiment_label = "positive" if sentiment > 0.25 else ("negative" if sentiment < -0.25 else "neutral")

    open_accts = [a for a in accts if a.balance > 0]
    total_bal = round(sum(a.balance for a in open_accts), 2)

    actions: list[str] = []
    if scenario != "third_party":
        actions.append("shared_balance")
    if scenario == "hardship" or (g.hardship and items["understanding_needs"]):
        if items["understanding_needs"]:
            actions += ["provided_fa_options", "fa_paperwork_sent"]
    if scenario == "insurance_issue" or result == "pursued_insurance":
        actions.append("requested_insurance_review")
    if scenario == "dispute" or result == "contested_balance":
        actions.append("provided_itemized")
    if result == "paid_plan_full":
        actions.append("set_up_plan")
    if items["capturing_data"]:
        actions.append("updated_contact_info")
    actions = list(dict.fromkeys(actions))

    # Money on the call.
    pay_now = 0.0
    promise_date = None
    if open_accts and result in ("paid_account_full", "partial_payment", "paid_plan_full"):
        post_at = when + pd.Timedelta(hours=int(rng.integers(1, 30)))
        if result == "paid_account_full":
            target = open_accts if rng.random() < 0.5 else [open_accts[0]]
            for a in target:
                pay_now += a.balance
                heapq.heappush(events, (post_at, "pay", a.aid, a.balance))
            if target is open_accts and len(open_accts) > 0:
                result = "paid_guarantor_full" if len(open_accts) == len(accts) else result
        elif result == "partial_payment":
            a = open_accts[0]
            amt = round(a.balance * float(rng.uniform(0.15, 0.6)), 2)
            pay_now = amt
            heapq.heappush(events, (post_at, "pay", a.aid, amt))
        else:  # plan
            n = int(rng.integers(4, 13))
            for a in open_accts:
                a.on_plan = True
                a.plan_amount = round(a.balance / n, 2)
                heapq.heappush(events, (post_at, "plan", a.aid, a.plan_amount))
            pay_now = sum(a.plan_amount for a in open_accts)
    if result == "promised_to_pay" and open_accts:
        days = int(rng.integers(3, 22))
        promise_date = (when + pd.Timedelta(days=days)).strftime("%m/%d")
        p_keep = 0.35 + 0.35 * g.ability + 0.2 * g.willingness
        if rng.random() < p_keep:
            a = open_accts[0]
            frac = 1.0 if rng.random() < 0.5 * g.ability + 0.2 else float(rng.uniform(0.2, 0.7))
            heapq.heappush(events, (when + pd.Timedelta(days=days + int(rng.integers(-2, 4))),
                                    "pay", a.aid, round(a.balance * frac, 2)))
    if "provided_fa_options" in actions:
        for a in open_accts:
            a.fa_provided = True
            if g.hardship and rng.random() < 0.6:
                heapq.heappush(events, (when + pd.Timedelta(days=int(rng.integers(25, 75))),
                                        "fa_adj", a.aid, round(a.balance * float(rng.uniform(0.5, 1.0)), 2)))
    if "requested_insurance_review" in actions and open_accts and rng.random() < 0.45:
        a = open_accts[0]
        heapq.heappush(events, (when + pd.Timedelta(days=int(rng.integers(20, 70))),
                                "ins_pay", a.aid, round(a.balance * float(rng.uniform(0.3, 1.0)), 2)))

    recorded = result if items["call_statusing"] else _misstatus(rng, result)
    transferred = None
    if not items["first_contact_resolution"] and (scenario in ("third_party", "insurance_issue") or rng.random() < 0.3):
        transferred = "provider" if scenario == "third_party" else ("billing_office" if rng.random() < 0.6 else "supervisor")
    collection_attempt = scenario != "third_party" and (not inbound or rng.random() < 0.85)
    talk = float(np.clip(rng.lognormal(5.6 + 0.3 * (scenario in ("dispute", "insurance_issue")), 0.45), 45, 2400))
    if hung_up:
        talk = float(rng.uniform(20, 120))

    contact_type = "patient" if g.caller_type == "patient" else "non_patient_guarantor"
    if scenario == "third_party":
        contact_type = "insurance"

    for a in accts:
        a.last_connect = when
        a.last_call_quality = quality
        a.last_call_ptp = result == "promised_to_pay"

    row = {
        "salud_gphpvcontacthistoryId": cid,
        "salud_GuarantorId": g.gid,
        "salud_DateandTimeofContact": when,
        "salud_ConnectionType": S.CONNECTION[direction],
        "salud_Contact_Type": S.CONTACT_TYPE[contact_type],
        "salud_CallDisposition": S.DISPOSITION["hung_up" if hung_up else "spoke_with_csr"],
        "salud_CollectionAttempt": S.YES if collection_attempt else S.NO,
        "salud_CollectionResult": S.RESULT[recorded] if recorded else None,
        "salud_ActionsTaken": S.ACTIONS_SEP.join(str(S.ACTION[a]) for a in actions) or None,
        "salud_PaymentAmount": round(pay_now, 2) if pay_now > 0 else None,
        "salud_ReviewBalance": int(scenario != "third_party"),
        "salud_inquireaboutfinancialassistance": int(scenario == "hardship"),
        "salud_RequestedCallback": int(not items["first_contact_resolution"] and transferred is None),
        "salud_CallbackDate": None,
        "salud_needsomeoneelsetransfer": int(transferred is not None),
        "salud_TransferredTo": S.TRANSFERRED_TO[transferred] if transferred else None,
        "salud_PreferredLanguage": S.LANGUAGE[g.language],
        "salud_AgentCallTalkDuration": round(talk, 1),
        "salud_AgentCallWrapUpDuration": round(float(rng.uniform(20, 180)), 1),
        "salud_Agent": agent.name,
        "salud_TCNRecordID": f"SYN-{rng.bytes(6).hex().upper()}",
        "salud_Note": agent_note(rng, scenario, recorded, actions, items["grammar"], promise_date),
    }
    if row["salud_RequestedCallback"]:
        row["salud_CallbackDate"] = (when + pd.Timedelta(days=int(rng.integers(1, 6)))).normalize() + pd.Timedelta(hours=10)

    truth = {
        "contact_id": cid, "guarantor_id": g.gid, "agent": agent.name, "agent_skill": agent.skill,
        "call_time": when, "direction": direction, "scenario": scenario, "caller_type": contact_type,
        "language": g.language, "hardship": g.hardship, "hung_up": hung_up,
        **{f"qa_{k}": items[k] for k in S.QA_KEYS},
        "quality_score": quality,
        "quality_score_transcript_items": float(np.mean([items[k] for k in transcript_items])),
        "sentiment_score": sentiment, "sentiment_label": sentiment_label,
        "true_result": result, "recorded_result": recorded,
        "promise_date": promise_date, "total_balance_discussed": total_bal,
        "n_accounts_discussed": len(accts), "actions": ";".join(actions),
        "transferred_to": transferred, "agent_warmth": agent.warmth,
        "ability": g.ability, "willingness": g.willingness,
    }
    w.call_truth.append(truth)
    return row


def _qa_review(w: World, contact: dict, truth: dict) -> None:
    rng = w.rng
    row = {
        "salud_gphpvqualityreviewId": w.uid(),
        "salud_QualityReviewId": contact["salud_gphpvcontacthistoryId"],
        "salud_reviewdate": (contact["salud_DateandTimeofContact"] + pd.Timedelta(days=int(rng.integers(1, 15)))).normalize(),
    }
    any_no = False
    for key, score_col, comment_col, _ in S.QA_ITEMS:
        passed = truth[f"qa_{key}"]
        if rng.random() < w.cfg.qa_reviewer_flip_rate:
            passed = not passed
        any_no |= not passed
        row[score_col] = S.YES if passed else S.NO
        row[comment_col] = qa_comment(rng, key, passed)
    row["salud_lessonslearned"] = lessons_learned(rng) if any_no and rng.random() < 0.5 else None
    w.qa.append(row)


# ---------- per-guarantor history ----------

NO_CONNECT_DISPOSITIONS = ["busy", "vm_no_message", "vm_left_message", "no_answer",
                           "left_message_with_person", "invalid_number"]


def simulate_history(w: World, g: Guarantor) -> None:
    rng, cfg = w.rng, w.cfg
    accts = [a for a in w.accounts.values() if a.gid == g.gid]
    first = min(a.placement for a in accts)
    events: list = []  # (time, kind, account id, amount)

    n_attempts = 1 + int(rng.poisson(cfg.mean_contact_attempts))
    for _ in range(n_attempts):
        heapq.heappush(events, (w.rand_time(first + pd.Timedelta(days=3), w.cutoff - pd.Timedelta(days=1)),
                                "contact", "", 0.0))
    # Patient-initiated inbound calls scale with willingness.
    for _ in range(int(rng.poisson(0.4 + 1.2 * g.willingness))):
        heapq.heappush(events, (w.rand_time(first + pd.Timedelta(days=5), w.cutoff - pd.Timedelta(days=1)),
                                "inbound", "", 0.0))
    # Spontaneous (statement-driven) payments.
    for a in accts:
        months = max((w.cutoff - a.placement).days / 30.0, 0.0)
        lam = months * 0.10 * float(_sigmoid(-1.0 + 2.5 * g.ability + 1.0 * g.willingness - 0.25 * np.log1p(a.balance) + 1.0))
        for _ in range(int(rng.poisson(lam))):
            heapq.heappush(events, (w.rand_time(a.placement + pd.Timedelta(days=10), w.cutoff, business_hours=False),
                                    "selfpay", a.aid, -1.0))

    while events:
        when, kind, aid, amount = heapq.heappop(events)
        if when > w.window_end:
            continue
        if kind in ("contact", "inbound"):
            if when > w.cutoff:
                continue
            _contact_event(w, g, accts, when, kind, events)
        elif kind == "selfpay":
            a = w.accounts[aid]
            if a.balance > 0 and when <= w.cutoff:
                frac = 1.0 if rng.random() < 0.15 + 0.3 * g.ability else float(rng.uniform(0.1, 0.5))
                w.post(a, when, a.balance * frac, S.TXN_PAYMENT, S.PAY_PATIENT)
        elif kind == "pay":
            a = w.accounts[aid]
            if when <= w.cutoff:  # post-cutoff money comes only from the outcome model
                w.post(a, when, min(amount, a.balance), S.TXN_PAYMENT, S.PAY_PATIENT)
        elif kind == "plan":
            a = w.accounts[aid]
            if when > w.cutoff or a.balance <= 0:
                continue
            if rng.random() < 0.55 + 0.35 * g.ability * (0.5 + g.willingness):
                w.post(a, when, min(amount, a.balance), S.TXN_PAYMENT, S.PAY_PATIENT)
                a.plan_payments += 1
                a.plan_last_payment = when
            else:
                a.plan_missed += 1
            if a.balance > 0:
                heapq.heappush(events, (when + pd.Timedelta(days=30), "plan", aid, amount))
            else:
                a.on_plan = False
        elif kind == "fa_adj":
            a = w.accounts[aid]
            w.post(a, when, min(amount, a.balance), S.TXN_ADJUSTMENT, None)
        elif kind == "ins_pay":
            a = w.accounts[aid]
            w.post(a, when, min(amount, a.balance), S.TXN_PAYMENT, S.PAY_INSURANCE)


def _contact_event(w: World, g: Guarantor, accts: list[Account], when: pd.Timestamp, kind: str, events: list) -> None:
    rng = w.rng
    open_accts = [a for a in accts if a.balance > 0 and a.placement <= when]
    if not open_accts:
        return
    covered = open_accts if rng.random() < 0.85 else [open_accts[int(rng.integers(len(open_accts)))]]
    cid = w.uid()

    if kind == "inbound":
        direction = "inbound_returned" if rng.random() < 0.25 else "inbound"
        row = simulate_connected_call(w, g, when, direction, covered, cid, events)
    else:
        channel = rng.choice(["call", "text", "email"], p=[0.65, 0.22, 0.13])
        if g.do_not_call and channel == "call":
            channel = "text"
        if channel == "call":
            connected = rng.random() < 0.15 + 0.55 * g.reachability
            for a in covered:
                a.last_attempt = when
            if connected:
                direction = "outbound_manual" if rng.random() < 0.5 else "outbound_campaign"
                w.outbound_calls.setdefault(g.gid, []).append((when, cid))
                row = simulate_connected_call(w, g, when, direction, covered, cid, events)
            else:
                w.outbound_calls.setdefault(g.gid, []).append((when, cid))
                disp = NO_CONNECT_DISPOSITIONS[int(rng.integers(len(NO_CONNECT_DISPOSITIONS)))]
                manual = rng.random() < 0.3
                row = _blank_contact(cid, g, when, "outbound_no_connection", disp)
                row["salud_Agent"] = w.agents[int(rng.integers(len(w.agents)))].name if manual else None
                row["salud_AgentCallTalkDuration"] = round(float(rng.uniform(0, 40)), 1)
                row["salud_Note"] = {"vm_left_message": "LVM", "vm_no_message": "VM, no msg left",
                                     "left_message_with_person": "Left msg w/ household member",
                                     "invalid_number": "Number not in service"}.get(disp, "No answer")
        else:
            ok = rng.random() < 0.85
            if channel == "text":
                ctype = "text_campaign" if ok else "text_no_connection"
                disp = "text_successful" if ok else "text_failed"
            else:
                ctype = "email_campaign" if ok else "email_no_connection"
                disp = "email_sent"
            row = _blank_contact(cid, g, when, ctype, disp)
    w.contacts.append(row)
    for a in covered:
        w.bridge.append({"salud_gphpvaccountsid": a.aid, "salud_gphpvcontacthistoryid": cid})
    if row["salud_CallDisposition"] in (S.DISPOSITION["spoke_with_csr"], S.DISPOSITION["hung_up"]):
        if rng.random() < w.cfg.qa_review_rate:
            _qa_review(w, row, w.call_truth[-1])


def _blank_contact(cid: str, g: Guarantor, when: pd.Timestamp, ctype: str, disp: str) -> dict:
    row = {c: None for c in S.COLUMNS[S.CONTACTS]}
    row.update({
        "salud_gphpvcontacthistoryId": cid, "salud_GuarantorId": g.gid,
        "salud_DateandTimeofContact": when, "salud_ConnectionType": S.CONNECTION[ctype],
        "salud_CallDisposition": S.DISPOSITION[disp], "salud_PreferredLanguage": S.LANGUAGE[g.language],
    })
    return row


# ---------- outcome window ----------

def simulate_outcomes(w: World, glookup: dict[str, Guarantor]) -> None:
    rng, cfg, oc = w.rng, w.cfg, w.cfg.outcome
    open_accts = [a for a in w.accounts.values() if a.balance > 0]
    if not open_accts:
        return
    feats = pd.DataFrame({
        "ability": [glookup[a.gid].ability for a in open_accts],
        "willingness": [glookup[a.gid].willingness for a in open_accts],
        "prior_payer": [float(a.paid_before_cutoff) for a in open_accts],
        "balance": [np.log1p(a.balance) for a in open_accts],
        "aging": [(w.cutoff - a.placement).days for a in open_accts],
        "call_quality": [a.last_call_quality if a.last_call_quality is not None else np.nan for a in open_accts],
        "promised_to_pay": [float(a.last_call_ptp) for a in open_accts],
        "hardship_fa": [float(glookup[a.gid].hardship and a.fa_provided) for a in open_accts],
        "on_plan": [float(a.on_plan) for a in open_accts],
    })
    z = feats.copy()
    for c in ("ability", "willingness", "balance", "aging", "call_quality"):
        col = feats[c]
        z[c] = (col - col.mean()) / (col.std() or 1.0)
    z["call_quality"] = z["call_quality"].fillna(0.0)  # no connected call -> average
    coefs = {k: getattr(oc, k) for k in feats.columns}
    lp = oc.intercept + sum(coefs[k] * z[k] for k in feats.columns)
    p = _sigmoid(lp.to_numpy())
    paid = rng.random(len(p)) < p

    horizon = cfg.outcome_horizon_days
    for i, a in enumerate(open_accts):
        g = glookup[a.gid]
        bal_at_cutoff = a.balance
        if paid[i]:
            if a.on_plan and a.plan_amount > 0:
                start = w.cutoff + pd.Timedelta(days=int(rng.integers(1, 30)))
                t = start
                while t <= w.window_end and a.balance > 0:
                    if t == start or rng.random() < 0.6 + 0.3 * g.ability:
                        w.post(a, t, min(a.plan_amount, a.balance), S.TXN_PAYMENT, S.PAY_PATIENT)
                        a.plan_payments += 1
                        a.plan_last_payment = t
                    else:
                        a.plan_missed += 1
                    t = t + pd.Timedelta(days=30)
            else:
                day = 1 + int((horizon - 1) * rng.beta(1.2, 1.5 + 2.5 * g.ability))
                t = w.cutoff + pd.Timedelta(days=day, hours=int(rng.integers(8, 20)))
                full = rng.random() < float(_sigmoid(-0.3 + 1.5 * (g.ability - 0.5) * 2 - 0.3 * (feats.loc[i, "balance"] - 6)))
                amt = a.balance if full else round(a.balance * float(rng.uniform(0.15, 0.7)), 2)
                w.post(a, t, amt, S.TXN_PAYMENT, S.PAY_PATIENT)
        w.account_truth.append({
            "account_id": a.aid, "guarantor_id": a.gid, "open_at_cutoff": True,
            "balance_at_cutoff": bal_at_cutoff, "planted_logit": float(lp.iloc[i]), "planted_p": float(p[i]),
            "planted_paid": bool(paid[i]),
            **{f"feat_{k}": float(feats.loc[i, k]) if pd.notna(feats.loc[i, k]) else None for k in feats.columns},
        })
    open_ids = {a.aid for a in open_accts}
    for a in w.accounts.values():
        if a.aid not in open_ids:
            w.account_truth.append({"account_id": a.aid, "guarantor_id": a.gid, "open_at_cutoff": False,
                                    "balance_at_cutoff": 0.0, "planted_logit": None, "planted_p": None,
                                    "planted_paid": None})


def _add_filter_traps(w: World) -> None:
    """A few rows that must NOT count as self-pay cash, to exercise the outcome filter."""
    rng = w.rng
    accts = list(w.accounts.values())
    for a in rng.choice(accts, size=max(1, len(accts) // 50), replace=False):
        w.post(a, w.rand_time(a.placement, w.window_end), float(rng.uniform(5, 50)),
               S.TXN_PAYMENT, S.PAY_PATIENT, exclude=S.TXN_EXCLUDE)
    for a in rng.choice(accts, size=max(1, len(accts) // 100), replace=False):
        w.post(a, w.rand_time(a.placement, w.window_end), float(rng.uniform(5, 50)),
               S.TXN_PAYMENT, None, exclude=None)


# ---------- worklist snapshot ----------

def worklist_rows(w: World) -> list[dict]:
    rng = w.rng
    rows = []
    for a in w.accounts.values():
        end = a.zero_date or w.snapshot
        stmts = pd.date_range(a.placement + pd.Timedelta(days=10), end, freq="30D")
        status = S.PLACEMENT_ZERO if a.balance <= 0 else (
            S.PLACEMENT_WITHDRAWN if rng.random() < 0.03 else S.PLACEMENT_ACTIVE)
        rows.append({
            "salud_gphpvaccountsId": a.aid, "salud_GuarantorId": a.gid, "salud_HBPB": a.hbpb,
            "salud_FinancialClass": a.fin_class, "salud_PlacementStatus": status,
            "salud_PlacementDate": a.placement, "salud_LastPlacementDate": a.last_placement,
            "salud_DischargeDate": a.discharge, "salud_PlacementPatientBalance": a.placement_balance,
            "salud_SelfPayBalance": round(a.balance, 2), "salud_SelfPayReceipts": round(a.receipts_total, 2),
            "salud_SelfPayAdjustments": round(a.adj_total, 2),
            "salud_SelfPayBalanceDate": a.last_txn or a.placement,
            "salud_OnPaymentPlan": S.YES if (a.on_plan and a.balance > 0) else S.NO,
            "salud_EpicPaymentPlanNumberofPayments": float(a.plan_payments) if a.plan_amount else None,
            "salud_EpicPaymentPlanPaymentsMissed": float(a.plan_missed) if a.plan_amount else None,
            "salud_EpicPaymentPlanLastPaymentDate": a.plan_last_payment,
            "salud_StatementCount": len(stmts),
            "salud_FirstStatementDate": stmts[0] if len(stmts) else None,
            "salud_LastStatementDate": stmts[-1] if len(stmts) else None,
            "salud_LastCallDate": max((t for t in (a.last_attempt, a.last_connect) if t is not None), default=None),
            "salud_LastAccountContact": a.last_connect,
            "salud_LastPatientPaymentFromTxnsDate": a.last_patient_payment,
            "salud_LastUpdatedfromClientSystem": (w.snapshot - pd.Timedelta(days=int(rng.integers(0, 4)))).normalize(),
        })
    return rows


def generate(cfg: GeneratorConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    history_start = pd.Timestamp(cfg.history_start)
    cutoff = pd.Timestamp(cfg.prediction_cutoff)
    window_end = cutoff + pd.Timedelta(days=cfg.outcome_horizon_days)
    w = World(cfg=cfg, rng=rng, cutoff=cutoff, window_end=window_end,
              snapshot=window_end + pd.Timedelta(days=1))

    build_agents(w)
    build_guarantors_and_accounts(w, history_start)
    for g in w.guarantors:
        simulate_history(w, g)
    glookup = {g.gid: g for g in w.guarantors}
    simulate_outcomes(w, glookup)
    _add_filter_traps(w)

    def frame(rows, table):
        return pd.DataFrame(rows, columns=S.COLUMNS[table])

    tables = {
        S.GUARANTOR: frame([{"salud_gphpvguarantorId": g.gid, "salud_DoNotCall": S.YES if g.do_not_call else S.NO}
                            for g in w.guarantors], S.GUARANTOR),
        S.ACCOUNTS: frame(worklist_rows(w), S.ACCOUNTS),
        S.CONTACTS: frame(w.contacts, S.CONTACTS),
        S.BRIDGE: frame(w.bridge, S.BRIDGE).drop_duplicates(),
        S.RECEIPTS: frame(w.receipts, S.RECEIPTS),
        S.QA_REVIEW: frame(w.qa, S.QA_REVIEW),
        "truth_guarantors": pd.DataFrame([g.__dict__ for g in w.guarantors]),
        "truth_agents": pd.DataFrame([{"agent": a.name, "skill": a.skill, "warmth": a.warmth,
                                       **{f"p_{k}": v for k, v in a.item_p.items()}} for a in w.agents]),
        "truth_calls": pd.DataFrame(w.call_truth),
        "truth_accounts": pd.DataFrame(w.account_truth),
    }
    return tables
