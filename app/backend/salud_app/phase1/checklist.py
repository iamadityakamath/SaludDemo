"""The QA checklist the AI review answers, the call outcomes it chooses from, and the QA score."""

from __future__ import annotations

# Checklist items visible in a transcript (Salud's QA form minus call statusing and grammar, which need CRM notes).
TRANSCRIPT_ITEMS = ["greeting", "culture", "capturing_data", "understanding_needs", "correct_info",
                    "first_contact_resolution", "compliance"]
SENTIMENTS = ["negative", "neutral", "positive"]
OUTCOMES = ["paid", "payment_plan", "promise_to_pay", "non_committal", "refused_or_disputed",
            "insurance_followup", "transferred", "hung_up"]

ITEM_DEFINITIONS = {
    "greeting": "Agent opens with the approved script: identifies themself and the department "
                "(and the recorded-line notice on outbound calls).",
    "culture": "Agent is courteous, patient and empathetic throughout; no pressure, blame or "
               "dismissive remarks.",
    "capturing_data": "Agent confirms or updates key contact data (callback number, mailing address).",
    "understanding_needs": "Agent identifies the caller's actual concern and responds to it with fitting "
                           "options (e.g. financial assistance or a plan for hardship) instead of a generic push to pay.",
    "correct_info": "Information about the balance, insurance and next steps is clear and accurate; "
                    "no vague or contradictory explanations.",
    "first_contact_resolution": "The caller's issue is resolved on this call without a callback, "
                                "transfer or 'call back later'.",
    "compliance": "Agent verifies identity (e.g. name and date of birth) BEFORE disclosing any account "
                  "or balance information, and does not disclose to unverified third parties.",
}

# How a call ends. One label per call; when several fit, the first in OUTCOME_PRECEDENCE wins.
OUTCOME_DEFINITIONS = {
    "hung_up": "The caller disconnected before the call reached a resolution.",
    "payment_plan": "A payment plan (recurring installments) was set up or changed on this call, even if the first "
                    "installment was taken today.",
    "paid": "The caller made a one-time payment (full or partial) during this call, with no payment plan set up.",
    "promise_to_pay": "No money was taken and no plan was set up, but the caller committed to pay a specific amount or "
                      "on a specific date or timeframe later.",
    "insurance_followup": "The resolution is to pursue insurance: a claim review or re-bill was requested, or the caller "
                          "was told to contact their insurer before paying.",
    "refused_or_disputed": "The caller said they will not pay, or disputed the balance or charges, without committing to pay.",
    "transferred": "The call was handed to another department, provider or person before resolution "
                   "(including insurance-representative calls routed elsewhere).",
    "non_committal": "Balance discussed but the caller made no commitment (e.g. 'I'll think about it', 'I'll see').",
}
OUTCOME_PRECEDENCE = ["hung_up", "payment_plan", "paid", "promise_to_pay", "insurance_followup",
                      "refused_or_disputed", "transferred", "non_committal"]


def call_quality(answers: dict) -> float | None:
    """Share of checklist items answered Yes among those answered Yes or No (N/A doesn't count)."""
    vals = [answers.get(k) for k in TRANSCRIPT_ITEMS]
    yes, no = vals.count("Yes"), vals.count("No")
    return round(yes / (yes + no), 3) if yes + no else None
