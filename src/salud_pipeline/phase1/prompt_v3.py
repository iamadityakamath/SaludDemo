"""Prompt assist-v3: a TRIAL version, not used by the dashboard.

Decision rules drawn from the v2 vs manual-tag review (2026-10-28). It lost to v2 (0.84 vs 0.94 agreement with the
team's tags), so v2 stays live; v3 is kept so the trial (`salud-phase1 trial --version assist-v3`) can be reproduced.
The live definitions are in checklist.py.
"""

ITEM_DEFINITIONS = {
    "greeting": "Agent opens with the approved script: identifies themself and the department "
                "(and the recorded-line notice on outbound calls).",
    "culture": "Agent's tone is courteous, patient and empathetic throughout: no rude, dismissive, blaming or "
               "sarcastic remarks, and no ultimatums. Calmly asking for payment or stating a real consequence "
               "(e.g. collections) is not by itself a failure.",
    "capturing_data": "Agent confirms or updates key contact data (callback number, mailing address).",
    "understanding_needs": "Agent identifies the caller's actual concern and responds to it with fitting "
                           "options (e.g. financial assistance or a plan for hardship) instead of a generic push to pay.",
    "correct_info": "What the agent says about the balance, account status, insurance and next steps is clear and "
                    "consistent: no self-contradiction, no hedging on terms they set ('I'm not sure'), no vague "
                    "next steps. N/A only if the agent gave no account information at all.",
    "first_contact_resolution": "Everything the caller needed from this call was done on it and they accepted it: "
                                "no transfer, callback or 'call back later', and their question was not left open. "
                                "A submitted request still pending outside (insurer review, assistance decision) "
                                "can still be Yes.",
    "compliance": "Agent verifies identity (e.g. name and date of birth) BEFORE disclosing any account "
                  "or balance information, and does not disclose to unverified third parties.",
}

OUTCOME_DEFINITIONS = {
    "hung_up": "The caller disconnected abruptly before saying whether they would pay (no commitment, refusal or "
               "dispute yet). The agent ending the call is not hung_up.",
    "payment_plan": "A payment plan (recurring installments) was set up or changed on this call, even if the first "
                    "installment was taken today.",
    "paid": "The caller made a one-time payment (full or partial) during this call, with no payment plan set up. "
            "A card or bank (ACH) payment authorized and processed on the call counts, even if it clears later.",
    "promise_to_pay": "No money was taken and no plan was set up, but the caller committed to pay a specific amount or "
                      "on a specific date or timeframe later.",
    "insurance_followup": "The resolution is to pursue the patient's insurer: a claim review, re-bill or insurance appeal "
                          "was requested, or the caller was told to contact their insurer before paying. A dispute of "
                          "the hospital's own charges sent to an internal review is refused_or_disputed.",
    "refused_or_disputed": "The caller said they will not pay, or will not pay until a condition is met (e.g. 'not until "
                           "I see the breakdown'), or disputed the balance or charges, without committing to pay.",
    "transferred": "The call was handed to another department, provider or person before resolution "
                   "(including insurance-representative calls routed elsewhere).",
    "non_committal": "Balance discussed but the caller made no commitment, refusal or dispute (e.g. 'I'll think about "
                     "it'), including calls that end with a financial-assistance application sent or pending.",
}
OUTCOME_PRECEDENCE = ["payment_plan", "paid", "promise_to_pay", "insurance_followup", "refused_or_disputed",
                      "transferred", "hung_up", "non_committal"]

RULES = """DECISION RULES (apply these before the definitions above):
- Transcripts are redacted: asterisks and [REDACTED ...] hide words. A redacted question answered with a redacted
  reply (e.g. "And your ******?" then "[REDACTED DOB RESPONSE]") counts as identity verification. Never answer
  "No" only because redaction hides what was said.
- culture: answer "No" only if you can quote a rude, dismissive, blaming, sarcastic or impatient remark, or an
  ultimatum ("pay today or ..."). Repeating the balance, asking for payment, or calmly stating a real consequence
  (late fees, collections, delinquency) is not a "No" when the tone stays polite.
- correct_info: judge what the agent did say, not what they failed to explain (an unanswered question belongs to
  understanding_needs). "No" when they contradict themselves, hedge on terms they are setting, or give vague or
  unreliable next steps. Explaining the account's status (e.g. bad debt, legal status, a pending claim) is
  information, so the item is not "N/A".
- first_contact_resolution: "Yes" when the caller's own concern was answered and every request they made was
  carried out on this call, so they have nothing left to chase. A promise to pay does not make it "Yes" if the
  caller's question about the charge was left open. A request submitted on the call that now waits on the insurer
  or an assistance review can still be "Yes"."""

SENTIMENT_NOTE = """  The caller's overall tone, weighting how they end the call. "positive": cooperative and clearly appreciative or
  relieved by the end, even if worried earlier. "negative": angry, hostile or distressed for much of the call and
  not won over; grudging acceptance ("I guess that's better than expected") stays "negative". "neutral":
  businesslike, or mild frustration that settles into cooperation."""

OUTCOME_NOTE = """A caller who announces "I'm hanging up" after refusing or disputing is "refused_or_disputed", not "hung_up"."""
