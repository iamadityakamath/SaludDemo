"""Agent notes and QA reviewer comments. All text is invented; nothing is copied from source data."""

from __future__ import annotations

import numpy as np

SCENARIO_NOTE = {
    "wants_to_pay": "Pt called to pay balance",
    "billing_question": "Pt had questions about statement",
    "insurance_issue": "Pt states ins should have covered",
    "hardship": "Pt reports financial hardship",
    "dispute": "Pt disputes balance",
    "third_party": "Ins rep called re: pt acct",
}

RESULT_NOTE = {
    "paid_guarantor_full": "paid guar bal in full over phone",
    "paid_account_full": "paid acct in full over phone",
    "partial_payment": "made partial pmt over phone",
    "promised_to_pay": "promised to pay by {date}",
    "non_committal": "would not commit to pmt",
    "contested_balance": "contests bal, req itemized",
    "would_not_pay": "states will not pay",
    "pursued_insurance": "advised to f/u w ins, acct sent for ins review",
    "na_transferred": "transferred",
    "paid_plan_full": "pmt plan set up, first pmt taken",
}

ACTION_NOTE = {
    "provided_fa_options": "reviewed FA options",
    "fa_paperwork_sent": "FA app mailed",
    "set_up_plan": "PP set up",
    "provided_itemized": "itemized bill queued",
    "requested_insurance_review": "ins review requested",
    "updated_contact_info": "updated contact info",
    "shared_balance": "verified and shared bal",
}

_TYPOS = {"patient": "pateint", "balance": "balence", "payment": "paymnet", "called": "caled",
          "states": "staes", "review": "reveiw", "mailed": "maild", "requested": "requsted"}


def agent_note(rng: np.random.Generator, scenario: str, result: str | None, actions: list[str],
               grammar_ok: bool, promise_date: str | None) -> str:
    parts = [SCENARIO_NOTE[scenario]]
    if result:
        parts.append(RESULT_NOTE.get(result, result.replace("_", " ")).format(date=promise_date or "TBD"))
    parts += [ACTION_NOTE[a] for a in actions if a in ACTION_NOTE]
    note = ". ".join(parts) + "."
    note = note.replace("Pt ", "Patient ", 1) if rng.random() < 0.5 else note
    if grammar_ok:
        return note[0].upper() + note[1:]
    # Grammar = No: lower-case run-ons, typos, dropped punctuation.
    words = note.lower().replace(". ", " ").replace(".", "").split()
    words = [_TYPOS.get(w, w) for w in words]
    if len(words) > 4 and rng.random() < 0.5:
        del words[int(rng.integers(1, len(words) - 1))]
    return " ".join(words)


_QA_YES = {
    "greeting": ["Used the full greeting script and gave name.", "Opened with name and department per script."],
    "culture": ["Courteous and patient throughout.", "Respectful tone, showed empathy."],
    "capturing_data": ["Confirmed callback number and updated record.", "Verified and captured contact details."],
    "understanding_needs": ["Restated the caller's concern and asked clarifying questions.",
                            "Identified the real need and offered fitting options."],
    "correct_info": ["Explained the balance and next steps accurately.", "Information given was correct."],
    "first_contact_resolution": ["Issue resolved on this call.", "Caller's question fully handled."],
    "compliance": ["Verified identity before discussing the account.", "Completed verification first."],
    "call_statusing": ["Disposition matches the call.", "Call statused correctly."],
    "grammar": ["Notes are clear and spelled correctly.", "Clean, readable notes."],
}
_QA_NO = {
    "greeting": ["Skipped greeting script.", "No name or department given at open."],
    "culture": ["Interrupted caller and sounded dismissive.", "Pressured the caller to pay immediately."],
    "capturing_data": ["Did not confirm callback number.", "Contact details not verified or updated."],
    "understanding_needs": ["Did not address the caller's actual concern.",
                            "Pushed payment without exploring hardship options."],
    "correct_info": ["Gave vague or conflicting information about the balance.",
                     "Incorrect explanation of insurance responsibility."],
    "first_contact_resolution": ["Caller will need a follow-up call.", "Issue left unresolved."],
    "compliance": ["Discussed balance before verifying identity.", "Verification incomplete."],
    "call_statusing": ["Disposition does not match what happened on the call.", "Wrong collection result recorded."],
    "grammar": ["Notes contain typos and are hard to read.", "Spelling and punctuation errors in notes."],
}
_LESSONS = [
    "Review the verification script before discussing balances.",
    "Offer financial assistance options when caller mentions hardship.",
    "Confirm disposition codes before wrap-up.",
    "Slow down and let the caller finish before responding.",
    "Proofread notes before saving.",
]


def qa_comment(rng: np.random.Generator, key: str, passed: bool) -> str:
    bank = _QA_YES if passed else _QA_NO
    return bank[key][int(rng.integers(len(bank[key])))]


def lessons_learned(rng: np.random.Generator) -> str:
    return _LESSONS[int(rng.integers(len(_LESSONS)))]
