"""LLM-written transcripts from planted call truth, via any OpenAI-compatible endpoint.

Intended for a self-hosted model (Ollama, vLLM, llama.cpp server) inside the approved
environment. Synthetic calls contain no PHI, but keep generation local anyway so the same
setup can later score real calls.

Use a different model family to *generate* calls than the one used to *score* them in
Phase 4, otherwise the LLM scorer is graded on its own writing style.
"""

from __future__ import annotations

import re

from .. import schema as S
from ..llm_client import LLMEndpoint

BEHAVIOR = {
    "greeting": ("opens with the full greeting script: name, department, recorded-line notice",
                 "skips the greeting script (no name or department, abrupt opening)"),
    "culture": ("stays courteous, patient and empathetic; never pressures",
                "sounds impatient or dismissive, or pressures the caller to pay immediately"),
    "capturing_data": ("confirms or updates the callback number and mailing address",
                       "never confirms contact details"),
    "understanding_needs": ("restates the caller's concern and offers options that fit it "
                            "(financial assistance or a plan when there is hardship)",
                            "ignores the actual concern and pushes for payment"),
    "correct_info": ("explains the balance and next steps accurately",
                     "gives vague or contradictory information"),
    "first_contact_resolution": ("fully resolves the caller's issue on this call",
                                 "leaves the issue unresolved (callback, transfer, or 'call back later')"),
    "compliance": ("verifies name and date of birth BEFORE discussing any balance",
                   "discusses the balance before verifying identity, or with an unverified third party"),
}


def _ending(call: dict) -> str:
    """The planted outcome in the same words the QA prompt uses to define it."""
    from ..qa.targets import _OUTCOME_MAP, OUTCOME_DEFINITIONS
    key = "hung_up" if call["hung_up"] else _OUTCOME_MAP.get(call["true_result"], "non_committal")
    return OUTCOME_DEFINITIONS[key]


def build_prompt(call: dict) -> str:
    """Prompt that pins every planted label the transcript must express."""
    behaviors = "\n".join(
        f"- {BEHAVIOR[k][0] if call[f'qa_{k}'] else BEHAVIOR[k][1]}" for k in BEHAVIOR
    )
    caller = {"patient": "the patient", "non_patient_guarantor": "a family member responsible for the bill",
              "insurance": "an insurance company representative"}[call["caller_type"]]
    return f"""Write a realistic phone call transcript between a hospital billing agent and {caller}.

Setting: US hospital patient billing office (healthcare revenue cycle). Direction: {call['direction'].replace('_', ' ')}.
Caller's situation: {call['scenario'].replace('_', ' ')}. Open balance discussed: ${call['total_balance_discussed']:,.2f}.
Caller mood: {call['sentiment_label']}. Language preference: {call['language']}.
How the call ends: {_ending(call)}
{f"Promised payment date: {call['promise_date']}." if isinstance(call.get('promise_date'), str) and call['promise_date'] else ''}

The agent MUST show exactly these behaviors:
{behaviors}

Format rules:
- One turn per paragraph as 'Speaker: text'. Speakers: Agent, Patient, Patient Representative,
  Insurance Representative, Automated System (inferred).
- Inbound calls start with an Automated System recording notice.
- Never invent names, dates of birth, phone numbers, addresses or account numbers. Use tokens such as
  [REDACTED NAME], [REDACTED DOB RESPONSE], [REDACTED PHONE NUMBER], [REDACTED SERVICE DATE].
- Natural speech with occasional fillers (um, uh), 10-20 turns, under 700 words. Output only the transcript."""


def prompt_record(call: dict) -> dict:
    return {"contact_id": call["contact_id"], "prompt": build_prompt(call),
            "labels": {k: bool(call[f"qa_{k}"]) for k in S.QA_KEYS},
            "sentiment_label": call["sentiment_label"], "true_result": call["true_result"]}


def clean_generation(raw: str) -> str:
    """Strip reasoning blocks, code fences and chatter around the transcript."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I)
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text.strip(), flags=re.M)
    lines = text.strip().splitlines()
    first = next((i for i, ln in enumerate(lines) if re.match(r"^[A-Z][^:\n]{0,44}:\s", ln)), 0)
    return "\n".join(lines[first:]).strip()


def generate_with_endpoint(prompt: str, endpoint: LLMEndpoint, temperature: float = 0.8) -> str:
    """One transcript; retried once if the model returns fewer than 6 speaker turns."""
    for _ in range(2):
        text = clean_generation(endpoint.chat([{"role": "user", "content": prompt}], temperature=temperature,
                                              max_tokens=950))  # Groq free tier: 1,000 output tokens/min
        if sum(1 for b in re.split(r"\n\s*\n", text) if re.match(r"^[^:\n]{1,45}:\s", b.strip())) >= 6:
            return text
    raise RuntimeError("model returned too few speaker turns")
