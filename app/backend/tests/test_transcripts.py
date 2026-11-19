from salud_app.repositories.calls import attach_turn_sentiment
from salud_app.transcripts import parse_turns, role_of

TEXT = """Editorial note: synthetic call, fictional.
Automated System: This call may be recorded.
Agent: Thank you for calling, how can I help?
Patient Representative: I'm calling about my mother's bill.
It arrived last week.
Agent (inferred): Let me look that up.
Insurance Representative: We need the claim number.
Unknown Speaker: Hello?"""


def test_parse_turns_labels_roles_and_continuations():
    turns = parse_turns(TEXT)
    assert [t["role"] for t in turns] == ["system", "agent", "caller", "agent", "caller", "unknown"]
    assert turns[2]["text"] == "I'm calling about my mother's bill. It arrived last week."     # continuation joined
    assert [t["turn"] for t in turns] == [1, 2, 3, 4, 5, 6]
    assert all("Editorial" not in t["speaker"] for t in turns)                                 # note skipped


def test_role_of():
    assert role_of("Patient / Account Holder") == "caller"
    assert role_of("Caller") == "caller"
    assert role_of("Note") is None          # a colon in normal text is not a speaker
    assert parse_turns(None) == [] and parse_turns("") == []


def test_turn_scores_attach_only_when_aligned():
    turns = parse_turns(TEXT)
    scores = [{"role": r, "score": 0.1 * i, "label": "neutral"} for i, r in enumerate(["agent", "caller", "agent", "caller", "agent"])]
    out = attach_turn_sentiment([dict(t) for t in turns], scores)
    assert out[0].get("sentiment") is None                                                     # system line: no score
    assert [t.get("sentiment") for t in out[1:]] == [0.0, 0.1, 0.2, 0.30000000000000004, 0.4]
    wrong = attach_turn_sentiment([dict(t) for t in turns], scores[:-1])                       # count differs
    assert all(t.get("sentiment") is None for t in wrong)
    flipped = attach_turn_sentiment([dict(t) for t in turns], [dict(s, role="caller") for s in scores])
    assert all(t.get("sentiment") is None for t in flipped)                                    # roles differ
