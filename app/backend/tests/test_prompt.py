from salud_app.phase1.checklist import TRANSCRIPT_ITEMS, call_quality
from salud_app.phase1.prompt import build_prompt, parse, split_turns

TEXT = "Agent: Thank you for calling, this is Dana.\n\nPatient: My bill is wrong.\n\nunlabelled block"


def test_split_turns():
    t = split_turns(TEXT)
    assert [x["speaker"] for x in t] == ["Agent", "Patient", "Note"] and [x["n"] for x in t] == [1, 2, 3]


def test_prompt_numbers_turns():
    p = build_prompt(split_turns(TEXT))
    assert "[T1] Agent: Thank you for calling" in p and "[T2] Patient: My bill is wrong." in p


def test_parse_keeps_only_real_quotes_and_fixes_turn_numbers():
    turns = split_turns(TEXT)
    raw = '{"checklist": {"greeting": {"answer": "yes", "reason": "ok", "evidence": [' \
          '{"turn": 1, "quote": "this is Dana"}, {"turn": 1, "quote": "My bill is wrong"}, {"turn": 9, "quote": "x"}]}},' \
          '"outcome": {"label": "Transferred"}, "summary": "s", "coaching": [{"item": "nope", "tip": "Be kind", "turn": 99}]}'
    out = parse(raw, turns)
    ev = out["checklist"]["greeting"]["evidence"]
    assert out["checklist"]["greeting"]["answer"] == "Yes"
    assert ev == [{"turn": 1, "quote": "this is Dana"}, {"turn": 2, "quote": "My bill is wrong"}]   # moved to T2
    assert out["evidence_check"] == {"kept": 2, "dropped": 1}
    assert out["outcome"]["label"] == "transferred"
    assert out["coaching"] == [{"item": "general", "tip": "Be kind", "turn": None}]
    assert set(out["checklist"]) == set(TRANSCRIPT_ITEMS)


def test_parse_garbage_is_not_ok():
    assert parse("not json", split_turns(TEXT))["ok"] is False


def test_call_quality():
    assert call_quality({"greeting": "Yes", "culture": "No", "compliance": "N/A"}) == 0.5
    assert call_quality({}) is None
