"""Analysis logic with a fake store, model and LLM: no database, no network."""
from datetime import datetime, timedelta, timezone

import pytest

from salud_app.phase1 import analysis


def test_call_status():
    now = datetime.now(timezone.utc)
    assert analysis.call_status({"scored": True, "analyzed": True}, True) == "done"
    assert analysis.call_status({"scored": True, "analyzed": False}, False) == "done"      # no LLM: sentiment is enough
    assert analysis.call_status({"scored": False, "analyzed": False, "added_at": now}, True, now) == "processing"
    old = now - timedelta(hours=1)
    assert analysis.call_status({"scored": True, "analyzed": False, "added_at": old}, True, now) == "incomplete"


@pytest.fixture
def fake(monkeypatch, tmp_path):
    db = {"text": "Agent: hi\n\nPatient: good", "sentiment": None, "review": None, "replaced": 0}
    monkeypatch.setattr(analysis.store, "transcript", lambda cid: db["text"] if cid == "c1" else None)
    monkeypatch.setattr(analysis.store, "sentiment", lambda cid: db["sentiment"])
    monkeypatch.setattr(analysis.store, "ai_analysis", lambda cid: db["review"])
    monkeypatch.setattr(analysis.store, "save_sentiment", lambda cid, rows, s, m: db.update(sentiment={"call": s}))
    monkeypatch.setattr(analysis.store, "add_ai_analysis", lambda cid, r, m, v: db.update(review=r))
    monkeypatch.setattr(analysis.store, "replace_ai_analysis",
                        lambda cid, r, m, v: db.update(review=r, replaced=db["replaced"] + 1))
    monkeypatch.setattr(analysis, "shared_model", lambda: type("M", (), {"score": lambda self, t: [
        {"score": 0.5, "magnitude": 0.5, "label": "positive", "p_negative": 0, "p_neutral": 0.5, "p_positive": 0.5}] * len(t)})())

    class FakeAssistant:
        def __init__(self, ep, cache): pass
        def review(self, text): return {"ok": True, "summary": "reviewed"}
    monkeypatch.setattr(analysis, "CallAssistant", FakeAssistant)
    return db


def test_fills_in_missing_parts(fake, monkeypatch):
    monkeypatch.setattr(analysis.LLMEndpoint, "ready", property(lambda self: True))
    r = analysis.analyze("c1")
    assert r["sentiment"]["call"]["caller_turns"] == 1 and r["analysis"]["summary"] == "reviewed" and not r["notes"]
    analysis.analyze("c1", redo=True)
    assert fake["replaced"] == 1                                                            # redo replaces the review


def test_without_llm_only_sentiment_with_a_note(fake, monkeypatch):
    monkeypatch.setattr(analysis.LLMEndpoint, "ready", property(lambda self: False))
    r = analysis.analyze("c1")
    assert r["analysis"] == {"none": True} and "no LLM API key" in r["notes"][0]


def test_unknown_call(fake):
    with pytest.raises(LookupError):
        analysis.analyze("nope")
