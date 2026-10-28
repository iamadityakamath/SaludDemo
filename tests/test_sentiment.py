import importlib.util

import pytest

from salud_pipeline.sentiment.aggregate import as_tag, label, summarize
from salud_pipeline.sentiment.model import from_probs
from salud_pipeline.sentiment.results import score_and_save, score_call, score_calls
from salud_pipeline.sentiment.turns import role, scorable_turns, weight

CALL = """Agent: Thank you for calling, this is Dana, how can I help you today?

Patient: I got another bill and I already paid it, this is ridiculous.

Editorial note: speaker inferred.

Agent: I'm sorry about that, let me look into it for you.

Patient: ok

Patient: Oh that fixed it, thank you so much, that's a relief."""


class FakeModel:
    """Scores by keyword so the tests need no download: 'ridiculous' negative, 'thank' positive."""

    def score(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            neg, pos = (0.9, 0.02) if "ridiculous" in t else (0.02, 0.9) if "thank" in t else (0.05, 0.05)
            out.append(from_probs(neg, 1 - neg - pos, pos))
        return out


def test_speakers_map_to_roles():
    assert role("Agent") == role("Agent (inferred)") == "agent"
    for s in ("Patient", "Patient Representative", "Patient / Account Holder", "Insurance Representative"):
        assert role(s) == "caller"
    for s in ("Editorial note", "Automated System (inferred)", "Note"):
        assert role(s) == "skip"


def test_turn_numbers_match_the_dashboard_and_notes_are_skipped():
    ts = scorable_turns(CALL)
    assert [t["n"] for t in ts] == [1, 2, 4, 5, 6]      # T3 is the editorial note
    assert [t["role"] for t in ts] == ["agent", "caller", "agent", "caller", "caller"]


def test_filler_turns_barely_count_and_long_turns_are_capped():
    assert weight("ok") == pytest.approx(0.2)
    assert weight("word " * 100) == 40


def test_score_and_magnitude_from_probabilities():
    s = from_probs(0.78, 0.20, 0.02)
    assert s["score"] == pytest.approx(-0.76) and s["magnitude"] == pytest.approx(0.80) and s["label"] == "negative"


def test_labels_separate_mixed_from_neutral():
    assert label(0.6, 0.7) == "positive" and label(-0.6, 0.7) == "negative"
    assert label(0.0, 0.8) == "mixed" and label(0.0, 0.1) == "neutral"
    assert as_tag("mixed") == "neutral"


def test_summary_captures_the_arc():
    turns = [{"n": i + 1, "score": s, "magnitude": abs(s), "weight": 10, "words": 10}
             for i, s in enumerate([-0.8, -0.6, 0.0, 0.2, 0.7, 0.8])]
    s = summarize(turns, total_turns=6)
    assert s["start"] < 0 < s["end"] and s["shift"] == pytest.approx(s["end"] - s["start"])
    assert s["min_turn"] == 1 and s["min_position"] == 0 and s["max_turn"] == 6
    assert summarize([], 0) == {"turns": 0}


def test_score_call_gives_turn_rows_and_a_summary():
    rows, summary = score_call(CALL, FakeModel())
    assert [r["turn"] for r in rows] == [1, 2, 4, 5, 6] and "text" not in rows[0]   # numbers only, never text
    assert summary["call_turns"] == 6 and summary["caller_turns"] == 3 and summary["agent_turns"] == 2
    assert summary["caller_start"] < 0 < summary["caller_end"] and summary["caller_end_label"] == "positive"
    assert summary["caller_min_turn"] == 2                          # "ok" (T5) can't be the low point
    turns, calls = score_calls({"real/a.txt": CALL}, FakeModel())
    assert len(turns) == 5 and calls.iloc[0]["source"] == "real"


def test_sentiment_is_saved_and_read_back(store):
    cid = store.add_new_call(CALL, store.test_rater)
    try:
        score_and_save(store, cid, CALL, FakeModel(), "fake")
        got = store.sentiment(cid)
        assert got["model"] == "fake" and [t["turn"] for t in got["turns"]] == [1, 2, 4, 5, 6]
        assert got["call"]["caller_end_label"] == "positive"
        score_and_save(store, cid, CALL, FakeModel(), "fake-2")         # a rescore replaces, not duplicates
        assert store.sentiment(cid)["model"] == "fake-2" and len(store.sentiment(cid)["turns"]) == 5
        assert store.sentiment("synthetic/nope.txt") is None
    finally:
        _drop(store, cid)


def test_new_calls_stay_out_of_tagging_and_comparisons(store):
    cid = store.add_new_call(CALL, store.test_rater)
    try:
        assert cid.startswith("new/") and store.transcript(cid) == CALL.strip()
        assert cid not in {c["call_id"] for c in store.calls()}
        assert cid in {c["call_id"] for c in store.calls(include_new=True)}
        row = next(c for c in store.analysis_list() if c["call_id"] == cid)
        assert row["source"] == "new" and row["added_by"] == store.test_rater and not row["analyzed"]
        with pytest.raises(ValueError):
            store.add_new_call("   ", store.test_rater)
    finally:
        _drop(store, cid)


def _drop(store, call_id):
    with store.pool.connection() as c:
        for t in ("ai_coaching", "ai_checklist", "ai_analyses"):
            c.execute(f"DELETE FROM phase1.{t} WHERE call_id = %s", (call_id,))
        c.execute("DELETE FROM phase1.calls WHERE call_id = %s", (call_id,))   # sentiment rows cascade


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None, reason="onnxruntime not installed")
def test_real_model_scores_obvious_sentences():
    from salud_pipeline.sentiment.model import SentimentModel

    try:
        m = SentimentModel()
    except OSError as e:   # not downloaded and offline
        pytest.skip(f"model unavailable: {e}")
    neg, pos = m.score(["This is ridiculous, I already paid this bill twice.", "Thank you so much, that really helps!"])
    assert neg["score"] < -0.5 < 0.5 < pos["score"]


def test_analyze_page_scores_a_pasted_call_without_the_llm(store, monkeypatch):
    import json
    import threading
    import urllib.request

    from salud_pipeline.phase1 import live
    from salud_pipeline.phase1.dashboard import serve

    monkeypatch.setattr(live, "llm_ready", lambda: False)   # never call a real LLM from tests
    monkeypatch.setattr("salud_pipeline.sentiment.model.shared_model", lambda: FakeModel())
    httpd = serve(store.test_rater, port=0, store=store)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}"

    def post(path, body):
        req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    cid = None
    try:
        r = post("/api/analyze/new", {"transcript": CALL})
        cid = r["id"]
        assert r["sentiment"]["call"]["caller_end_label"] == "positive" and r["analysis"] == {"none": True}
        assert "isn't set up" in r["notes"][0]
        listed = json.loads(urllib.request.urlopen(url + "/api/analyze/list").read())
        assert any(c["call_id"] == cid and c["scored"] for c in listed["calls"]) and listed["llm"] is False
        assert cid not in {c["call_id"] for c in store.calls()}            # pasted calls stay out of comparisons
        again = post("/api/analyze/run", {"id": cid})                  # existing results load, nothing redone
        assert again["sentiment"] == r["sentiment"]
    finally:
        httpd.shutdown()
        if cid:
            _drop(store, cid)
