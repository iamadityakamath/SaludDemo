import importlib.util

import pytest

from salud_pipeline.sentiment.aggregate import as_tag, label, summarize
from salud_pipeline.sentiment.model import from_probs
from salud_pipeline.sentiment.results import Results, save, score_calls
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


def test_score_calls_and_results_roundtrip(tmp_path):
    turns, calls = score_calls({"real/a.txt": CALL}, FakeModel())
    assert len(turns) == 5 and "text" not in turns.columns          # numbers only, never transcript text
    row = calls.iloc[0]
    assert row["source"] == "real" and row["caller_turns"] == 3 and row["agent_turns"] == 2
    assert row["caller_start"] < 0 < row["caller_end"]
    assert row["caller_min_turn"] == 2                              # "ok" (T5) can't be the low point

    res = Results(tmp_path)
    assert res.for_call("real/a.txt") is None                       # never run: nothing to show
    save(turns, calls, {"model": "fake"}, tmp_path)
    got = res.for_call("real/a.txt")
    assert got["model"] == "fake" and [t["turn"] for t in got["turns"]] == [1, 2, 4, 5, 6]
    assert got["call"]["caller_end_label"] == "positive"


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="optional [sentiment] install")
def test_real_model_scores_obvious_sentences():
    from salud_pipeline.sentiment.model import SentimentModel

    try:
        m = SentimentModel(device="cpu")
    except OSError as e:   # not downloaded and offline
        pytest.skip(f"model unavailable: {e}")
    neg, pos = m.score(["This is ridiculous, I already paid this bill twice.", "Thank you so much, that really helps!"])
    assert neg["score"] < -0.5 < 0.5 < pos["score"]
