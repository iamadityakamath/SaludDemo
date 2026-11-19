from salud_app.sentiment.model import from_probs
from salud_app.sentiment.scoring import label, role, score_call


class FakeModel:
    """Scores 'good' text positive, 'bad' text negative, anything else neutral."""
    def score(self, texts):
        return [from_probs(*((0.05, 0.1, 0.85) if "good" in t else (0.85, 0.1, 0.05) if "bad" in t else (0.1, 0.8, 0.1)))
                for t in texts]


def test_roles_and_labels():
    assert role("Agent (inferred)") == "agent" and role("Automated System") == "skip" and role("Note") == "skip"
    assert role("Patient Representative") == "caller"
    assert label(0.3, 0.2) == "positive" and label(-0.3, 0.2) == "negative"
    assert label(0.0, 0.6) == "mixed" and label(0.0, 0.1) == "neutral"


def test_score_call_arc():
    text = ("Automated System: This call is recorded.\n\nAgent: hello there how are you\n\n"
            "Patient: this is bad bad bad really\n\nAgent: let me help you now\n\nPatient: that is good good good thanks")
    rows, s = score_call(text, FakeModel())
    assert [r["turn"] for r in rows] == [2, 3, 4, 5]                     # the system line isn't scored
    assert s["call_turns"] == 5 and s["caller_turns"] == 2
    assert s["caller_start"] < 0 < s["caller_end"] and s["caller_shift"] > 1
    assert s["caller_end_label"] == "positive" and s["caller_min_turn"] == 3 and s["caller_max_turn"] == 5
