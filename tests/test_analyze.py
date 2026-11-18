import json

import pytest

from salud_pipeline.llm_client import LLMEndpoint
from salud_pipeline.phase1 import analyze
from salud_pipeline.phase1.checklist import TRANSCRIPT_ITEMS
from salud_pipeline.phase1.prompt import CallAssistant

GOOD = {"ok": True, "checklist": {k: {"answer": "Yes", "reason": "r", "evidence": [{"turn": 1, "quote": "Hello"}]}
                                  for k in TRANSCRIPT_ITEMS},
        "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": "paid", "evidence": []},
        "summary": "s", "coaching": [{"item": "greeting", "tip": "t", "turn": 1}]}


class FakeAssistant(CallAssistant):
    """Real cache handling; the LLM call is replaced by a stub that can fail."""

    def __init__(self, cache_dir, model, fail=False, version=None):
        super().__init__(LLMEndpoint(base_url="http://127.0.0.1:9/v1", model=model, api_key=None), cache_dir,
                         *([version] if version else []))
        self.requests, self.fail = 0, fail

    def review(self, text):
        if self.fail:
            raise RuntimeError("LLM endpoint returned HTTP 429: daily token limit")
        self.requests += 1
        self._path(text).parent.mkdir(parents=True, exist_ok=True)
        self._path(text).write_text(json.dumps(GOOD))
        return GOOD


def _add_temp_call(store, monkeypatch):
    # Run as the prompt version the test branch already holds, so a PROMPT_VERSION bump doesn't trip the mix guard.
    # (An empty test branch has no version yet: keep the code's own.)
    monkeypatch.setattr(analyze, "PROMPT_VERSION", store.ai_info()["prompt_version"] or analyze.PROMPT_VERSION)
    with store.pool.connection() as c:
        c.execute("INSERT INTO phase1.calls (call_id, source, file_id, transcript) VALUES (%s, 'synthetic', %s, %s)",
                  (store.test_call, store.test_call.split("/")[1], "Agent: Hello, this is billing.\n\nPatient: Hi."))


def test_analyze_adds_only_missing_calls_and_never_changes_existing(store, tmp_path, monkeypatch):
    _add_temp_call(store, monkeypatch)
    model = store.ai_info()["model"]
    before = store.ai_analyses()
    c = analyze.run(store, FakeAssistant(tmp_path, model), progress=False)
    after = store.ai_analyses()
    assert c["new_llm_requests"] == 1 and store.test_call in after
    assert {k: after[k] for k in before} == before                         # existing rows untouched
    with store.pool.connection() as conn:
        n = conn.execute("SELECT count(*) FROM phase1.ai_checklist WHERE call_id = %s", (store.test_call,)).fetchone()
    assert n["count"] == len(TRANSCRIPT_ITEMS)
    assert store.add_ai_analysis(store.test_call, GOOD, model, analyze.PROMPT_VERSION) is False   # no overwrite


def test_analyze_refuses_to_mix_models(store, tmp_path, monkeypatch):
    _add_temp_call(store, monkeypatch)
    with pytest.raises(RuntimeError, match="Nothing was changed"):
        analyze.run(store, FakeAssistant(tmp_path, "some-other-model"), progress=False)
    assert store.test_call not in store.ai_analyses()


def test_analyze_stops_cleanly_on_provider_error(store, tmp_path, monkeypatch):
    _add_temp_call(store, monkeypatch)
    c = analyze.run(store, FakeAssistant(tmp_path, store.ai_info()["model"], fail=True), progress=False)
    assert c["not_done"] == 1 and "429" in c["stopped_early"] and store.test_call not in store.ai_analyses()


def test_trial_version_is_kept_apart_from_the_dashboard_version(store, tmp_path, monkeypatch):
    from salud_pipeline.phase1 import prompt, trial

    _add_temp_call(store, monkeypatch)
    monkeypatch.setitem(prompt.VERSIONS, "pytest-trial", prompt.VERSIONS["assist-v3"])
    model = store.ai_info()["model"] or "fake-model"
    before = store.ai_analyses()
    out = tmp_path / "pytest-trial.json"
    try:
        trial.run(store, FakeAssistant(tmp_path, model, version="pytest-trial"), out, source="synthetic",
                  progress=False)
        assert store.test_call in store.ai_versions(model, "pytest-trial")
        assert store.ai_analyses() == before                                     # dashboard version untouched
        assert store.test_call in json.loads(out.read_text())["calls"]
        assert store.add_ai_version(store.test_call, GOOD, model, "pytest-trial") is False   # no overwrite
    finally:                                                                     # clean up even if an assert fails
        with store.pool.connection() as conn:
            conn.execute("DELETE FROM phase1.ai_analysis_versions WHERE prompt_version = 'pytest-trial'")


def test_live_prompt_is_v2_and_v3_differs():
    from salud_pipeline.phase1 import prompt

    turns = prompt.split_turns("Agent: Hello.\n\nPatient: Hi.")
    assert prompt.PROMPT_VERSION == "assist-v2"
    assert "DECISION RULES" not in prompt.build_prompt(turns)
    assert "DECISION RULES" in prompt.build_prompt(turns, "assist-v3")
