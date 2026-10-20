import json

import pytest

from salud_pipeline.llm_client import LLMEndpoint
from salud_pipeline.phase1 import analyze
from salud_pipeline.phase1.checklist import TRANSCRIPT_ITEMS
from salud_pipeline.phase1.prompt import PROMPT_VERSION, CallAssistant

GOOD = {"ok": True, "checklist": {k: {"answer": "Yes", "reason": "r", "evidence": [{"turn": 1, "quote": "Hello"}]}
                                  for k in TRANSCRIPT_ITEMS},
        "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": "paid", "evidence": []},
        "summary": "s", "coaching": [{"item": "greeting", "tip": "t", "turn": 1}]}


class FakeAssistant(CallAssistant):
    """Real cache handling; the LLM call is replaced by a stub that can fail."""

    def __init__(self, cache_dir, model, fail=False):
        super().__init__(LLMEndpoint(base_url="http://127.0.0.1:9/v1", model=model, api_key=None), cache_dir)
        self.requests, self.fail = 0, fail

    def review(self, text):
        if self.fail:
            raise RuntimeError("LLM endpoint returned HTTP 429: daily token limit")
        self.requests += 1
        self._path(text).parent.mkdir(parents=True, exist_ok=True)
        self._path(text).write_text(json.dumps(GOOD))
        return GOOD


def _add_temp_call(store):
    with store.pool.connection() as c:
        c.execute("INSERT INTO phase1.calls (call_id, source, file_id, transcript) VALUES (%s, 'synthetic', %s, %s)",
                  (store.test_call, store.test_call.split("/")[1], "Agent: Hello, this is billing.\n\nPatient: Hi."))


def test_analyze_adds_only_missing_calls_and_never_changes_existing(store, tmp_path):
    _add_temp_call(store)
    model = store.ai_info()["model"]
    before = store.ai_analyses()
    c = analyze.run(store, FakeAssistant(tmp_path, model), progress=False)
    after = store.ai_analyses()
    assert c["new_llm_requests"] == 1 and store.test_call in after
    assert {k: after[k] for k in before} == before                         # existing rows untouched
    with store.pool.connection() as conn:
        n = conn.execute("SELECT count(*) FROM phase1.ai_checklist WHERE call_id = %s", (store.test_call,)).fetchone()
    assert n["count"] == len(TRANSCRIPT_ITEMS)
    assert store.add_ai_analysis(store.test_call, GOOD, model, PROMPT_VERSION) is False   # no overwrite


def test_analyze_refuses_to_mix_models(store, tmp_path):
    _add_temp_call(store)
    with pytest.raises(RuntimeError, match="Nothing was changed"):
        analyze.run(store, FakeAssistant(tmp_path, "some-other-model"), progress=False)
    assert store.test_call not in store.ai_analyses()


def test_analyze_stops_cleanly_on_provider_error(store, tmp_path):
    _add_temp_call(store)
    c = analyze.run(store, FakeAssistant(tmp_path, store.ai_info()["model"], fail=True), progress=False)
    assert c["not_done"] == 1 and "429" in c["stopped_early"] and store.test_call not in store.ai_analyses()
