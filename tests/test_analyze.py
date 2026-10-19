import json

from salud_pipeline.llm_client import LLMEndpoint
from salud_pipeline.phase1 import analyze
from salud_pipeline.phase1.calls import CallSource
from salud_pipeline.phase1.checklist import TRANSCRIPT_ITEMS, clean_transcript
from salud_pipeline.phase1.prompt import PROMPT_VERSION, CallAssistant

GOOD = {"ok": True, "checklist": {k: {"answer": "Yes", "reason": "r", "evidence": []} for k in TRANSCRIPT_ITEMS},
        "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": "paid", "evidence": []},
        "summary": "s", "coaching": []}


class FakeAssistant(CallAssistant):
    """Real cache handling; the LLM call is replaced by a stub that can fail after N requests."""

    def __init__(self, cache_dir, fail_after=None):
        super().__init__(LLMEndpoint(base_url="http://127.0.0.1:9/v1", model="fake", api_key=None), cache_dir)
        self.requests, self.fail_after = 0, fail_after

    def review(self, text, force=False):
        if self.cached(text) is not None:
            return self.cached(text)
        if self.fail_after is not None and self.requests >= self.fail_after:
            raise RuntimeError("LLM endpoint returned HTTP 429: daily token limit")
        self.requests += 1
        self._path(text).parent.mkdir(parents=True, exist_ok=True)
        self._path(text).write_text(json.dumps(GOOD))
        return GOOD


def sources(tmp_path, n=4):
    d = tmp_path / "calls"
    d.mkdir()
    for i in range(n):
        (d / f"c{i}.txt").write_text(f"Agent: Hello, call {i}.\n\nPatient: Hi.")
    return [CallSource("real", "Real", d, tmp_path / "tags.csv", real=True)]


def test_analyze_writes_one_json_keyed_by_call_id(tmp_path):
    out = tmp_path / "ai.json"
    c = analyze.run(sources(tmp_path), FakeAssistant(tmp_path / "cache"), out, progress=False)
    data = json.loads(out.read_text())
    assert c["new_llm_requests"] == 4 and c["in_file"] == 4
    assert data["model"] == "fake" and data["prompt_version"] == PROMPT_VERSION
    assert set(data["calls"]) == {f"real/c{i}.txt" for i in range(4)}
    assert data["calls"]["real/c0.txt"]["analysis"]["checklist"]["greeting"]["answer"] == "Yes"


def test_analyze_reuses_cache_respects_limit_and_resumes(tmp_path):
    src, out, cache = sources(tmp_path), tmp_path / "ai.json", tmp_path / "cache"
    pre = FakeAssistant(cache)
    pre.review(clean_transcript((src[0].transcripts_dir / "c0.txt").read_text()))  # analyzed on the dashboard

    c = analyze.run(src, FakeAssistant(cache), out, limit=1, progress=False)
    assert (c["from_cache"], c["new_llm_requests"], c["not_done"], c["in_file"]) == (1, 1, 2, 2)

    c = analyze.run(src, FakeAssistant(cache), out, progress=False)           # next run picks up the rest
    assert (c["already_in_file"], c["new_llm_requests"], c["in_file"]) == (2, 2, 4)


def test_analyze_stops_on_provider_error_and_keeps_progress(tmp_path):
    src, out = sources(tmp_path), tmp_path / "ai.json"
    c = analyze.run(src, FakeAssistant(tmp_path / "cache", fail_after=2), out, progress=False)
    assert c["in_file"] == 2 and c["not_done"] == 2 and "429" in c["stopped_early"]
    assert len(json.loads(out.read_text())["calls"]) == 2
