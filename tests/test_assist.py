import json
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from salud_pipeline import llm_client
from salud_pipeline.llm_client import LLMEndpoint, load_dotenv
from salud_pipeline.qa.assist import CallAssistant, parse, split_turns
from salud_pipeline.qa.tag_server import CallSource, serve
from salud_pipeline.qa.targets import TRANSCRIPT_ITEMS

TEXT = ("Agent: Thank you for calling billing, this is [REDACTED NAME]. How can I help?\n\n"
        "Patient: I got a bill I don't understand.\n\n"
        "Agent: Can I verify your name and date of birth first?\n\n"
        "[Call disconnected by caller]")


def test_split_turns_numbers_and_unlabeled_blocks():
    turns = split_turns(TEXT)
    assert [t["n"] for t in turns] == [1, 2, 3, 4]
    assert turns[0]["speaker"] == "Agent" and turns[3]["speaker"] == "Note"


def test_parse_keeps_real_quotes_and_drops_invented_ones():
    turns = split_turns(TEXT)
    raw = json.dumps({
        "checklist": {
            "greeting": {"answer": "yes", "reason": "Gave name.",
                         "evidence": [{"turn": 1, "quote": "Thank you for calling billing"},
                                      {"turn": 1, "quote": "We value your business"}]},
            "compliance": {"answer": "Yes", "evidence": [{"turn": 1, "quote": "verify your name and date of birth"}]},
            "culture": {"answer": "maybe"},
        },
        "sentiment": {"label": "Neutral", "evidence": [{"turn": "T2", "quote": "I got a bill"}]},
        "outcome": "teleported",
        "summary": "Short billing call.",
        "coaching": [{"item": "capturing_data", "tip": "Confirm the callback number.", "turn": 3},
                     {"item": "made_up", "tip": "Be nice.", "turn": 99}, {"tip": ""}],
    })
    r = parse("Here you go:\n" + raw, turns)
    assert r["checklist"]["greeting"]["answer"] == "Yes"
    assert r["checklist"]["greeting"]["evidence"] == [{"turn": 1, "quote": "Thank you for calling billing"}]
    # Real quote under the wrong turn number is re-pointed to the turn that contains it.
    assert r["checklist"]["compliance"]["evidence"] == [{"turn": 3, "quote": "verify your name and date of birth"}]
    assert r["checklist"]["culture"]["answer"] is None
    assert r["sentiment"] == {"label": "neutral", "evidence": [{"turn": 2, "quote": "I got a bill"}]}
    assert r["outcome"]["label"] is None
    assert r["coaching"] == [{"item": "capturing_data", "tip": "Confirm the callback number.", "turn": 3},
                             {"item": "general", "tip": "Be nice.", "turn": None}]
    assert r["evidence_check"] == {"kept": 3, "dropped": 1}


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nexport GROQ_API_KEY="from-file"\nGROQ_MODEL=file-model\n')
    monkeypatch.setenv("GROQ_MODEL", "from-shell")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    load_dotenv(env)
    import os
    assert os.environ["GROQ_API_KEY"] == "from-file"
    assert os.environ["GROQ_MODEL"] == "from-shell"


def test_groq_key_selects_groq_defaults(monkeypatch):
    monkeypatch.setattr(llm_client, "load_dotenv", lambda *a, **k: None)
    for v in ("SALUD_LLM_BASE_URL", "SALUD_LLM_MODEL", "SALUD_LLM_API_KEY", "GROQ_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    ep = LLMEndpoint()
    assert ep.base_url == llm_client.GROQ_BASE_URL and ep.provider == "Groq"
    assert ep.model == llm_client.GROQ_DEFAULT_MODEL and ep.api_key == "gsk_test"
    assert not ep.is_local


class FakeLLM(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        FakeLLM.calls += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        first = re.search(r"\[T1\] [^:]+: (.{0,30})", prompt).group(1).strip()
        review = {"checklist": {k: {"answer": "Yes", "reason": "ok", "evidence": [{"turn": 1, "quote": first}]}
                                for k in TRANSCRIPT_ITEMS},
                  "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": "paid", "evidence": []},
                  "summary": "A call.", "coaching": [{"item": "general", "tip": "Keep it up.", "turn": 1}]}
        out = json.dumps({"choices": [{"message": {"content": json.dumps(review)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def llm_url():
    FakeLLM.calls = 0
    httpd = HTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_assistant_caches(llm_url, tmp_path):
    a = CallAssistant(LLMEndpoint(base_url=llm_url, model="fake", api_key=None), tmp_path)
    r1 = a.review(TEXT)
    r2 = a.review(TEXT)
    assert r1 == r2 and FakeLLM.calls == 1 and r1["coaching"][0]["tip"] == "Keep it up."


def _tagger(tmp_path, endpoint, real):
    tdir = tmp_path / "calls"
    tdir.mkdir(exist_ok=True)
    (tdir / "a.txt").write_text(TEXT)
    (tdir / "b.txt").write_text(TEXT.replace("billing", "accounts"))
    csv = tmp_path / "labels.csv"
    key = "real" if real else "synthetic"
    httpd = serve([CallSource(key, key.title(), tdir, csv, real=real)], "AK", port=0, endpoint=endpoint,
                  cache_dir=tmp_path / "cache")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}", csv, key


def _get(url):
    return json.loads(urllib.request.urlopen(url).read())


def _post(url, path, payload):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def test_llm_runs_only_on_analyze(llm_url, tmp_path):
    httpd, url, csv, key = _tagger(tmp_path, LLMEndpoint(base_url=llm_url, model="fake", api_key=None), real=False)
    try:
        # Opening the page and a call, and reading its analysis, never call the LLM.
        _get(url + "/api/state")
        _get(url + f"/api/call?id={key}/a.txt")
        assert _get(url + f"/api/analysis?id={key}/a.txt") == {"none": True}
        assert FakeLLM.calls == 0
        # GET on the analyze route does nothing.
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + f"/api/analyze?id={key}/a.txt")
        assert FakeLLM.calls == 0
        # Analyze: one LLM call; afterwards the analysis is served from cache.
        s = _post(url, "/api/analyze", {"id": f"{key}/a.txt"})
        assert s["checklist"]["greeting"]["answer"] == "Yes" and s["analyzed_at"]
        assert FakeLLM.calls == 1
        assert _get(url + f"/api/analysis?id={key}/a.txt")["checklist"]
        _post(url, "/api/analyze", {"id": f"{key}/a.txt"})            # cached
        assert FakeLLM.calls == 1
        _post(url, "/api/analyze", {"id": f"{key}/a.txt", "force": True})  # Re-analyze
        assert FakeLLM.calls == 2
        # The other call was never touched (no background prefetch).
        assert _get(url + f"/api/analysis?id={key}/b.txt") == {"none": True}
    finally:
        httpd.shutdown()


def test_save_records_provenance(llm_url, tmp_path):
    httpd, url, csv, key = _tagger(tmp_path, LLMEndpoint(base_url=llm_url, model="fake", api_key=None), real=False)
    try:
        values = {k: "Yes" for k in TRANSCRIPT_ITEMS} | {"greeting": "No", "sentiment": "neutral", "outcome": "paid"}
        llm = {"model": "fake", "answers": {k: "Yes" for k in TRANSCRIPT_ITEMS} | {"sentiment": "neutral", "outcome": "paid"}}
        _post(url, "/api/label", {"id": f"{key}/a.txt", "values": values, "llm": llm})
        row = pd.read_csv(csv, dtype=str, keep_default_na=False).query("file_id == 'a.txt'").iloc[0]
        assert row["llm_model"] == "fake" and row["overrides"] == "greeting"
    finally:
        httpd.shutdown()


def test_real_calls_analyzed_like_synthetic(llm_url, tmp_path):
    httpd, url, _, key = _tagger(tmp_path, LLMEndpoint(base_url=llm_url, model="fake", api_key=None), real=True)
    try:
        assert _get(url + "/api/state")["config"]["llm"]["real"] is None
        assert _get(url + f"/api/analysis?id={key}/a.txt") == {"none": True} and FakeLLM.calls == 0
        assert _post(url, "/api/analyze", {"id": f"{key}/a.txt"})["checklist"]["greeting"]["answer"] == "Yes"
        assert FakeLLM.calls == 1
    finally:
        httpd.shutdown()


def test_no_llm_configured(tmp_path):
    httpd, url, _, _ = _tagger(tmp_path, None, real=False)
    try:
        assert "GROQ_API_KEY" in _get(url + "/api/state")["config"]["llm"]["synthetic"]
    finally:
        httpd.shutdown()
