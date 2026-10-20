import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from salud_pipeline import llm_client
from salud_pipeline.llm_client import LLMEndpoint, load_dotenv
from salud_pipeline.phase1.prompt import CallAssistant, parse, split_turns
from salud_pipeline.phase1.checklist import TRANSCRIPT_ITEMS

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
