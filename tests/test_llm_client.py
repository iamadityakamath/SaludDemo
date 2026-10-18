import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from salud_pipeline.llm_client import LLMEndpoint
from salud_pipeline.qa.targets import TRANSCRIPT_ITEMS

REPLY = {"checklist": {k: {"answer": "Yes", "reason": "ok", "evidence": [{"turn": 1, "quote": "hi"}]}
                       for k in TRANSCRIPT_ITEMS},
         "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": "paid", "evidence": []},
         "summary": "s", "coaching": []}


class FakeServer(BaseHTTPRequestHandler):
    requests: list = []

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeServer.requests.append({"auth": self.headers.get("Authorization"), "body": body})
        self._send({"choices": [{"message": {"content": json.dumps(REPLY)}}]})

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    FakeServer.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_endpoint_reads_env_and_sends_bearer_token(server, monkeypatch):
    monkeypatch.setenv("SALUD_LLM_BASE_URL", server)
    monkeypatch.setenv("SALUD_LLM_MODEL", "fake-model")
    monkeypatch.setenv("SALUD_LLM_API_KEY", "secret-token")
    ep = LLMEndpoint()
    assert ep.list_models() == ["fake-model"]
    ep.chat([{"role": "user", "content": "hi"}], json_mode=True)
    req = FakeServer.requests[-1]
    assert req["auth"] == "Bearer secret-token"
    assert req["body"]["response_format"] == {"type": "json_object"}
    assert "secret-token" not in repr(ep)


def test_remote_hosts_are_not_local():
    assert LLMEndpoint(base_url="http://localhost:11434/v1").is_local
    assert not LLMEndpoint(base_url="https://llm.team-server.edu/v1").is_local
