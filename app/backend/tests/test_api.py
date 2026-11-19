"""API checks with the store faked: shapes, validation, 404s. No database or LLM."""
import pytest
from fastapi.testclient import TestClient

from salud_app import db
from salud_app.main import app
from salud_app.phase1 import analysis, store


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(store, "analysis_list", lambda: [{"call_id": "synthetic/a.txt", "source": "synthetic", "file_id": "a.txt",
                        "added_by": None, "added_at": None, "analyzed": True, "scored": True, "end_label": "positive"}])
    monkeypatch.setattr(store, "transcript", lambda cid: "Agent: hi" if cid == "synthetic/a.txt" else None)
    monkeypatch.setattr(analysis, "llm_ready", lambda: True)
    monkeypatch.setattr(db, "ping", lambda: True)
    return TestClient(app)


def test_list_shape(client):
    d = client.get("/api/analyze/list").json()
    assert d["llm"] is True and d["website"] is False and len(d["items"]) == 7
    assert d["calls"][0]["status"] == "done"


def test_call_and_404(client):
    assert client.get("/api/call", params={"id": "synthetic/a.txt"}).json()["turns"][0]["speaker"] == "Agent"
    assert client.get("/api/call", params={"id": "nope"}).status_code == 404
    assert client.get("/api/result", params={"id": "nope"}).status_code == 404


def test_upload_validation(client, monkeypatch):
    added = []
    monkeypatch.setattr(store, "add_new_call", lambda t, by: added.append(t) or f"new/{len(added)}")
    monkeypatch.setattr(analysis, "analyze_in_background", lambda ids: None)
    assert client.post("/api/analyze/new", json={"transcripts": []}).status_code == 422
    assert client.post("/api/analyze/new", json={"transcripts": ["   "]}).status_code == 400
    assert client.post("/api/analyze/new", json={"transcripts": ["x" * 60_001]}).status_code == 400
    assert client.post("/api/analyze/new", json={"transcripts": ["a"] * 21}).status_code == 422
    assert client.post("/api/analyze/new", json={"transcripts": ["Agent: hi"]}).json() == {"ids": ["new/1"]}


def test_health(client):
    assert client.get("/api/health").json()["database"] is True
