"""API tests with a fake data layer: no database or local files needed."""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from salud_app import db
from salud_app.main import app
from salud_app.repositories import calls as calls_repo
from salud_app.repositories import phase2 as phase2_repo
from salud_app.schemas import Phase2Summary, SentimentSummary

RECORDS = [
    {"key": "final-C1", "source": "real", "dataset": "final", "db_id": "C1", "phase1_call_id": None,
     "contacted_at": datetime(2026, 7, 1, 10), "outcome": "Promised to Pay", "transcript_text": "Agent: Hi\nCaller: Hello",
     "sentiment": SentimentSummary(caller_end=0.4, caller_end_label="positive"), "qa_score": 0.71, "ai": None,
     "facts": {"connection": "Inbound Call", "talk_seconds": 120.0}},
    {"key": "p1-synthetic/abc", "source": "synthetic", "dataset": "phase1", "db_id": "synthetic/abc",
     "phase1_call_id": "synthetic/abc", "contacted_at": None, "outcome": "paid", "transcript_text": "Agent: Hi",
     "sentiment": None, "qa_score": None, "ai": None, "facts": {}},
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(calls_repo, "load_records", lambda force=False: RECORDS)
    monkeypatch.setattr(calls_repo, "_turn_sentiment", lambda call_id: [])
    monkeypatch.setattr(phase2_repo, "summary", lambda: Phase2Summary(available=False))
    monkeypatch.setattr(db, "ping", lambda: True)
    return TestClient(app)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_list_filter_and_paging(client):
    r = client.get("/api/calls").json()
    assert r["total"] == 2 and r["items"][0]["key"] == "final-C1"       # dated real calls first
    assert client.get("/api/calls", params={"source": "synthetic"}).json()["total"] == 1
    assert client.get("/api/calls", params={"search": "promised"}).json()["total"] == 1
    assert len(client.get("/api/calls", params={"limit": 1, "offset": 1}).json()["items"]) == 1
    assert client.get("/api/calls", params={"source": "bogus"}).status_code == 422


def test_detail_including_key_with_slash(client):
    d = client.get("/api/calls/final-C1").json()
    assert [t["role"] for t in d["transcript"]] == ["agent", "caller"] and d["call_facts"]["talk_seconds"] == 120.0
    assert client.get("/api/calls/p1-synthetic%2Fabc").json()["source"] == "synthetic"
    assert client.get("/api/calls/nope").status_code == 404


def test_phase2_summary(client):
    assert client.get("/api/phase2/summary").json()["available"] is False


def test_only_get_is_allowed(client):
    assert client.post("/api/calls").status_code == 405


def test_missing_database_gives_503(monkeypatch):
    monkeypatch.setattr(calls_repo, "load_records", lambda force=False: (_ for _ in ()).throw(db.DatabaseNotConfigured("x")))
    assert TestClient(app).get("/api/calls").status_code == 503
