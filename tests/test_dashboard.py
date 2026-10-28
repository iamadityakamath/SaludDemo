import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from salud_pipeline.phase1 import live
from salud_pipeline.phase1.checklist import OUTCOMES, TRANSCRIPT_ITEMS
from salud_pipeline.phase1.dashboard import serve
from salud_pipeline.sentiment.model import from_probs

CALL = """Agent: Thank you for calling, how can I help?

Patient: This bill is ridiculous, I already paid it.

Agent: I'm sorry, let me fix that for you.

Patient: Thank you, that's a relief."""


class FakeModel:
    def score(self, texts):
        return [from_probs(0.9, 0.08, 0.02) if "ridiculous" in t else from_probs(0.02, 0.08, 0.9) for t in texts]


def fake_review(summary):
    return {"ok": True, "summary": summary, "coaching": [{"item": "greeting", "tip": "Say your name.", "turn": 1}],
            "checklist": {k: {"answer": "Yes", "reason": "fine", "evidence": []} for k in TRANSCRIPT_ITEMS},
            "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": OUTCOMES[0], "evidence": []}}


@pytest.fixture()
def no_cloud(monkeypatch):
    """No real model download and no real LLM: a keyword sentiment model and a canned review per call."""
    reviews = []

    class FakeAssistant:
        def __init__(self, endpoint, cache_dir):
            pass

        def review(self, text):
            reviews.append(text)
            return fake_review(f"review {len(reviews)}")

    monkeypatch.setattr("salud_pipeline.sentiment.model.shared_model", lambda: FakeModel())
    monkeypatch.setattr(live, "llm_ready", lambda: True)
    monkeypatch.setattr(live, "LLMEndpoint", lambda: SimpleNamespace(model="fake-model"))
    monkeypatch.setattr(live, "CallAssistant", FakeAssistant)
    return reviews


@pytest.fixture()
def url(store):
    httpd = serve(store.test_rater, port=0, store=store)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _get(url):
    return json.loads(urllib.request.urlopen(url).read())


def _post(url, path, payload):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def _some_call(store, source):
    return next(c["call_id"] for c in store.calls() if c["source"] == source)


def _drop(store, call_id):
    with store.pool.connection() as c:
        for t in ("ai_coaching", "ai_checklist", "ai_analyses"):
            c.execute(f"DELETE FROM phase1.{t} WHERE call_id = %s", (call_id,))
        c.execute("DELETE FROM phase1.calls WHERE call_id = %s", (call_id,))   # sentiment rows cascade


def test_page_and_call_list_come_from_the_database(store, url):
    for path in ("/", "/analyze"):
        assert b"Agent Dashboard" in urllib.request.urlopen(url + path).read()
    d = _get(url + "/api/analyze/list")
    assert {"real", "synthetic"} <= {c["source"] for c in d["calls"]} and d["me"] == store.test_rater
    assert any(c["analyzed"] for c in d["calls"]) and d["items"] == TRANSCRIPT_ITEMS
    call = _get(url + f"/api/call?id={_some_call(store, 'real')}")
    assert call["turns"] and "Editorial note" not in call["text"]


def test_tagging_routes_are_gone(url):
    for path, body in (("/api/state", None), ("/api/label", {"id": "x", "values": {}})):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                                          headers={"Content-Type": "application/json"}))
        assert e.value.code == 404
    assert url.startswith("http://127.0.0.1:")


def test_analyze_reuses_stored_results_and_reanalyze_replaces_them(store, url, no_cloud):
    cid = store.add_new_call(CALL, store.test_rater)
    try:
        first = _post(url, "/api/analyze/run", {"id": cid})
        assert first["analysis"]["summary"] == "review 1" and first["sentiment"]["call"]["caller_end_label"] == "positive"
        again = _post(url, "/api/analyze/run", {"id": cid})                  # nothing missing: no new review
        assert again["analysis"]["summary"] == "review 1" and len(no_cloud) == 1
        redo = _post(url, "/api/analyze/run", {"id": cid, "redo": True})      # Re-analyze: a fresh review replaces it
        assert redo["analysis"]["summary"] == "review 2" and len(no_cloud) == 2
        with store.pool.connection() as c:
            counts = [c.execute(f"SELECT count(*) FROM phase1.{t} WHERE call_id = %s", (cid,)).fetchone()["count"]
                      for t in ("ai_analyses", "ai_checklist", "ai_coaching", "call_sentiment")]
        assert counts == [1, len(TRANSCRIPT_ITEMS), 1, 1]                      # replaced, not duplicated
    finally:
        _drop(store, cid)


def test_reanalyze_without_an_llm_keeps_the_stored_review(store, url, no_cloud, monkeypatch):
    cid = store.add_new_call(CALL, store.test_rater)
    try:
        _post(url, "/api/analyze/run", {"id": cid})
        monkeypatch.setattr(live, "llm_ready", lambda: False)
        r = _post(url, "/api/analyze/run", {"id": cid, "redo": True})
        assert r["analysis"]["summary"] == "review 1" and "stored review was kept" in r["notes"][0]
    finally:
        _drop(store, cid)


def test_unknown_call_is_rejected(url, no_cloud):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(url, "/api/analyze/run", {"id": "real/does-not-exist.txt"})
    assert e.value.code == 400


# ---- the website (Vercel): team passcode, then a name ----
PASS = "test-passcode-1234"


@pytest.fixture()
def site(store):
    from http.server import ThreadingHTTPServer

    from salud_pipeline.phase1.dashboard import make_handler

    some = _some_call(store, "synthetic")
    store.assign([(some, store.test_rater)])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, passcode=PASS))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _req(url, path, payload=None, cookie=None):
    req = urllib.request.Request(url + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **({"Cookie": cookie} if cookie else {})})
    try:
        r = urllib.request.urlopen(req)
        return r.status, json.loads(r.read()), r.headers.get("Set-Cookie")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), e.headers.get("Set-Cookie")


def test_website_needs_passcode_and_name_before_any_data(store, site):
    assert _req(site, "/api/analyze/list")[0] == 401
    assert _req(site, "/api/call?id=" + _some_call(store, "real"))[0] == 401
    assert _req(site, "/api/analyze/run", {"id": _some_call(store, "real")})[0] == 401
    assert _req(site, "/api/analyze/new", {"transcript": CALL})[0] == 401
    assert b"Agent Call Analysis" in urllib.request.urlopen(site + "/").read()   # the page itself has no data
    assert _req(site, "/api/login", {"passcode": "wrong"})[0] == 401
    status, body, _ = _req(site, "/api/login", {"passcode": PASS})
    assert status == 200 and store.test_rater in body["raters"]
    assert _req(site, "/api/login", {"passcode": PASS, "rater": "not-on-the-team"})[0] == 400


def test_website_session_identifies_the_user(store, site):
    status, _, set_cookie = _req(site, "/api/login", {"passcode": PASS, "rater": store.test_rater.upper()})
    assert status == 200 and "HttpOnly" in set_cookie and "Secure" in set_cookie and "SameSite=Strict" in set_cookie
    cookie = set_cookie.split(";")[0]
    status, d, _ = _req(site, "/api/analyze/list", cookie=cookie)
    assert status == 200 and d["me"] == store.test_rater and d["website"]
    name, value = cookie.split("=", 1)
    forged = f"{name}=avni|{value.split('|')[1]}|{value.split('|')[2]}"
    assert _req(site, "/api/analyze/list", cookie=forged)[0] == 401      # can't change the name in the cookie
    assert "Max-Age=0" in _req(site, "/api/logout", {})[2]


def test_session_signature_depends_on_passcode():
    from salud_pipeline.phase1.dashboard import make_session, read_session

    v = make_session("passcode-one-12", "avni")
    assert read_session("passcode-one-12", v) == "avni"
    assert read_session("passcode-two-12", v) is None                  # new passcode signs everyone out
    assert read_session("passcode-one-12", make_session("passcode-one-12", "avni", now=0)) is None   # expired
