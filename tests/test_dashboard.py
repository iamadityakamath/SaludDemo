import json
import threading
import urllib.error
import urllib.request

import pytest

from salud_pipeline.phase1.dashboard import serve

GOOD = {"greeting": "Yes", "culture": "No", "capturing_data": "N/A", "understanding_needs": "Yes",
        "correct_info": "Yes", "first_contact_resolution": "No", "compliance": "Yes",
        "sentiment": "neutral", "outcome": "transferred", "notes": "garbled middle"}


@pytest.fixture()
def url(store):
    httpd = serve(store.test_rater, port=0, store=store)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _get(url):
    return json.loads(urllib.request.urlopen(url).read())


def _post(url, payload):
    req = urllib.request.Request(url + "/api/label", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def _some_call(store, source):
    return next(c["call_id"] for c in store.calls() if c["source"] == source)


def test_state_lists_real_and_synthetic_calls_from_the_database(store, url):
    st = _get(url + "/api/state")
    sources = {c["source"] for c in st["calls"]}
    assert sources == {"real", "synthetic"}
    assert st["config"]["ai"]["analyzed"] >= 1 and st["labels"] == {}   # a fresh rater has no tags
    cid = _some_call(store, "real")
    call = _get(url + f"/api/call?id={cid}")
    assert call["turns"] and "Editorial note" not in call["text"]
    assert _get(url + f"/api/analysis?id={cid}")["checklist"]
    assert b"Call QA Review" in urllib.request.urlopen(url + "/").read()


def test_state_lists_each_calls_assigned_raters(store, url):
    cid = _some_call(store, "real")
    assert store.assign([(cid, store.test_rater), (cid, store.test_rater)]) == 1   # duplicates ignored
    st = _get(url + "/api/state")
    call = next(c for c in st["calls"] if c["id"] == cid)
    assert store.test_rater in call["assignees"] and store.test_rater in st["config"]["raters"]


def test_save_tag_writes_the_raters_row_with_provenance(store, url):
    cid = _some_call(store, "synthetic")
    before = store.ai_analysis(cid)
    llm = {"model": "gpt-test", "answers": {"greeting": "Yes", "culture": "Yes"}}
    _post(url, {"id": cid, "values": GOOD, "llm": llm})
    _post(url, {"id": cid, "values": {**GOOD, "greeting": "No"}, "llm": llm})   # re-save replaces the row
    rows = store.tags(store.test_rater)
    assert len(rows) == 1
    r = rows.iloc[0]
    assert r["greeting"] == "No" and r["capturing_data"] == "N/A" and r["llm_model"] == "gpt-test"
    assert r["overrides"] == ["greeting", "culture"]
    assert store.ai_analysis(cid) == before                                   # AI analysis untouched
    assert cid in _get(url + "/api/state")["labels"]


def test_raters_only_see_their_own_tags(store, url):
    cid = _some_call(store, "synthetic")
    store.save_tag(store.test_rater + "-other", cid, GOOD)
    try:
        assert _get(url + "/api/state")["labels"] == {}
    finally:
        with store.pool.connection() as c:
            c.execute("DELETE FROM phase1.reviewed_tags WHERE rater = %s", (store.test_rater + "-other",))


@pytest.mark.parametrize("payload", [
    {"values": {**GOOD, "greeting": "Maybe"}},
    {"values": {**GOOD, "outcome": "teleported"}},
    {"id": "real/does-not-exist.txt", "values": GOOD},
])
def test_rejects_bad_input(store, url, payload):
    payload = {"id": _some_call(store, "real"), **payload}
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(url, payload)
    assert e.value.code == 400


def test_no_llm_route_and_loopback_only(url):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(url + "/api/analyze", data=b"{}"))
    assert e.value.code == 404
    assert url.startswith("http://127.0.0.1:")
