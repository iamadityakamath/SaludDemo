import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from salud_pipeline.llm_client import LLMEndpoint
from salud_pipeline.qa.tag_server import CallSource, serve
from salud_pipeline.qa.targets import load_ground_truth, write_template

GOOD = {"greeting": "Yes", "culture": "No", "capturing_data": "N/A", "understanding_needs": "Yes",
        "correct_info": "Yes", "first_contact_resolution": "No", "compliance": "Yes",
        "sentiment": "neutral", "outcome": "transferred", "notes": "garbled middle"}
TEXT = "Editorial note: x\n\nAgent: Hello.\n\nPatient: Hi."


@pytest.fixture()
def setup(tmp_path):
    def make(name, files):
        d = tmp_path / name
        d.mkdir()
        for f in files:
            (d / f).write_text(TEXT)
        return d

    real_dir, syn_dir = make("real", ["a.txt", "b.txt"]), make("syn", ["s1.txt", "s2.txt", "s3.txt"])
    real_csv, syn_csv = tmp_path / "real.csv", tmp_path / "syn.csv"
    write_template(real_csv, real_dir)
    sources = [CallSource("real", "Real", real_dir, real_csv, real=True),
               CallSource("synthetic", "Synthetic", syn_dir, syn_csv, real=False)]
    servers = []

    def start(rater, endpoint=None, include_real=False):
        httpd = serve(sources, rater, port=0, endpoint=endpoint, include_real=include_real,
                      cache_dir=tmp_path / "cache")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}"

    yield start, real_csv, syn_csv, real_dir
    for s in servers:
        s.shutdown()


def _post(url, payload):
    req = urllib.request.Request(url + "/api/label", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def _get(url):
    return json.loads(urllib.request.urlopen(url).read())


def test_one_page_lists_both_sources(setup):
    start, *_ = setup
    url = start("AK")
    assert b"Call QA Review" in urllib.request.urlopen(url + "/").read()
    st = _get(url + "/api/state")
    assert [c["id"] for c in st["calls"]] == ["real/a.txt", "real/b.txt", "synthetic/s1.txt",
                                             "synthetic/s2.txt", "synthetic/s3.txt"]
    assert [s["key"] for s in st["config"]["sources"]] == ["real", "synthetic"]
    assert "text" not in st["calls"][0]  # transcripts load on demand
    call = _get(url + "/api/call?id=real/a.txt")
    assert "Editorial note" not in call["text"] and call["turns"][0]["speaker"] == "Agent"


def test_labels_route_to_each_sources_csv(setup):
    start, real_csv, syn_csv, real_dir = setup
    url = start("AK")
    _post(url, {"id": "real/a.txt", "values": GOOD})
    _post(url, {"id": "real/a.txt", "values": {**GOOD, "greeting": "No"}})  # re-save overwrites
    _post(url, {"id": "synthetic/s2.txt", "values": GOOD})
    real = pd.read_csv(real_csv, dtype=str, keep_default_na=False)
    rows = real[real["file_id"] == "a.txt"]
    assert len(rows) == 1 and rows.iloc[0]["greeting"] == "No"
    assert (real["file_id"] == "b.txt").sum() == 1  # untouched template row stays
    assert "s2.txt" not in set(real["file_id"])
    syn = pd.read_csv(syn_csv, dtype=str, keep_default_na=False)
    assert list(syn["file_id"]) == ["s2.txt"]
    gt = load_ground_truth(real_csv, real_dir).dropna(subset=["sentiment"])
    assert gt.iloc[0]["contact_id"] == "a" and gt.iloc[0]["capturing_data"] == "N/A"
    labels = _get(url + "/api/state")["labels"]
    assert set(labels) == {"real/a.txt", "synthetic/s2.txt"}


def test_raters_are_blind_to_each_other(setup):
    start, real_csv, *_ = setup
    ak, rm = start("AK"), start("RM")
    _post(ak, {"id": "real/a.txt", "values": GOOD})
    assert "real/a.txt" in _get(ak + "/api/state")["labels"]
    assert _get(rm + "/api/state")["labels"] == {}
    _post(rm, {"id": "real/a.txt", "values": {**GOOD, "sentiment": "negative"}})
    df = pd.read_csv(real_csv, dtype=str, keep_default_na=False)
    assert sorted(df[df["file_id"] == "a.txt"]["rater"]) == ["AK", "RM"]


def test_ai_guard_is_per_source(setup):
    start, *_ = setup
    remote = LLMEndpoint(base_url="https://api.groq.com/openai/v1", model="m", api_key="k")
    llm = _get(start("AK", endpoint=remote) + "/api/state")["config"]["llm"]
    assert llm["synthetic"] is None and "--include-real" in llm["real"]
    llm = _get(start("AK", endpoint=remote, include_real=True) + "/api/state")["config"]["llm"]
    assert llm["real"] is None and llm["synthetic"] is None


@pytest.mark.parametrize("payload", [
    {"id": "real/a.txt", "values": {**GOOD, "greeting": "Maybe"}},
    {"id": "real/a.txt", "values": {**GOOD, "outcome": "teleported"}},
    {"id": "real/../../etc/passwd", "values": GOOD},
    {"id": "a.txt", "values": GOOD},
])
def test_rejects_bad_input(setup, payload):
    start, *_ = setup
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(start("AK"), payload)
    assert e.value.code == 400


def test_unknown_call_404(setup):
    start, *_ = setup
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(start("AK") + "/api/call?id=synthetic/nope.txt")
    assert e.value.code == 404


def test_binds_loopback_only(setup):
    start, *_ = setup
    assert start("AK").startswith("http://127.0.0.1:")
