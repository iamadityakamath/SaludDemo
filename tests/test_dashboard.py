import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from salud_pipeline.phase1.calls import CallSource
from salud_pipeline.phase1.dashboard import serve
from salud_pipeline.phase1.compare import load_labels
from salud_pipeline.phase1.checklist import LABEL_COLUMNS

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
    # An older labels file can hold blank placeholder rows (no rater); saving replaces them per call.
    pd.DataFrame([{c: "" for c in LABEL_COLUMNS} | {"file_id": f} for f in ("a.txt", "b.txt")]).to_csv(real_csv, index=False)
    sources = [CallSource("real", "Real", real_dir, real_csv, real=True),
               CallSource("synthetic", "Synthetic", syn_dir, syn_csv, real=False)]
    servers = []

    def start(rater, ai_file=None):
        httpd = serve(sources, rater, port=0, ai_file=ai_file or tmp_path / "no_ai.json")
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
    assert (real["file_id"] == "b.txt").sum() == 1  # untouched placeholder row stays
    assert "s2.txt" not in set(real["file_id"])
    syn = pd.read_csv(syn_csv, dtype=str, keep_default_na=False)
    assert list(syn["file_id"]) == ["s2.txt"]
    saved = load_labels([CallSource("real", "Real", real_dir, real_csv, real=True)])
    assert list(saved["call_id"]) == ["real/a.txt"] and saved.iloc[0]["capturing_data"] == "N/A"
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


AI = {"ok": True, "model": "gpt-test",
      "checklist": {k: {"answer": "Yes", "reason": "r", "evidence": [{"turn": 1, "quote": "Hello."}]}
                    for k in ["greeting", "culture", "capturing_data", "understanding_needs", "correct_info",
                              "first_contact_resolution", "compliance"]},
      "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": "paid", "evidence": []},
      "summary": "s", "coaching": []}


def _ai_file(tmp_path, call_ids):
    f = tmp_path / "ai_analyses.json"
    f.write_text(json.dumps({"model": "gpt-test", "prompt_version": "v", "calls": {
        cid: {"source": cid.split("/")[0], "file_id": cid.split("/")[1], "analysis": AI} for cid in call_ids}}))
    return f


def test_serves_ai_answers_from_file_and_never_writes_it(setup, tmp_path):
    start, real_csv, *_ = setup
    f = _ai_file(tmp_path, ["real/a.txt"])
    before = f.read_bytes()
    url = start("AK", ai_file=f)
    st = _get(url + "/api/state")["config"]["ai"]
    assert st["analyzed"] == 1 and st["model"] == "gpt-test"
    assert _get(url + "/api/analysis?id=real/a.txt")["checklist"]["greeting"]["answer"] == "Yes"
    assert _get(url + "/api/analysis?id=real/b.txt") == {"none": True}
    _post(url, {"id": "real/a.txt", "values": {**GOOD, "greeting": "No"},
                "llm": {"model": "gpt-test", "answers": {"greeting": "Yes", "culture": "Yes"}}})
    assert f.read_bytes() == before                      # the AI file is untouched
    row = pd.read_csv(real_csv, dtype=str, keep_default_na=False).query("file_id == 'a.txt'").iloc[0]
    assert row["llm_model"] == "gpt-test" and row["overrides"] == "greeting;culture"  # GOOD has culture No
    with pytest.raises(urllib.error.HTTPError) as e:     # the dashboard can no longer call the LLM
        urllib.request.urlopen(urllib.request.Request(url + "/api/analyze", data=b'{"id": "real/a.txt"}'))
    assert e.value.code == 404


def test_picks_up_calls_added_to_the_file_later(setup, tmp_path):
    start, *_ = setup
    f = _ai_file(tmp_path, ["real/a.txt"])
    url = start("AK", ai_file=f)
    assert _get(url + "/api/analysis?id=synthetic/s1.txt") == {"none": True}
    import os, time
    data = json.loads(f.read_text())
    data["calls"]["synthetic/s1.txt"] = {"source": "synthetic", "file_id": "s1.txt", "analysis": AI}
    f.write_text(json.dumps(data))
    os.utime(f, (time.time() + 5, time.time() + 5))  # make sure the change is visible on coarse clocks
    assert _get(url + "/api/analysis?id=synthetic/s1.txt")["checklist"]
    assert _get(url + "/api/state")["config"]["ai"]["analyzed"] == 2


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
