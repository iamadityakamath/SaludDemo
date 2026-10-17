import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from salud_pipeline.qa.tag_server import serve
from salud_pipeline.qa.targets import load_ground_truth, write_template

GOOD = {"greeting": "Yes", "culture": "No", "capturing_data": "N/A", "understanding_needs": "Yes",
        "correct_info": "Yes", "first_contact_resolution": "No", "compliance": "Yes",
        "sentiment": "neutral", "outcome": "transferred", "notes": "garbled middle"}


@pytest.fixture()
def setup(tmp_path):
    tdir = tmp_path / "calls"
    tdir.mkdir()
    for name in ("a.txt", "b.txt"):
        (tdir / name).write_text("Editorial note: x\n\nAgent: Hello.\n\nPatient: Hi.")
    csv = tmp_path / "labels.csv"
    write_template(csv, tdir)
    servers = []

    def start(rater):
        httpd = serve(csv, tdir, rater, port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}"

    yield start, csv, tdir
    for s in servers:
        s.shutdown()


def _post(url, payload):
    req = urllib.request.Request(url + "/api/label", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def _state(url):
    return json.loads(urllib.request.urlopen(url + "/api/state").read())


def test_page_and_state(setup):
    start, _, _ = setup
    url = start("AK")
    assert b"Call QA Review" in urllib.request.urlopen(url + "/").read()
    st = _state(url)
    assert [c["file_id"] for c in st["calls"]] == ["a.txt", "b.txt"]
    assert "Editorial note" not in st["calls"][0]["text"]


def test_save_replaces_template_row_and_is_readable_by_scorer(setup):
    start, csv, tdir = setup
    url = start("AK")
    _post(url, {"file_id": "a.txt", "values": GOOD})
    _post(url, {"file_id": "a.txt", "values": {**GOOD, "greeting": "No"}})  # re-save overwrites
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    rows = df[df["file_id"] == "a.txt"]
    assert len(rows) == 1 and rows.iloc[0]["greeting"] == "No"
    assert (df["file_id"] == "b.txt").sum() == 1  # untouched template row stays
    gt = load_ground_truth(csv, tdir).dropna(subset=["sentiment"])
    assert gt.iloc[0]["contact_id"] == "a" and gt.iloc[0]["capturing_data"] == "N/A"


def test_raters_are_blind_to_each_other(setup):
    start, csv, _ = setup
    ak, rm = start("AK"), start("RM")
    _post(ak, {"file_id": "a.txt", "values": GOOD})
    assert "a.txt" in _state(ak)["labels"]
    assert _state(rm)["labels"] == {}
    _post(rm, {"file_id": "a.txt", "values": {**GOOD, "sentiment": "negative"}})
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert sorted(df[df["file_id"] == "a.txt"]["rater"]) == ["AK", "RM"]


@pytest.mark.parametrize("payload", [
    {"file_id": "a.txt", "values": {**GOOD, "greeting": "Maybe"}},
    {"file_id": "a.txt", "values": {**GOOD, "outcome": "teleported"}},
    {"file_id": "../../etc/passwd", "values": GOOD},
])
def test_rejects_bad_input(setup, payload):
    start, _, _ = setup
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(start("AK"), payload)
    assert e.value.code == 400


def test_binds_loopback_only(setup):
    start, _, _ = setup
    assert start("AK").startswith("http://127.0.0.1:")
