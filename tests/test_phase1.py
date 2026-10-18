import json

import openpyxl
import pandas as pd

from salud_pipeline.llm_client import LLMEndpoint
from salud_pipeline.qa import phase1
from salud_pipeline.qa.assist import CallAssistant
from salud_pipeline.qa.tag_server import CallSource
from salud_pipeline.qa.targets import LABEL_COLUMNS, TRANSCRIPT_ITEMS, clean_transcript

TEXT = "Agent: Thank you for calling billing, this is [REDACTED NAME].\n\nPatient: Hi, I want to pay."


def _analysis(answers: dict, outcome="paid", evidence=True):
    return {"ok": True, "model": "fake", "analyzed_at": "2026-10-18T12:00:00",
            "checklist": {k: {"answer": v, "reason": "r",
                              "evidence": [{"turn": 1, "quote": "Thank you for calling"}] if evidence else []}
                          for k, v in answers.items()},
            "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": outcome, "evidence": []},
            "summary": "A call.", "coaching": [{"item": "greeting", "tip": "Keep it up.", "turn": 1}]}


def _label_row(file_id, rater, answers, llm_model=""):
    row = {c: "" for c in LABEL_COLUMNS}
    row.update({"file_id": file_id, "rater": rater, **answers, "sentiment": "neutral", "outcome": "paid",
                "llm_model": llm_model})
    return row


def test_phase1_end_to_end(tmp_path):
    real_dir, syn_dir = tmp_path / "real", tmp_path / "syn" / "transcripts"
    real_dir.mkdir()
    syn_dir.mkdir(parents=True)
    for d, names in ((real_dir, ["r1.txt", "r2.txt"]), (syn_dir, ["s1.txt"])):
        for n in names:
            (d / n).write_text(TEXT.replace("pay", f"pay {n}"))

    all_yes = {k: "Yes" for k in TRANSCRIPT_ITEMS}
    one_no = {**all_yes, "compliance": "No"}
    pd.DataFrame([_label_row("r1.txt", "AK", all_yes),                      # tagged with AI off
                  _label_row("r1.txt", "RM", one_no, llm_model="fake"),    # second rater, AI pre-filled
                  _label_row("r2.txt", "AK", one_no, llm_model="fake")]).to_csv(tmp_path / "real.csv", index=False)
    pd.DataFrame([_label_row("s1.txt", "AK", all_yes)]).to_csv(tmp_path / "syn.csv", index=False)
    sources = [CallSource("real", "Real", real_dir, tmp_path / "real.csv", real=True),
               CallSource("synthetic", "Synthetic", syn_dir, tmp_path / "syn.csv", real=False)]

    # Cached AI analyses for r1 (agrees with AK) and r2 (answers N/A on compliance, no evidence on greeting).
    a = CallAssistant(LLMEndpoint(base_url="http://127.0.0.1:9/v1", model="fake", api_key=None), tmp_path / "cache")
    a.cache_dir.mkdir()
    r1_text = clean_transcript((real_dir / "r1.txt").read_text())
    r2_text = clean_transcript((real_dir / "r2.txt").read_text())
    a._path(r1_text).write_text(json.dumps(_analysis(all_yes)))
    r2 = _analysis({**all_yes, "compliance": "N/A"})
    r2["checklist"]["greeting"]["evidence"] = []
    a._path(r2_text).write_text(json.dumps(r2))

    res = phase1.build(sources, a, progress=False)
    c = res["counts"].iloc[0]
    assert (c["calls_available"], c["calls_with_ai_analysis"], c["calls_tagged"]) == (3, 2, 3)
    assert c["tagged_calls_missing_ai_analysis"] == 1 and c["tags_made_with_ai_off"] == 2

    calls = res["calls"].set_index("call_id")
    assert calls.loc["real/r1.txt", "n_raters"] == 2
    assert calls.loc["real/r2.txt", "ai_greeting_needs_review"] == True  # noqa: E712
    assert calls.loc["real/r2.txt", "ai_quality_score"] == 1.0           # N/A excluded from the score
    assert calls.loc["real/r2.txt", "human_quality_score"] == round(6 / 7, 3)
    assert calls.loc["real/r2.txt", "agree_compliance"] == False         # noqa: E712  N/A vs No

    ag = res["agreement"].set_index(["group", "target"])
    # r1: raters split Yes/No on compliance -> no consensus, excluded; r2: human No vs AI N/A -> disagree
    assert ag.loc[("Real calls", "compliance"), "n"] == 1
    assert ag.loc[("Real calls", "compliance"), "agreement"] == 0.0
    assert ag.loc[("Real calls", "greeting"), "agreement"] == 1.0
    assert ("Tagged with AI off", "greeting") in ag.index
    assert len(res["rater_agreement"]) and len(res["coaching"]) == 2

    xlsx = phase1.write(res, tmp_path / "out")
    wb = openpyxl.load_workbook(xlsx)
    assert wb.sheetnames == ["Summary", "Calls", "Coaching", "AI vs human", "Rater agreement",
                             "Synthetic check", "Definitions"]
    assert (tmp_path / "out" / "calls.csv").exists() and "AI vs human" in (tmp_path / "out" / "summary.md").read_text()


def test_phase1_with_nothing_tagged(tmp_path):
    d = tmp_path / "real"
    d.mkdir()
    (d / "x.txt").write_text(TEXT)
    res = phase1.build([CallSource("real", "Real", d, tmp_path / "none.csv", real=True)], None, progress=False)
    assert res["counts"].iloc[0]["calls_tagged"] == 0
    assert "No tagged calls" in phase1.summary_markdown(res)
    phase1.write(res, tmp_path / "out")
