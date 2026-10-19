import openpyxl
import pandas as pd

from salud_pipeline.phase1 import compare
from salud_pipeline.phase1.calls import CallSource, read_call_list
from salud_pipeline.phase1.checklist import LABEL_COLUMNS, TRANSCRIPT_ITEMS

TEXT = "Agent: Thank you for calling billing, this is [REDACTED NAME].\n\nPatient: Hi, I want to pay."


def analysis(answers: dict, outcome="paid", evidence=True):
    return {"ok": True, "model": "fake", "analyzed_at": "2026-10-18T12:00:00",
            "checklist": {k: {"answer": v, "reason": "r",
                              "evidence": [{"turn": 1, "quote": "Thank you for calling"}] if evidence else []}
                          for k, v in answers.items()},
            "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": outcome, "evidence": []},
            "summary": "A call.", "coaching": [{"item": "greeting", "tip": "Keep it up.", "turn": 1}]}


def label_row(file_id, rater, answers, llm_model=""):
    row = {c: "" for c in LABEL_COLUMNS}
    row.update({"file_id": file_id, "rater": rater, **answers, "sentiment": "neutral", "outcome": "paid",
                "llm_model": llm_model})
    return row


def make_sources(tmp_path):
    real_dir, syn_dir = tmp_path / "real", tmp_path / "syn" / "transcripts"
    real_dir.mkdir()
    syn_dir.mkdir(parents=True)
    for d, names in ((real_dir, ["r1.txt", "r2.txt"]), (syn_dir, ["s1.txt"])):
        for n in names:
            (d / n).write_text(TEXT.replace("pay", f"pay {n}"))
    return [CallSource("real", "Real", real_dir, tmp_path / "real.csv", real=True),
            CallSource("synthetic", "Synthetic", syn_dir, tmp_path / "syn.csv", real=False)]


def test_compare_end_to_end(tmp_path):
    sources = make_sources(tmp_path)
    all_yes = {k: "Yes" for k in TRANSCRIPT_ITEMS}
    one_no = {**all_yes, "compliance": "No"}
    pd.DataFrame([label_row("r1.txt", "AK", all_yes),                      # tagged with AI off
                  label_row("r1.txt", "RM", one_no, llm_model="fake"),    # second rater, AI pre-filled
                  label_row("r2.txt", "AK", one_no, llm_model="fake")]).to_csv(tmp_path / "real.csv", index=False)
    pd.DataFrame([label_row("s1.txt", "AK", all_yes)]).to_csv(tmp_path / "syn.csv", index=False)

    r2 = analysis({**all_yes, "compliance": "N/A"})
    r2["checklist"]["greeting"]["evidence"] = []
    analyses = {"real/r1.txt": analysis(all_yes), "real/r2.txt": r2}

    res = compare.build(sources, analyses)
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

    xlsx = compare.write(res, tmp_path / "out")
    wb = openpyxl.load_workbook(xlsx)
    assert wb.sheetnames == ["Summary", "Calls", "Coaching", "AI vs human", "Rater agreement",
                             "Synthetic check", "Definitions"]
    assert (tmp_path / "out" / "calls.csv").exists() and "AI vs human" in (tmp_path / "out" / "summary.md").read_text()


def test_compare_with_nothing_yet(tmp_path):
    sources = make_sources(tmp_path)
    res = compare.build(sources, {})
    assert res["counts"].iloc[0]["calls_tagged"] == 0
    assert "No tagged calls" in compare.summary_markdown(res)
    compare.write(res, tmp_path / "out")


def test_call_list_limits_synthetic_calls(tmp_path):
    sources = make_sources(tmp_path)
    lst = tmp_path / "list.txt"
    lst.write_text("# comment line\ns1\n\nnot-a-call\n")
    assert read_call_list(lst) == {"s1", "not-a-call"}
    assert read_call_list(tmp_path / "missing.txt") is None
    syn = CallSource("synthetic", "Synthetic", sources[1].transcripts_dir, tmp_path / "s.csv", real=False,
                     include=read_call_list(lst))
    assert [p.name for p in syn.files()] == ["s1.txt"]
    assert len(compare.load_calls([sources[0], syn])) == 3
