import openpyxl
import pandas as pd

from salud_pipeline.phase1 import compare
from salud_pipeline.phase1.calls import CallSource, read_call_list
from salud_pipeline.phase1.checklist import TARGETS, TRANSCRIPT_ITEMS

CALLS = [{"call_id": "real/r1.txt", "source": "real", "file_id": "r1.txt"},
         {"call_id": "real/r2.txt", "source": "real", "file_id": "r2.txt"},
         {"call_id": "synthetic/s1.txt", "source": "synthetic", "file_id": "s1.txt"}]


def analysis(answers: dict, outcome="paid", evidence=True):
    return {"ok": True, "model": "fake", "analyzed_at": "2026-10-18T12:00:00",
            "checklist": {k: {"answer": v, "reason": "r",
                              "evidence": [{"turn": 1, "quote": "Thank you for calling"}] if evidence else []}
                          for k, v in answers.items()},
            "sentiment": {"label": "neutral", "evidence": []}, "outcome": {"label": outcome, "evidence": []},
            "summary": "A call.", "coaching": [{"item": "greeting", "tip": "Keep it up.", "turn": 1}]}


def tag(call_id, rater, answers, llm_model=""):
    return {"call_id": call_id, "rater": rater, **answers, "sentiment": "neutral", "outcome": "paid",
            "notes": None, "llm_model": llm_model, "overrides": ""}


def test_compare_end_to_end(tmp_path):
    all_yes = {k: "Yes" for k in TRANSCRIPT_ITEMS}
    one_no = {**all_yes, "compliance": "No"}
    labels = pd.DataFrame([tag("real/r1.txt", "AK", all_yes),                      # tagged with AI off
                           tag("real/r1.txt", "RM", one_no, llm_model="fake"),    # second rater, AI pre-filled
                           tag("real/r2.txt", "AK", one_no, llm_model="fake"),
                           tag("synthetic/s1.txt", "AK", all_yes)])
    r2 = analysis({**all_yes, "compliance": "N/A"})
    r2["checklist"]["greeting"]["evidence"] = []
    analyses = {"real/r1.txt": analysis(all_yes), "real/r2.txt": r2, "synthetic/s1.txt": analysis(all_yes)}
    planted = pd.DataFrame([{"call_id": "synthetic/s1.txt", **all_yes, "sentiment": "neutral", "outcome": "paid"}])

    res = compare.build(CALLS, labels, analyses, planted)
    c = res["counts"].iloc[0]
    assert (c["calls_available"], c["calls_with_ai_analysis"], c["calls_tagged"]) == (3, 3, 3)
    assert c["tags_made_with_ai_off"] == 2

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
    assert len(res["rater_agreement"]) and len(res["coaching"]) == 3
    assert set(res["synthetic_check"]["comparison"]) == {"Human vs planted label", "AI vs planted label"}

    xlsx = compare.write(res, tmp_path / "out")
    assert openpyxl.load_workbook(xlsx).sheetnames == ["Summary", "Calls", "Coaching", "AI vs human",
                                                       "Rater agreement", "Synthetic check", "Salud format",
                                                       "Definitions"]
    assert "AI vs human" in (tmp_path / "out" / "summary.md").read_text()


def test_salud_format_uses_salud_codes_and_never_writes_na():
    all_yes = {k: "Yes" for k in TRANSCRIPT_ITEMS}
    labels = pd.DataFrame([tag("real/r1.txt", "AK", {**all_yes, "compliance": "No", "correct_info": "N/A"}),
                           tag("synthetic/s1.txt", "RM", all_yes)])
    out = compare.salud_format(labels).set_index("call_id")
    r1, s1 = out.loc["real/r1.txt"], out.loc["synthetic/s1.txt"]
    assert r1["salud_AppropriateGreetings"] == 785570000 and r1["salud_Compliance"] == 785570001
    assert pd.isna(r1["salud_CorrectInformation"]) and r1["na_items"] == "correct_info"   # N/A -> NULL
    assert pd.isna(r1["salud_CorrectCallStatusing"]) and pd.isna(r1["salud_Grammar"])   # not scored in Phase 1
    assert pd.isna(r1["salud_QualityReviewId"]) and s1["salud_QualityReviewId"] == "s1"  # synthetic contact only
    assert not out["review_complete"].any()
    scores = out[[c for c in out if c.startswith("salud_") and not c.endswith(("Id", "Comment", "Comments", "date",
                                                                                "lessonslearned"))]]
    assert set(scores.stack().dropna().unique()) <= {785570000, 785570001}
    assert compare.salud_format(labels).equals(compare.salud_format(labels))            # stable review IDs


def test_compare_with_nothing_yet(tmp_path):
    empty = pd.DataFrame(columns=["call_id", "rater"] + TARGETS + ["notes", "llm_model", "overrides"])
    res = compare.build(CALLS, empty, {})
    assert res["counts"].iloc[0]["calls_tagged"] == 0
    assert "No tagged calls" in compare.summary_markdown(res)
    compare.write(res, tmp_path / "out")


def test_call_list_limits_synthetic_calls(tmp_path):
    d = tmp_path / "syn"
    d.mkdir()
    for n in ("s1", "s2"):
        (d / f"{n}.txt").write_text("Agent: Hi.")
    lst = tmp_path / "list.txt"
    lst.write_text("# comment line\ns1\n\nnot-a-call\n")
    assert read_call_list(lst) == {"s1", "not-a-call"}
    assert read_call_list(tmp_path / "missing.txt") is None
    src = CallSource("synthetic", "Synthetic", d, tmp_path / "s.csv", real=False, include=read_call_list(lst))
    assert [p.name for p in src.files()] == ["s1.txt"]
