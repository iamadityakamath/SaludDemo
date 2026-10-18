import numpy as np
import pandas as pd

from salud_pipeline.qa import compare as C
from salud_pipeline.qa.targets import TARGETS
from salud_pipeline.synth.cli import main as synth_main


def test_metric_ci_brackets_point():
    y = np.array(["Yes"] * 30 + ["No"] * 10)
    p = y.copy()
    p[:6] = "No"
    point, lo, hi = C.metric_with_ci(y, p, "accuracy")
    assert point == 0.85 and lo <= point <= hi


def test_llm_split_uses_order_file(tmp_path):
    synth_main(["--out", str(tmp_path), "--n-guarantors", "120", "--seed", "4"])
    ids = sorted(p.stem for p in (tmp_path / "transcripts").glob("*.txt"))
    order = ids[::-1][:30]
    (tmp_path / "llm_order.txt").write_text("\n".join(order + ["not-a-call"]) + "\n")
    test, train = C.load_llm_split(tmp_path, test_n=10)
    assert list(test["contact_id"]) == order[:10]
    assert list(train["contact_id"]) == order[10:]


def test_evaluate_and_summary_shapes():
    truth = pd.DataFrame({"contact_id": list("abcd"), **{t: ["Yes", "No", "Yes", "Yes"] for t in TARGETS}})
    pred = truth.copy()
    res = C.evaluate(truth, pred, "perfect")
    assert (res["accuracy"] == 1.0).all() and len(res) == len(TARGETS)
    table = C.summary_table(res)
    assert table.loc["checklist average (7 items)", "perfect"] == "1.00"
