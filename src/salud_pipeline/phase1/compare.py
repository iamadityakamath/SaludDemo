"""Phase 1 route 2, step 2: compare the script's AI analyses with the manual tags, and export per call.

    salud-phase1 compare

Inputs:
  - AI analyses from `salud-phase1 analyze` (reports/phase1/ai_analyses.json)
  - manual tags saved from the dashboard (data/phase1/manual_tags/*.csv)

Outputs (reports/phase1/comparison/):
  - phase1_results.xlsx  Summary, Calls, Coaching, AI vs human, Rater agreement, Synthetic check, Definitions
  - calls.csv, coaching.csv, ai_vs_human.csv, rater_agreement.csv  (same tables, database-friendly)
  - summary.md

Call-quality score (provisional until Salud sets weights): share of applicable checklist items the agent
passed = Yes / (Yes + No) over the 7 transcript-visible items; N/A items are left out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calls import CallSource
from .checklist import (ITEM_DEFINITIONS, LABEL_COLUMNS, OUTCOME_DEFINITIONS, OUTCOME_GROUPS, TARGETS,
                        TRANSCRIPT_ITEMS, load_synthetic)
from .metrics import agreement_table, consensus, inter_rater

ITEM_NAMES = {
    "greeting": "Appropriate greeting", "culture": "Culture & values", "capturing_data": "Captures key data",
    "understanding_needs": "Understands needs", "correct_info": "Correct information",
    "first_contact_resolution": "First-contact resolution", "compliance": "Privacy compliance",
    "sentiment": "Caller sentiment", "outcome": "Call outcome",
}


def call_quality(answers: dict) -> float | None:
    vals = [answers.get(k) for k in TRANSCRIPT_ITEMS]
    yes, no = vals.count("Yes"), vals.count("No")
    return round(yes / (yes + no), 3) if yes + no else None


def load_calls(sources: list[CallSource]) -> list[dict]:
    calls = []
    for s in sources:
        if not s.transcripts_dir.exists():
            continue
        for p in s.files():
            calls.append({"call_id": f"{s.key}/{p.name}", "source": s.key, "file_id": p.name, "real": s.real})
    return calls


def load_labels(sources: list[CallSource]) -> pd.DataFrame:
    """Every saved human tag (one row per call per rater). Blank template rows are dropped."""
    frames = []
    for s in sources:
        if not s.labels_csv.exists():
            continue
        df = pd.read_csv(s.labels_csv, dtype=str, keep_default_na=False).reindex(columns=LABEL_COLUMNS, fill_value="")
        df = df[(df["rater"] != "") & (df[TARGETS] != "").any(axis=1)].copy()
        df.insert(0, "call_id", s.key + "/" + df["file_id"])
        df.insert(1, "source", s.key)
        frames.append(df)
    cols = ["call_id", "source"] + LABEL_COLUMNS
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def ai_flat(call_id: str, a: dict) -> dict:
    """One analysis as one spreadsheet row."""
    row = {"call_id": call_id}
    answers = {}
    for k in TRANSCRIPT_ITEMS:
        c = (a.get("checklist") or {}).get(k) or {}
        answers[k] = c.get("answer")
        ev = c.get("evidence") or []
        row[f"ai_{k}"] = c.get("answer")
        row[f"ai_{k}_reason"] = c.get("reason")
        row[f"ai_{k}_evidence"] = " | ".join(f"T{e['turn']}: {e['quote']}" for e in ev)
        row[f"ai_{k}_needs_review"] = (not c.get("answer")) or not ev
    row["ai_sentiment"] = (a.get("sentiment") or {}).get("label")
    row["ai_outcome"] = (a.get("outcome") or {}).get("label")
    row["ai_outcome_group"] = OUTCOME_GROUPS.get(row["ai_outcome"])
    row["ai_quality_score"] = call_quality(answers)
    row["ai_items_needing_review"] = sum(row[f"ai_{k}_needs_review"] for k in TRANSCRIPT_ITEMS)
    row["ai_summary"] = a.get("summary")
    row["ai_model"] = a.get("model")
    row["ai_analyzed_at"] = a.get("analyzed_at")
    return row


def _ai_answers(ai: pd.DataFrame) -> pd.DataFrame:
    """AI rows renamed to plain target names, for agreement tables."""
    return ai[["call_id"] + [f"ai_{t}" for t in TARGETS]].rename(columns={f"ai_{t}": t for t in TARGETS})


def _avg_row(table: pd.DataFrame, label: str) -> dict:
    items = table[table["target"].isin(TRANSCRIPT_ITEMS)]
    return {"group": label, "calls": int(items["n"].max()) if len(items) else 0,
            "checklist_agreement": items["agreement"].mean(), "checklist_kappa": items["kappa"].mean(),
            "sentiment_agreement": table.loc[table["target"] == "sentiment", "agreement"].mean(),
            "outcome_agreement": table.loc[table["target"] == "outcome", "agreement"].mean()}


def build(sources: list[CallSource], analyses: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """analyses: call ID -> AI analysis, as saved by analyze.py."""
    calls = load_calls(sources)
    labels = load_labels(sources)
    tagged = set(labels["call_id"])

    ai_rows, coaching = [], []
    for c in calls:
        a = analyses.get(c["call_id"])
        if a and a.get("checklist"):
            ai_rows.append(ai_flat(c["call_id"], a))
            for i, t in enumerate(a.get("coaching") or [], 1):
                coaching.append({"call_id": c["call_id"], "source": c["source"], "tip_no": i,
                                 "checklist_item": ITEM_NAMES.get(t.get("item"), t.get("item")),
                                 "tip": t.get("tip"), "transcript_turn": t.get("turn")})
    ai = pd.DataFrame(ai_rows, columns=["call_id"] if not ai_rows else None)

    # Human tags: consensus per call, plus a version built only from tags made with the AI off.
    human = consensus(labels) if len(labels) else pd.DataFrame(columns=["call_id"] + TARGETS)
    human_only_rows = labels[labels["llm_model"] == ""]
    human_only = consensus(human_only_rows) if len(human_only_rows) else pd.DataFrame(columns=["call_id"] + TARGETS)
    raters = labels.groupby("call_id").agg(raters=("rater", lambda s: ", ".join(sorted(s))),
                                           n_raters=("rater", "nunique"),
                                           tagged_with_ai_off=("llm_model", lambda s: bool((s == "").any())))

    # One row per call that has an AI analysis or a human tag.
    meta = pd.DataFrame([{k: c[k] for k in ("call_id", "source", "file_id")} for c in calls])
    keep = set(ai["call_id"]) if len(ai) else set()
    keep |= tagged
    out = meta[meta["call_id"].isin(keep)].merge(ai, on="call_id", how="left") if len(meta) else meta
    h = human.rename(columns={t: f"human_{t}" for t in TARGETS})
    out = out.merge(h, on="call_id", how="left").merge(raters, left_on="call_id", right_index=True, how="left")
    if len(out):
        out["human_quality_score"] = [call_quality({k: r.get(f"human_{k}") for k in TRANSCRIPT_ITEMS})
                                      for r in out.to_dict("records")]
        for t in TARGETS:
            if f"ai_{t}" not in out:
                continue
            both = out[f"human_{t}"].notna() & out[f"ai_{t}"].notna()
            out[f"agree_{t}"] = np.where(both, out[f"ai_{t}"] == out[f"human_{t}"], None)

    # Agreement: AI vs human, for several groups.
    ai_ans = _ai_answers(ai) if len(ai) else pd.DataFrame(columns=["call_id"] + TARGETS)
    groups = {"All tagged calls": human, "Real calls": human[human["call_id"].str.startswith("real/")],
              "Synthetic calls": human[human["call_id"].str.startswith("synthetic/")],
              "Tagged with AI off": human_only}
    per_target, overview = [], []
    for name, truth in groups.items():
        truth = truth[truth["call_id"].isin(set(ai_ans["call_id"]))]
        if not len(truth):
            continue
        t = agreement_table(truth, ai_ans)
        t.insert(0, "group", name)
        per_target.append(t)
        overview.append(_avg_row(t, name))
    agreement = pd.concat(per_target, ignore_index=True) if per_target else pd.DataFrame()
    overview = pd.DataFrame(overview)

    irr = inter_rater(labels) if labels["rater"].nunique() > 1 else pd.DataFrame()

    # Synthetic calls: do the transcripts express their planted labels? (human vs planted, AI vs planted)
    sanity = pd.DataFrame()
    syn = next((s for s in sources if not s.real), None)
    if syn is not None and (syn.transcripts_dir.parent / "truth" / "truth_calls.csv").exists():
        planted = load_synthetic(syn.transcripts_dir.parent)
        planted["call_id"] = "synthetic/" + planted["contact_id"] + ".txt"
        planted = planted[["call_id"] + TARGETS]
        parts = []
        for name, pred in (("Human vs planted label", human), ("AI vs planted label", ai_ans)):
            pred = pred[pred["call_id"].str.startswith("synthetic/")]
            if len(pred) and len(planted[planted["call_id"].isin(set(pred["call_id"]))]):
                t = agreement_table(planted[planted["call_id"].isin(set(pred["call_id"]))], pred)
                t.insert(0, "comparison", name)
                parts.append(t)
        sanity = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    counts = {
        "calls_available": len(calls),
        "calls_with_ai_analysis": len(ai),
        "calls_tagged": len(tagged),
        "real_tagged": sum(c.startswith("real/") for c in tagged),
        "synthetic_tagged": sum(c.startswith("synthetic/") for c in tagged),
        "tagged_calls_with_ai_analysis": len(tagged & set(ai["call_id"])) if len(ai) else 0,
        "tagged_calls_missing_ai_analysis": len(tagged - (set(ai["call_id"]) if len(ai) else set())),
        "tags_made_with_ai_off": int((labels["llm_model"] == "").sum()),
        "raters": ", ".join(sorted(labels["rater"].unique())),
    }
    return {"counts": pd.DataFrame([counts]), "overview": overview, "calls": out,
            "coaching": pd.DataFrame(coaching), "agreement": agreement, "rater_agreement": irr,
            "synthetic_check": sanity}


def _definitions() -> pd.DataFrame:
    rows = [{"term": ITEM_NAMES[k], "definition": d} for k, d in ITEM_DEFINITIONS.items()]
    rows += [{"term": f"Outcome: {k}", "definition": d} for k, d in OUTCOME_DEFINITIONS.items()]
    rows += [
        {"term": "Call-quality score", "definition": "Yes / (Yes + No) over the 7 checklist items; N/A excluded. "
                                                     "Provisional until Salud sets item weights."},
        {"term": "Agreement", "definition": "Share of calls where the AI gave the same answer as the human tag. "
                                            "N/A counts as an answer; a missing AI answer counts as a disagreement."},
        {"term": "Kappa", "definition": "Cohen's kappa: agreement beyond what matching the most common answer "
                                        "would give. 0 = chance, 1 = perfect; above 0.6 is usually considered good."},
        {"term": "Needs review", "definition": "The AI gave no answer, or gave one without a transcript quote "
                                               "that could be verified."},
        {"term": "Tagged with AI off", "definition": "Tags saved without pressing Analyze first. These are the fairest "
                                                     "benchmark, because pre-filled tags tend to follow the AI."},
    ]
    return pd.DataFrame(rows)


def write(results: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("calls", "coaching", "agreement", "rater_agreement", "synthetic_check"):
        if len(results[name]):
            results[name].to_csv(out_dir / f"{'ai_vs_human' if name == 'agreement' else name}.csv", index=False)
    xlsx = out_dir / "phase1_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        summary = results["counts"].T.reset_index()
        summary.columns = ["metric", "value"]
        summary.to_excel(xw, sheet_name="Summary", index=False)
        if len(results["overview"]):
            results["overview"].round(3).to_excel(xw, sheet_name="Summary", index=False, startrow=len(summary) + 2)
        sheets = [("Calls", "calls"), ("Coaching", "coaching"), ("AI vs human", "agreement"),
                  ("Rater agreement", "rater_agreement"), ("Synthetic check", "synthetic_check")]
        for sheet, key in sheets:
            df = results[key]
            (df.round(3) if len(df) else pd.DataFrame({"note": ["No data yet"]})).to_excel(xw, sheet_name=sheet, index=False)
        _definitions().to_excel(xw, sheet_name="Definitions", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:50])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
            ws.freeze_panes = "B2" if ws.title in ("Calls", "Coaching") else None
    (out_dir / "summary.md").write_text(summary_markdown(results))
    return xlsx


def summary_markdown(results: dict[str, pd.DataFrame]) -> str:
    c = results["counts"].iloc[0]
    lines = ["# Phase 1: AI call analysis vs manual tags", "",
             f"- Calls available: {c['calls_available']} | with AI analysis: {c['calls_with_ai_analysis']}",
             f"- Calls tagged by people: {c['calls_tagged']} (real {c['real_tagged']}, synthetic {c['synthetic_tagged']}); "
             f"raters: {c['raters'] or 'none yet'}",
             f"- Tagged calls missing an AI analysis: {c['tagged_calls_missing_ai_analysis']}",
             f"- Tags made with the AI off: {c['tags_made_with_ai_off']}", ""]
    ov = results["overview"]
    if len(ov):
        lines += ["## AI vs human (averages)", "", "| group | calls | checklist agreement | checklist kappa | sentiment | outcome |",
                  "|---|---|---|---|---|---|"]
        for r in ov.to_dict("records"):
            f = lambda v: "–" if pd.isna(v) else f"{v:.2f}"
            lines.append(f"| {r['group']} | {r['calls']} | {f(r['checklist_agreement'])} | {f(r['checklist_kappa'])} | "
                         f"{f(r['sentiment_agreement'])} | {f(r['outcome_agreement'])} |")
        ag = results["agreement"]
        allc = ag[ag["group"] == "All tagged calls"]
        if len(allc):
            lines += ["", "## Per item (all tagged calls, 95% CI)", "", "| item | calls | agreement | kappa |", "|---|---|---|---|"]
            for r in allc.to_dict("records"):
                k = "–" if pd.isna(r["kappa"]) else f"{r['kappa']:.2f}"
                lines.append(f"| {ITEM_NAMES[r['target']]} | {r['n']} | {r['agreement']:.2f} "
                             f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {k} |")
    else:
        lines += ["No tagged calls with an AI analysis yet: tag calls on the dashboard, then run `salud-phase1 analyze`."]
    lines += ["", "Small samples: with 10 real calls, each item's 95% CI spans about ±0.3. "
              "Treat per-item differences as indicative."]
    return "\n".join(lines) + "\n"
