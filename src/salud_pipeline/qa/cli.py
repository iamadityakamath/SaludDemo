"""Phase 4 QA scoring: LLM vs traditional ML, validated against synthetic truth and human tags.

    salud-qa template                      # blank ground-truth sheet for the 10 real calls
    salud-qa tag --rater AK                # local tagging page for the real calls
    salud-qa ping                          # check the team LLM endpoint and list its models
    salud-qa ml   --run data/synthetic/run_default
    salud-qa llm  --run data/synthetic/run_default --n 150 [--include-real]
    salud-qa summary                       # side-by-side table from the saved reports

Endpoint config: $SALUD_LLM_BASE_URL, $SALUD_LLM_MODEL, $SALUD_LLM_API_KEY (or --base-url/--model).
Reports go to reports/qa/.

Real (redacted) client transcripts are sent to a non-local endpoint only with --include-real.
Pass it only once that server is confirmed to sit inside the HIPAA-approved environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..llm_client import LLMEndpoint
from ..schema import REPO_ROOT
from .evaluate import bootstrap_accuracy_ci, consensus, inter_rater, score_table
from .llm_scorer import LLMScorer
from .ml_scorer import MLScorer
from .targets import TARGETS, load_ground_truth, load_synthetic, write_template

REAL_DIR = REPO_ROOT / "data" / "Redacted Transcripts with Inferred Speakers"
GT_CSV = REPO_ROOT / "data" / "ground_truth" / "real_calls_labels.csv"
REPORTS = REPO_ROOT / "reports" / "qa"
SIZES = [25, 50, 100, 200, 500, 1000, 2000]


def split(df: pd.DataFrame, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.random.default_rng(seed).permutation(len(df))
    n_test = int(len(df) * test_frac)
    return df.iloc[idx[n_test:]], df.iloc[idx[:n_test]]


def _ground_truth() -> pd.DataFrame | None:
    if not GT_CSV.exists():
        return None
    gt = load_ground_truth(GT_CSV, REAL_DIR)
    filled = gt.dropna(subset=["sentiment"])
    if filled.empty:
        return None
    if filled["rater"].nunique() > 1:
        irr = inter_rater(filled)
        irr.to_csv(REPORTS / "inter_rater.csv", index=False)
        print("Inter-rater kappa by target:\n", irr.groupby("target")["kappa"].mean().round(2).to_string())
    return consensus(filled)


def cmd_template(args) -> None:
    GT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if GT_CSV.exists() and not args.force:
        sys.exit(f"{GT_CSV} exists; pass --force to overwrite")
    write_template(GT_CSV, REAL_DIR)
    print(f"Wrote {GT_CSV}: one row per real call. Duplicate rows per rater, fill rater initials.")


def cmd_tag(args) -> None:
    import webbrowser

    from .tag_server import serve

    if args.run:
        transcripts, csv = args.run / "transcripts", GT_CSV.with_name("synthetic_practice_labels.csv")
        if args.limit:  # practice on a small, fixed subset of a synthetic run
            subset = REPORTS.parent / "tag_practice" / args.run.name
            subset.mkdir(parents=True, exist_ok=True)
            for p in sorted(transcripts.glob("*.txt"))[: args.limit]:
                (subset / p.name).write_text(p.read_text())
            transcripts = subset
    else:
        transcripts, csv = REAL_DIR, GT_CSV
    if not transcripts.exists():
        sys.exit(f"No transcripts at {transcripts}")
    httpd = serve(csv, transcripts, args.rater.strip(), args.port)
    url = f"http://127.0.0.1:{httpd.server_port}/"
    print(f"Tagging {transcripts.name} as {args.rater} -> {csv}\nOpen {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_ml(args) -> None:
    data = load_synthetic(args.run)
    train, test = split(data, 0.25, args.seed)
    rows = []
    for size in [s for s in SIZES if s <= len(train)] + [len(train)]:
        for rep in range(args.reps):
            sub = train.sample(n=size, random_state=args.seed + rep)
            st = score_table(test, MLScorer().fit(sub).predict(test))
            st["train_size"], st["rep"] = size, rep
            rows.append(st)
    curve = pd.concat(rows).groupby(["train_size", "target"])[["accuracy", "macro_f1", "kappa"]].mean().reset_index()
    curve.to_csv(REPORTS / "ml_learning_curve.csv", index=False)
    print("ML learning curve (accuracy on synthetic test set):")
    print(curve.pivot(index="target", columns="train_size", values="accuracy").round(3).to_string())

    full = MLScorer().fit(train)
    final = score_table(test, full.predict(test))
    final.to_csv(REPORTS / "ml_synthetic_test.csv", index=False)
    gt = _ground_truth()
    if gt is not None:
        real = score_table(gt, full.predict(gt))
        real.to_csv(REPORTS / "ml_real_calls.csv", index=False)
        print("\nML on real hand-tagged calls:\n", real.round(3).to_string(index=False))


def _endpoint(args) -> LLMEndpoint:
    ep = LLMEndpoint()
    ep.base_url = args.base_url or ep.base_url
    ep.model = args.model or ep.model
    return ep


def cmd_ping(args) -> None:
    ep = _endpoint(args)
    print(f"Endpoint: {ep.base_url}  (local={ep.is_local}, api key={'set' if ep.api_key else 'not set'})")
    models = ep.list_models()
    print("Models:", ", ".join(models) or "(none listed)")
    if ep.model:
        reply = ep.chat([{"role": "user", "content": 'Reply with the JSON {"ok": true}'}], json_mode=True)
        print(f"{ep.model} replied: {reply[:200]}")


def cmd_llm(args) -> None:
    ep = _endpoint(args)
    if not ep.model:
        sys.exit("No model: pass --model or set SALUD_LLM_MODEL (run `salud-qa ping` to list models)")
    tag = ep.model.replace(":", "_").replace("/", "_")
    scorer = LLMScorer(ep, REPORTS / f"llm_cache_{tag}.jsonl")
    data = load_synthetic(args.run)
    _, test = split(data, 0.25, args.seed)
    test = test.head(args.n)
    print(f"LLM {args.model}: scoring {len(test)} synthetic test calls")
    pred = scorer.predict(test)
    pred.to_csv(REPORTS / f"llm_{tag}_synthetic_predictions.csv", index=False)
    st = score_table(test, pred)
    st["acc_ci_low"], st["acc_ci_high"] = zip(*[bootstrap_accuracy_ci(test, pred, t) for t in st["target"]])
    st.to_csv(REPORTS / f"llm_{tag}_synthetic_test.csv", index=False)
    print(st.round(3).to_string(index=False))
    gt = _ground_truth()
    if gt is not None and not ep.is_local and not args.include_real:
        print(f"\nSkipping {len(gt)} real calls: {ep.base_url} is not local. Re-run with --include-real "
              "once the server is confirmed inside the HIPAA-approved environment.")
        gt = None
    if gt is not None:
        print(f"\nScoring {len(gt)} real hand-tagged calls")
        rp = scorer.predict(gt)
        rp.to_csv(REPORTS / f"llm_{tag}_real_predictions.csv", index=False)
        real = score_table(gt, rp)
        real.to_csv(REPORTS / f"llm_{tag}_real_calls.csv", index=False)
        print(real.round(3).to_string(index=False))


def cmd_summary(args) -> None:
    frames = []
    for p in sorted(REPORTS.glob("*_test.csv")) + sorted(REPORTS.glob("*_real_calls.csv")):
        df = pd.read_csv(p)[["target", "accuracy", "kappa"]]
        df["report"] = p.stem
        frames.append(df)
    if not frames:
        sys.exit("No reports yet.")
    table = pd.concat(frames).pivot(index="target", columns="report", values="accuracy").reindex(TARGETS)
    print(table.round(3).to_string())
    table.to_csv(REPORTS / "summary_accuracy.csv")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--force", action="store_true")
    for name in ("ml", "llm"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, default=REPO_ROOT / "data" / "synthetic" / "run_default")
        p.add_argument("--seed", type=int, default=0)
    sub.choices["ml"].add_argument("--reps", type=int, default=3)
    tag = sub.add_parser("tag")
    tag.add_argument("--rater", required=True, help="your initials; each rater tags blind")
    tag.add_argument("--run", type=Path, help="tag a synthetic run instead (practice / calibration)")
    tag.add_argument("--limit", type=int, default=20, help="calls to include from a synthetic run")
    tag.add_argument("--port", type=int, default=8765)
    tag.add_argument("--no-browser", action="store_true")
    ping = sub.add_parser("ping")
    for p in (sub.choices["llm"], ping):
        p.add_argument("--base-url", help="default: $SALUD_LLM_BASE_URL or local Ollama")
        p.add_argument("--model", help="default: $SALUD_LLM_MODEL")
    sub.choices["llm"].add_argument("--n", type=int, default=150, help="synthetic test calls to score")
    sub.choices["llm"].add_argument("--include-real", action="store_true",
                                    help="also send the real redacted calls to a non-local endpoint")
    sub.add_parser("summary")
    args = ap.parse_args(argv)
    REPORTS.mkdir(parents=True, exist_ok=True)
    try:
        {"template": cmd_template, "tag": cmd_tag, "ping": cmd_ping, "ml": cmd_ml, "llm": cmd_llm,
         "summary": cmd_summary}[args.cmd](args)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
