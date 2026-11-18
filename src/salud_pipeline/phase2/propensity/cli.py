"""Payment propensity, Step 1: the combined modeling sheet.

    salud-propensity build                 # real snapshots -> reports/phase2/propensity_dataset.xlsx,
                                           # synthetic snapshots -> reports/phase2/synthetic_snapshots.csv
                                           #   (one row per call) and synthetic_fixed_date.csv (per patient)
    salud-propensity sentiment --synthetic # score the 4,152 synthetic transcripts (local model, slow, cached)
    salud-propensity qa                    # Stage 1 QA of the 84 real calls with the dashboard's prompt
                                           #   (SENDS the redacted transcripts to the LLM provider; cached)

Real data is read from the Neon schema `final`; synthetic from data/phase2/synthetic_world. Sentiment runs
locally (nothing leaves the machine). Outputs are gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from ...paths import DEFAULT_WORLD, PHASE2_REPORTS, PHASE2_WORLDS
from . import dictionary as D
from .snapshots import WINDOW_DAYS, build, fixed_date_world, open_balance, score_qa, score_sentiment, synthetic_qa
from .modeling import world_files
from .sources import load_real, load_synthetic

DATASET_XLSX = PHASE2_REPORTS / "propensity_dataset.xlsx"
SENTIMENT_CACHE = PHASE2_REPORTS / "sentiment_cache"
REAL_QA = PHASE2_REPORTS / "qa_real.csv"          # from `salud-propensity qa`


def summary(df: pd.DataFrame) -> pd.DataFrame:
    usable = df[df["label_reliable"] & df["window_complete"]]
    follow = usable[~usable["paid_on_call"]]
    rows = [
        ("Rows (one per patient per call date)", len(df)),
        ("Patients", df["guarantor_id"].nunique()),
        ("Rows without a known patient", int(df["guarantor_id"].isna().sum())),
        ("Rows whose patient has no accounts in the extract (label unreliable)",
         int((df["guarantor_id"].notna() & ~df["label_reliable"]).sum())),
        (f"Rows without a full {WINDOW_DAYS}-day window", int((~df["window_complete"]).sum())),
        ("Usable rows for the pre-call model (reliable label + full window)", len(usable)),
        ("  of which paid within 30 days", int(usable["paid_30d"].sum())),
        ("  pay rate", round(usable["paid_30d"].mean(), 3) if len(usable) else None),
        ("Usable rows for the post-call model (also no payment taken on the call)", len(follow)),
        ("  of which paid in days 1-30 after the call", int(follow["paid_after_call_30d"].sum())),
        ("Payment taken on the call", int(df["paid_on_call"].sum())),
        ("Rows with caller sentiment", int(df["caller_sentiment_overall"].notna().sum())),
        ("Rows with a QA score", int(df["qa_score"].notna().sum())),
        ("As-of dates", f"{df['as_of'].min():%Y-%m-%d} to {df['as_of'].max():%Y-%m-%d}"),
        ("Self-pay rule", D.SELF_PAY_RULE),
        ("Snapshot fields NOT used (taken after the calls, would leak the outcome)", ", ".join(D.LEAKING_SNAPSHOT_FIELDS)),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def write_xlsx(real: pd.DataFrame, path=DATASET_XLSX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet = real.copy()
    sheet["as_of"] = sheet["as_of"].dt.strftime("%Y-%m-%d %H:%M")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sheet.to_excel(xw, sheet_name="Patient snapshots", index=False)
        D.as_frame().to_excel(xw, sheet_name="Data dictionary", index=False)
        summary(real).to_excel(xw, sheet_name="Summary", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in list(col)[:50])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)


def cmd_build(args) -> None:
    print("Loading the real package from Neon (schema final)…")
    real_world = load_real()
    sent = score_sentiment(real_world, SENTIMENT_CACHE / "real.csv")
    qa = pd.read_csv(REAL_QA).set_index("call_id")["qa_score"] if REAL_QA.exists() else None
    if qa is None:
        print("  (no real QA scores yet: `salud-propensity qa` sends the transcripts to the LLM provider)")
    real = build(real_world, sentiment=sent, qa=qa)
    write_xlsx(real)
    print(f"Wrote {DATASET_XLSX} ({len(real)} rows)")
    print(summary(real).head(13).to_string(index=False))

    print("\nLoading the synthetic world…")
    wdir = PHASE2_WORLDS[args.world]
    SYNTHETIC_CSV, SYNTHETIC_FIXED_CSV = world_files(args.world)
    syn_world = load_synthetic(wdir)
    cache = SENTIMENT_CACHE / ("synthetic.csv" if args.world == "original" else f"synthetic_{args.world}.csv")
    syn_sent = pd.read_csv(cache) if cache.exists() else None
    if syn_sent is None:
        print(f"  (no synthetic sentiment yet: run `salud-propensity sentiment --synthetic --world {args.world}`, then build again)")
    syn = build(syn_world, sentiment=syn_sent, qa=synthetic_qa(wdir))
    syn.to_csv(SYNTHETIC_CSV, index=False)
    s = summary(syn)
    print(f"Wrote {SYNTHETIC_CSV} ({len(syn)} rows); usable pre-call rows "
          f"{s.loc[5, 'Value']}, pay rate {s.loc[7, 'Value']}")

    # Fixed-date design: every patient with an open balance, scored on the world's cutoff date.
    cutoff = pd.Timestamp(json.loads((wdir / "manifest.json").read_text())["config"]["prediction_cutoff"])
    fixed = open_balance(build(fixed_date_world(syn_world, cutoff)))
    fixed.to_csv(SYNTHETIC_FIXED_CSV, index=False)
    print(f"Wrote {SYNTHETIC_FIXED_CSV} ({len(fixed)} patients with an open balance on {cutoff:%Y-%m-%d}); "
          f"pay rate {fixed['paid_30d'].mean():.3f}")


def cmd_sentiment(args) -> None:
    if args.synthetic:
        world = load_synthetic(PHASE2_WORLDS[args.world])
        name = "synthetic.csv" if args.world == "original" else f"synthetic_{args.world}.csv"
    else:
        world, name = load_real(), "real.csv"
    score_sentiment(world, SENTIMENT_CACHE / name)


def cmd_qa(args) -> None:
    world = load_real()
    print(f"Stage 1 QA of {len(world.transcripts)} real calls: the redacted transcripts are sent to the LLM provider.")
    score_qa(world, REAL_QA, limit=args.limit)
    print(f"Wrote {REAL_QA}. Run `salud-propensity build` to add the scores to the real snapshots.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="salud-propensity", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the combined sheet (real) and synthetic snapshots")
    b.add_argument("--world", choices=sorted(PHASE2_WORLDS), default=DEFAULT_WORLD, help="which synthetic world")
    s = sub.add_parser("sentiment", help="score transcripts with the local sentiment model (cached)")
    s.add_argument("--synthetic", action="store_true", help="the synthetic world instead of the real package")
    s.add_argument("--world", choices=sorted(PHASE2_WORLDS), default=DEFAULT_WORLD)
    q = sub.add_parser("qa", help="Stage 1 QA of the real calls (sends transcripts to the LLM provider; cached)")
    q.add_argument("--limit", type=int, help="maximum new LLM requests this run")
    args = ap.parse_args(argv)
    try:
        {"build": cmd_build, "sentiment": cmd_sentiment, "qa": cmd_qa}[args.cmd](args)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
