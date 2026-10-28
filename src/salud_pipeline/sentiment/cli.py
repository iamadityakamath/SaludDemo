"""salud-sentiment: turn-level call sentiment, run locally.

    salud-sentiment run [--source real|synthetic|new] [--redo]   score calls not scored yet -> team database
    salud-sentiment eval                                         call labels vs planted labels, raters' tags, the LLM

`run` reads transcripts from the team database (DATABASE_URL in .env), scores them on this machine and writes
phase1.turn_sentiment / phase1.call_sentiment (numbers and labels only). `eval` only reads. Transcripts are
never printed.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from . import aggregate
from .aggregate import as_tag


def _store():
    from ..phase1.store import Store

    return Store()


def cmd_run(args) -> None:
    from .model import MODEL_NAME, SentimentModel
    from .results import score_and_save

    store = _store()
    calls = [c for c in store.calls(include_new=True) if not args.source or c["source"] == args.source]
    done = set() if args.redo else set(store.sentiment_summaries())
    todo = [c for c in calls if c["call_id"] not in done]
    if not todo:
        print(f"All {len(calls)} calls are already scored. Use --redo to score them again.")
        return
    print(f"Loading {MODEL_NAME} (first run downloads ~130 MB)…")
    model = SentimentModel()
    t0 = time.time()
    for c in todo:
        score_and_save(store, c["call_id"], store.transcript(c["call_id"]) or "", model, MODEL_NAME)
    print(f"Scored {len(todo)} calls in {time.time() - t0:.0f}s -> team database "
          f"(phase1.turn_sentiment, phase1.call_sentiment); {len(calls) - len(todo)} were already scored.")
    summary = pd.DataFrame(store.sentiment_summaries().values())
    for role in ("caller", "agent"):
        if f"{role}_label" in summary:
            col = summary[f"{role}_label"].value_counts().reindex(aggregate.LABELS, fill_value=0)
            print(f"  {role} call labels: " + ", ".join(f"{k} {v}" for k, v in col.items()))


def _agreement(pred: pd.Series, truth: pd.Series) -> str:
    ok = truth.notna() & (truth != "")
    if not ok.any():
        return "n/a (none yet)"
    p, t = pred[ok].map(as_tag), truth[ok]
    return f"{(p == t).mean():.0%} of {ok.sum()}"


def _confusion(pred: pd.Series, truth: pd.Series) -> pd.DataFrame:
    ok = truth.notna() & (truth != "")
    return pd.crosstab(truth[ok].rename("truth"), pred[ok].map(as_tag).rename("model"))


def cmd_eval(args) -> None:
    store = _store()
    scored = store.sentiment_summaries()
    if not scored:
        sys.exit("No results yet. Run `salud-sentiment run` first.")
    calls = pd.DataFrame.from_dict(scored, orient="index")
    calls = calls[~calls.index.str.startswith("new/")]   # pasted calls have no references
    truths = {}
    planted = store.planted().set_index("call_id")
    if "sentiment" in planted:
        truths["planted (synthetic ground truth)"] = planted["sentiment"]
    llm = {cid: (a.get("sentiment") or {}).get("label") for cid, a in store.ai_analyses().items()}
    truths["LLM analysis label"] = pd.Series(llm, dtype=object)
    tags = store.tags()
    tags = tags[tags["sentiment"].fillna("") != ""]

    print(f"Caller sentiment, model label vs each reference (mixed counts as neutral). "
          f"Thresholds: positive >= {aggregate.POS}, negative <= {aggregate.NEG}.\n")
    print(f"{'reference':38} {'overall label':>16} {'end label':>16}")
    for name, truth in truths.items():
        truth = truth.reindex(calls.index)
        print(f"{name:38} {_agreement(calls['caller_label'], truth):>16} {_agreement(calls['caller_end_label'], truth):>16}")
    if len(tags):
        joined = tags.join(calls[["caller_label", "caller_end_label"]], on="call_id", how="inner")
        print(f"{'raters tags (every rater x call)':38} {_agreement(joined['caller_label'], joined['sentiment']):>16} "
              f"{_agreement(joined['caller_end_label'], joined['sentiment']):>16}")
    else:
        print(f"{'raters tags':38} {'n/a (none yet)':>16}")

    if args.detail:
        for name, truth in truths.items():
            print(f"\n{name}, overall label:\n{_confusion(calls['caller_label'], truth.reindex(calls.index))}")

    # Which symmetric threshold would have matched the references best? A guide for tuning, not a fit to apply blindly.
    refs = [t.reindex(calls.index) for t in truths.values()]
    if len(tags):
        refs.append(tags.set_index("call_id")["sentiment"])
    print("\nThreshold sweep on caller overall score (agreement with all references pooled):")
    for th in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        lab = calls["caller_overall"].map(lambda s: "positive" if s >= th else "negative" if s <= -th else "neutral")
        hits = total = 0
        for ref in refs:
            ref = ref[ref.notna() & (ref != "")]
            pred = lab.reindex(ref.index)
            ok = pred.notna()
            hits += (pred[ok] == ref[ok]).sum()
            total += ok.sum()
        print(f"  ±{th:<5} {hits / total:.0%} of {total}" if total else f"  ±{th:<5} n/a")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="salud-sentiment", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score calls not scored yet, locally -> team database")
    r.add_argument("--source", choices=["real", "synthetic", "new"], help="only these calls (default: all)")
    r.add_argument("--redo", action="store_true", help="score already-scored calls again")
    r.set_defaults(func=cmd_run)
    e = sub.add_parser("eval", help="call labels vs planted labels, raters' tags and the LLM's label")
    e.add_argument("--detail", action="store_true", help="also print confusion tables")
    e.set_defaults(func=cmd_eval)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
