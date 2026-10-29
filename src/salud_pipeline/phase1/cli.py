"""Phase 1: call analysis on the team's shared Neon database. Two routes.

Route 1, dashboard (each call's stored analysis; new transcripts are analyzed automatically):
    salud-phase1 assign --raters avni,sharayu,aditya,ibrahim,ananya,rudrakshi   # team lead, once
    salud-phase1 dashboard --rater avni
    salud-phase1 process      # analyze stored transcripts missing sentiment or an AI review, then exit

Route 2, script (AI analysis of every call, then the comparison):
    salud-phase1 analyze      # LLM on calls without an analysis -> Neon (phase1.ai_analyses)
    salud-phase1 compare      # AI vs every rater's tags -> reports/phase1/comparison/ (local Excel/CSV)

Settings come from .env: DATABASE_URL (team database), GROQ_API_KEY / GROQ_MODEL (only `analyze` uses the LLM).
"""

from __future__ import annotations

import argparse
import sys

from ..llm_client import LLMEndpoint
from ..paths import COMPARISON, LLM_CACHE
from .assign import rater_name
from .store import Store, as_label_frame


def cmd_dashboard(args) -> None:
    import webbrowser

    from .dashboard import serve

    store = Store()
    calls, ai = store.calls(), store.ai_info()
    if not calls:
        sys.exit("The team database has no calls yet. The team lead runs `salud-db setup` once to load them.")
    rater = rater_name(args.rater)
    assigned = store.assignments()
    mine = sum(rater in rs for rs in assigned.values())
    if assigned and not mine:
        names = sorted({r for rs in assigned.values() for r in rs})
        print(f"Note: no calls are assigned to '{rater}'. Assigned raters: {', '.join(names)}. "
              "Check the spelling, or pick 'All raters' on the page.")
    try:
        httpd = serve(rater, args.port, store=store, auto=not args.no_auto)
    except OSError as e:
        if e.errno != 48 and "in use" not in str(e):
            raise
        sys.exit(f"Port {args.port} is already in use, probably by an earlier dashboard still running.\n"
                 f"Stop it with Ctrl+C in its terminal (or `lsof -ti:{args.port} | xargs kill`), "
                 f"or pick another port with --port {args.port + 1}.")
    url = f"http://127.0.0.1:{httpd.server_port}/"
    by_source = {s: sum(c["source"] == s for c in calls) for s in ("real", "synthetic")}
    print(f"Rater {rater} ({mine} calls assigned). Team database: {by_source['real']} real + {by_source['synthetic']} synthetic calls, "
          f"{ai['analyzed']} with AI answers ({ai['model'] or 'none yet'}).")
    if not args.no_auto:
        from .dashboard import start_sweeper

        start_sweeper(store)
        print("Uploaded calls are analyzed right away; other new transcripts in the database within 15 seconds.")
    print(f"Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_process(args) -> None:
    from . import live

    store = Store()
    if not live.llm_ready():
        print("No LLM configured (GROQ_API_KEY): only sentiment is scored, AI reviews are left for later.")
    r = live.process_pending(store, limit=args.limit, log=lambda cid, _: print(f"  analyzed {cid}"))
    print(f"Analyzed {r['done']} of {r['pending']} calls that were missing results.")
    if r["stopped"]:
        print(f"Stopped early: {r['stopped']}")
    if r["left"]:
        print(f"{r['left']} calls left. Run `salud-phase1 process` again.")


def cmd_assign(args) -> None:
    from .assign import plan

    store = Store()
    calls = store.calls()
    if args.raters:
        pairs = plan(calls, args.raters.split(","), store.assignments(), real_per_call=args.real_per_call)
        print(f"Added {store.assign(pairs)} assignments.")
    assigned = store.assignments()
    if not assigned:
        sys.exit("No calls are assigned yet. Run: salud-phase1 assign --raters name1,name2,...")
    source = {c["call_id"]: c["source"] for c in calls}
    tagged = {(r["call_id"], r["rater"]) for r in store.tags().to_dict("records")}
    print(f"\n{'rater':12s} {'real':>5s} {'synth':>6s} {'total':>6s} {'tagged':>7s}")
    for r in sorted({r for rs in assigned.values() for r in rs}):
        mine = [cid for cid, rs in assigned.items() if r in rs]
        real = sum(source.get(cid) == "real" for cid in mine)
        done = sum((cid, r) in tagged for cid in mine)
        print(f"{r:12s} {real:5d} {len(mine) - real:6d} {len(mine):6d} {done:7d}")
    short = [c["call_id"] for c in calls if c["call_id"] not in assigned]
    print(f"\n{len(assigned)} of {len(calls)} calls assigned" + (f"; unassigned: {len(short)}" if short else "."))


def cmd_analyze(args) -> None:
    from .analyze import run
    from .prompt import PROMPT_VERSION, CallAssistant

    ep = LLMEndpoint()
    if not ep.model:
        sys.exit("No LLM configured: set GROQ_API_KEY in .env")
    ep.max_retries = 8  # wait out per-minute rate limits instead of failing
    print(f"Analyzing calls with {ep.provider} / {ep.model} (prompt {PROMPT_VERSION}). "
          "Real calls are included and sent to this provider.")
    c = run(Store(), CallAssistant(ep, LLM_CACHE), limit=args.limit)
    print(f"\n{c['analyzed_total']} of {c['calls']} calls now have an AI analysis in the team database")
    print(f"  this run: {c['new_llm_requests']} new LLM requests, {c['from_cache']} from cache, "
          f"{c['already_analyzed']} already analyzed, {c['failed']} failed")
    if c["not_done"]:
        why = f"stopped early ({c['stopped_early']})" if c["stopped_early"] else f"--limit {args.limit} reached"
        print(f"  {c['not_done']} calls left: {why}. Run `salud-phase1 analyze` again later to continue.")


def cmd_compare(args) -> None:
    from . import compare

    store = Store()
    analyses = store.ai_analyses()
    if not analyses:
        print("Note: no AI analyses in the team database yet. Run `salud-phase1 analyze` first.")
    results = compare.build(store.calls(), as_label_frame(store.tags()), analyses, store.planted())
    xlsx = compare.write(results, COMPARISON)
    print(compare.summary_markdown(results))
    info = store.ai_info()
    if analyses:
        print(f"AI analyses: {info['model']} / {info['prompt_version']}")
    print(f"Saved {xlsx}\n  plus CSVs and summary.md in {COMPARISON}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="salud-phase1", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dashboard", help="route 1: the agent dashboard (analyzes new transcripts automatically)")
    d.add_argument("--rater", required=True, help="your first name, as used in `assign` (e.g. avni)")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--no-browser", action="store_true")
    d.add_argument("--no-auto", action="store_true", help="don't analyze uploads or new transcripts in the background")

    pr = sub.add_parser("process", help="analyze stored transcripts missing sentiment or an AI review")
    pr.add_argument("--limit", type=int, help="maximum calls this run")

    s = sub.add_parser("assign", help="split the calls across raters (no options: show who has what)")
    s.add_argument("--raters", help="comma-separated first names, e.g. avni,sharayu,aditya")
    s.add_argument("--real-per-call", type=int, default=2, help="raters per real call (default 2)")

    a = sub.add_parser("analyze", help="route 2, step 1: LLM analysis of every call -> team database")
    a.add_argument("--limit", type=int, help="maximum new LLM requests this run")

    sub.add_parser("compare", help="route 2, step 2: AI vs every rater's tags -> Excel + CSV")

    args = ap.parse_args(argv)
    try:
        {"dashboard": cmd_dashboard, "process": cmd_process, "assign": cmd_assign, "analyze": cmd_analyze, "compare": cmd_compare}[args.cmd](args)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
