"""Phase 1: call analysis. Two routes.

Route 1, dashboard (assists manual tagging):
    salud-phase1 dashboard --rater AK            # Analyze button pre-fills the checklist
    salud-phase1 dashboard --rater AK --no-llm   # tag with the AI switched off

Route 2, script (AI analysis of every call, then the comparison):
    salud-phase1 analyze      # LLM on the 60 Phase 1 calls -> reports/phase1/ai_analyses.json
    salud-phase1 compare      # AI vs manual tags -> reports/phase1/comparison/

LLM settings come from .env (GROQ_API_KEY, GROQ_MODEL) or SALUD_LLM_BASE_URL / SALUD_LLM_MODEL /
SALUD_LLM_API_KEY.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..llm_client import LLMEndpoint
from ..paths import AI_ANALYSES, COMPARISON, LLM_CACHE, PHASE1_SYNTHETIC_CALLS, REAL_TRANSCRIPTS
from .calls import phase1_sources


def cmd_dashboard(args) -> None:
    import webbrowser

    from .dashboard import serve

    sources = [s for s in phase1_sources(args.calls) if s.transcripts_dir.exists()]
    if not sources:
        sys.exit(f"No transcripts found in {REAL_TRANSCRIPTS} or {args.calls / 'transcripts'}")
    endpoint = None if args.no_llm else LLMEndpoint()
    try:
        httpd = serve(sources, args.rater.strip(), args.port, endpoint=endpoint, cache_dir=LLM_CACHE)
    except OSError as e:
        if e.errno != 48 and "in use" not in str(e):
            raise
        sys.exit(f"Port {args.port} is already in use, probably by an earlier dashboard still running.\n"
                 f"Stop it with Ctrl+C in its terminal (or `lsof -ti:{args.port} | xargs kill`), "
                 f"or pick another port with --port {args.port + 1}.")
    url = f"http://127.0.0.1:{httpd.server_port}/"
    llm = "off" if endpoint is None else f"{endpoint.provider} / {endpoint.model or 'no model set'}"
    print(f"Rater {args.rater}. LLM: {llm}")
    for s in sources:
        print(f"  {s.label:9s} {len(s.files()):3d} calls  -> tags saved to {s.labels_csv}")
    print(f"Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_analyze(args) -> None:
    from .analyze import run
    from .prompt import PROMPT_VERSION, CallAssistant

    ep = LLMEndpoint()
    if not ep.model:
        sys.exit("No LLM configured: set GROQ_API_KEY in .env")
    ep.max_retries = 8  # wait out per-minute rate limits instead of failing
    print(f"Analyzing Phase 1 calls with {ep.provider} / {ep.model} (prompt {PROMPT_VERSION}). "
          "Real calls are included and sent to this provider.")
    c = run(phase1_sources(args.calls), CallAssistant(ep, LLM_CACHE), AI_ANALYSES, limit=args.limit)
    print(f"\n{c['in_file']} of {c['calls']} calls now in {AI_ANALYSES}")
    print(f"  this run: {c['new_llm_requests']} new LLM requests, {c['from_cache']} from cache, "
          f"{c['already_in_file']} already in the file, {c['failed']} failed")
    if c["not_done"]:
        why = f"stopped early ({c['stopped_early']})" if c["stopped_early"] else f"--limit {args.limit} reached"
        print(f"  {c['not_done']} calls left: {why}. Run `salud-phase1 analyze` again later to continue.")


def cmd_compare(args) -> None:
    from . import compare
    from .analyze import load

    data = load(AI_ANALYSES)
    analyses = {cid: entry["analysis"] for cid, entry in data.get("calls", {}).items()}
    if not analyses:
        print(f"Note: no AI analyses yet ({AI_ANALYSES}). Run `salud-phase1 analyze` first.")
    results = compare.build(phase1_sources(args.calls), analyses)
    xlsx = compare.write(results, COMPARISON)
    print(compare.summary_markdown(results))
    if analyses:
        print(f"AI analyses: {data.get('model')} / {data.get('prompt_version')}, updated {data.get('updated_at')}")
    print(f"Saved {xlsx}\n  plus CSVs and summary.md in {COMPARISON}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="salud-phase1", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dashboard", help="route 1: dashboard that assists manual tagging")
    d.add_argument("--rater", required=True, help="your initials; each rater's tags are kept separately")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--no-browser", action="store_true")
    d.add_argument("--no-llm", action="store_true", help="hide AI suggestions while tagging")

    a = sub.add_parser("analyze", help="route 2, step 1: LLM analysis of every call -> JSON")
    a.add_argument("--limit", type=int, help="maximum new LLM requests this run")

    sub.add_parser("compare", help="route 2, step 2: AI vs manual tags -> Excel + CSV")

    for p in sub.choices.values():
        p.add_argument("--calls", type=Path, default=PHASE1_SYNTHETIC_CALLS, help=argparse.SUPPRESS)

    args = ap.parse_args(argv)
    try:
        {"dashboard": cmd_dashboard, "analyze": cmd_analyze, "compare": cmd_compare}[args.cmd](args)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
