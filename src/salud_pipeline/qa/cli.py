"""Phase 1: call analysis with the LLM, checked against manual tags.

    salud-qa tag --rater AK      # review page: 10 real + LLM-written synthetic calls, Analyze, tag, save
    salud-qa phase1              # AI vs human agreement + per-call export (reports/phase1/)
    salud-qa ping                # check the LLM endpoint (.env) and list its models
    salud-qa template            # blank labels sheet for the real calls

Endpoint config: GROQ_API_KEY / GROQ_MODEL in .env, or $SALUD_LLM_BASE_URL, $SALUD_LLM_MODEL,
$SALUD_LLM_API_KEY (or --base-url / --model).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..llm_client import LLMEndpoint
from ..schema import REPO_ROOT
from .sources import ASSIST_CACHE, REAL_DIR, REAL_LABELS, SYNTHETIC_RUN, phase1_sources
from .targets import write_template

PHASE1_OUT = REPO_ROOT / "reports" / "phase1"


def _endpoint(args) -> LLMEndpoint:
    ep = LLMEndpoint()
    ep.base_url = getattr(args, "base_url", None) or ep.base_url
    ep.model = getattr(args, "model", None) or ep.model
    return ep


def cmd_template(args) -> None:
    REAL_LABELS.parent.mkdir(parents=True, exist_ok=True)
    if REAL_LABELS.exists() and not args.force:
        sys.exit(f"{REAL_LABELS} exists; pass --force to overwrite")
    write_template(REAL_LABELS, REAL_DIR)
    print(f"Wrote {REAL_LABELS}: one row per real call.")


def cmd_tag(args) -> None:
    import webbrowser

    from .tag_server import serve

    sources = phase1_sources(args.run)
    present = [s for s in sources if s.transcripts_dir.exists()]
    if not present:
        sys.exit(f"No transcripts found in {REAL_DIR} or {args.run / 'transcripts'}")
    endpoint = None if args.no_llm else _endpoint(args)
    try:
        httpd = serve(present, args.rater.strip(), args.port, endpoint=endpoint, cache_dir=ASSIST_CACHE)
    except OSError as e:
        if e.errno != 48 and "in use" not in str(e):
            raise
        sys.exit(f"Port {args.port} is already in use, probably by an earlier `salud-qa tag` still running.\n"
                 f"Stop it with Ctrl+C in its terminal (or `lsof -ti:{args.port} | xargs kill`), "
                 f"or pick another port with --port {args.port + 1}.")
    url = f"http://127.0.0.1:{httpd.server_port}/"
    llm = "off" if endpoint is None else f"{endpoint.provider} / {endpoint.model or 'no model set'}"
    print(f"Rater {args.rater}. LLM: {llm}")
    for s in sources:
        n = len(list(s.transcripts_dir.glob("*.txt"))) if s.transcripts_dir.exists() else 0
        print(f"  {s.label:9s} {n:5d} calls  -> {s.labels_csv}" + ("" if n else "  (not found, skipped)"))
    print(f"Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_ping(args) -> None:
    ep = _endpoint(args)
    print(f"Endpoint: {ep.base_url}  (local={ep.is_local}, api key={'set' if ep.api_key else 'not set'})")
    models = ep.list_models()
    print("Models:", ", ".join(models) or "(none listed)")
    if ep.model:
        reply = ep.chat([{"role": "user", "content": 'Reply with the JSON {"ok": true}'}], json_mode=True,
                        max_tokens=200)
        print(f"{ep.model} replied: {reply[:200]}")


def cmd_phase1(args) -> None:
    from . import phase1
    from .assist import CallAssistant

    ep = _endpoint(args)
    assistant = CallAssistant(ep, ASSIST_CACHE) if ep.model else None
    if args.analyze_missing:
        if assistant is None:
            sys.exit("--analyze-missing needs an LLM: set GROQ_API_KEY in .env")
        ep.max_retries = 8
        print(f"Analyzing tagged calls that have no AI analysis yet with {ep.provider} / {ep.model} "
              "(real calls included: they are sent to this provider).")
    results = phase1.build(phase1_sources(args.run), assistant, analyze_missing=args.analyze_missing)
    xlsx = phase1.write(results, args.out)
    print(phase1.summary_markdown(results))
    if len(results["failed"]):
        print(f"{len(results['failed'])} AI analyses failed (e.g. Groq daily limit); rerun later to fill them in.")
    print(f"Saved {xlsx}\n  and CSVs + summary.md in {args.out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="blank labels sheet for the real calls")
    t.add_argument("--force", action="store_true")

    tag = sub.add_parser("tag", help="review page")
    tag.add_argument("--rater", required=True, help="your initials; each rater tags blind")
    tag.add_argument("--run", type=Path, default=SYNTHETIC_RUN, help="synthetic run listed with the real calls")
    tag.add_argument("--port", type=int, default=8765)
    tag.add_argument("--no-browser", action="store_true")
    tag.add_argument("--no-llm", action="store_true", help="tag without AI suggestions")

    p1 = sub.add_parser("phase1", help="AI vs human agreement and per-call export")
    p1.add_argument("--run", type=Path, default=SYNTHETIC_RUN)
    p1.add_argument("--out", type=Path, default=PHASE1_OUT)
    p1.add_argument("--analyze-missing", action="store_true",
                    help="run the LLM for tagged calls with no AI analysis yet (uses Groq quota)")

    ping = sub.add_parser("ping", help="check the LLM endpoint")
    for p in (tag, p1, ping):
        p.add_argument("--base-url", help="default: $SALUD_LLM_BASE_URL, Groq if GROQ_API_KEY is set")
        p.add_argument("--model", help="default: $SALUD_LLM_MODEL / $GROQ_MODEL")

    args = ap.parse_args(argv)
    try:
        {"template": cmd_template, "tag": cmd_tag, "ping": cmd_ping, "phase1": cmd_phase1}[args.cmd](args)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
