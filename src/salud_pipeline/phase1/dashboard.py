"""Phase 1 route 1: the dashboard for reviewing and tagging calls, backed by the team's Neon database.

    salud-phase1 dashboard --rater AK

Runs on each teammate's laptop (127.0.0.1 only) and talks to the shared Neon database (store.py):
calls and transcripts, the AI analyses that `salud-phase1 analyze` added, and every rater's tags.
Each call opens pre-filled with its AI answers, quotes, sentiment, outcome and coaching tips; the rater
checks, corrects and saves. Saving writes that rater's row in phase1.reviewed_tags, recording the model
whose answers pre-filled it (llm_model) and the items the rater changed (overrides).

The dashboard never calls the LLM and never changes AI analyses.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..paths import REPO_ROOT
from .checklist import ITEM_DEFINITIONS, OUTCOMES, SENTIMENTS, TARGETS, TRANSCRIPT_ITEMS
from .prompt import split_turns
from .store import Store

BRANDING_DIR = REPO_ROOT / "branding"
LOGO_NAMES = ["logo.svg", "logo.png", "logo.webp", "logo.jpg", "logo.jpeg"]
SOURCE_LABELS = {"real": "Real", "synthetic": "Synthetic"}


def _logo_path() -> Path | None:
    return next((BRANDING_DIR / n for n in LOGO_NAMES if (BRANDING_DIR / n).exists()), None)


def make_handler(store: Store, rater: str):
    page = files("salud_pipeline.phase1").joinpath("static/dashboard.html").read_text()
    turns_cache: dict[str, dict] = {}
    cache_lock = threading.Lock()

    def transcript(call_id: str) -> dict | None:
        with cache_lock:
            if call_id in turns_cache:
                return turns_cache[call_id]
        text = store.transcript(call_id)
        if text is None:
            return None
        out = {"text": text, "turns": split_turns(text)}
        with cache_lock:
            turns_cache[call_id] = out
        return out

    def state() -> dict:
        calls = store.calls()
        tags = store.tags(rater)
        labels = {r["call_id"]: {k: (r[k] or "") for k in TARGETS + ["notes"]} for r in tags.to_dict("records")}
        sources = list(dict.fromkeys(c["source"] for c in calls))
        ai = store.ai_info() | {"file": "the team database"}
        return {"config": {"rater": rater, "items": TRANSCRIPT_ITEMS, "definitions": ITEM_DEFINITIONS,
                           "sentiments": SENTIMENTS, "outcomes": OUTCOMES, "ai": ai,
                           "sources": [{"key": s, "label": SOURCE_LABELS.get(s, s)} for s in sources],
                           "has_logo": _logo_path() is not None},
                "calls": [{"id": c["call_id"], "file_id": c["file_id"], "source": c["source"]} for c in calls],
                "labels": labels}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200):
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            cid = parse_qs(url.query).get("id", [""])[0]
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, page.encode(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(state())
                elif url.path == "/api/call":
                    t = transcript(cid)
                    self._json(t) if t else self._json({"error": "unknown call"}, 404)
                elif url.path == "/api/analysis":
                    # The saved AI analysis for this call (read-only); the dashboard never calls the LLM.
                    if store.transcript(cid) is None:
                        return self._json({"error": "unknown call"}, 404)
                    self._json(store.ai_analysis(cid) or {"none": True})
                elif url.path == "/brand/logo":
                    p = _logo_path()
                    if p is None:
                        return self._json({"error": "no logo"}, 404)
                    self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
                else:
                    self._json({"error": "not found"}, 404)
            except RuntimeError as e:  # e.g. database unreachable
                self._json({"error": str(e)}, 503)

        def do_POST(self):
            if urlparse(self.path).path != "/api/label":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                store.save_tag(rater, body["id"], body["values"], body.get("llm"))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, 400)
            self._json({"ok": True})

        def log_message(self, *args):
            pass

    return Handler


def serve(rater: str, port: int = 8765, store: Store | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(store or Store(), rater))
