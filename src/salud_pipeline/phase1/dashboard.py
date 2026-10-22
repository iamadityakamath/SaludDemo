"""Phase 1 route 1: the dashboard for reviewing and tagging calls, backed by the team's Neon database.

    salud-phase1 dashboard --rater avni

Runs on each teammate's laptop (127.0.0.1 only) and talks to the shared Neon database (store.py):
calls and transcripts, the AI analyses that `salud-phase1 analyze` added, and every rater's tags.
The list opens on the calls assigned to this rater (salud-phase1 assign). Each call opens pre-filled with its AI answers, quotes, sentiment, outcome and coaching tips; the rater
checks, corrects and saves. Saving writes that rater's row in phase1.reviewed_tags, recording the model
whose answers pre-filled it (llm_model) and the items the rater changed (overrides).

The dashboard never calls the LLM and never changes AI analyses.

Two ways to run it:
- On a laptop (`salud-phase1 dashboard --rater avni`): 127.0.0.1 only, the rater is fixed, no login.
- As the team website on Vercel (api/index.py): no fixed rater. Every data request needs a session cookie,
  which /api/login sets after the team passcode (TEAM_PASSCODE) and a rater name from the assignments.
  The cookie is signed with a key derived from the passcode, so changing the passcode signs everyone out.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..paths import REPO_ROOT
from .assign import rater_name
from .checklist import ITEM_DEFINITIONS, OUTCOMES, SENTIMENTS, TARGETS, TRANSCRIPT_ITEMS
from .prompt import split_turns
from .store import Store

BRANDING_DIR = REPO_ROOT / "branding"
LOGO_NAMES = ["logo.svg", "logo.png", "logo.webp", "logo.jpg", "logo.jpeg"]
SOURCE_LABELS = {"real": "Real", "synthetic": "Synthetic"}
SESSION_COOKIE = "qa_session"
SESSION_DAYS = 30


def _logo_path() -> Path | None:
    return next((BRANDING_DIR / n for n in LOGO_NAMES if (BRANDING_DIR / n).exists()), None)


def _session_key(passcode: str) -> bytes:
    return hashlib.sha256(f"agent-call-analysis-session|{passcode}".encode()).digest()


def make_session(passcode: str, rater: str, now: float | None = None) -> str:
    expires = int((time.time() if now is None else now) + SESSION_DAYS * 86400)
    body = f"{rater}|{expires}"
    sig = hmac.new(_session_key(passcode), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}|{sig}"


def read_session(passcode: str, value: str) -> str | None:
    """The rater in a valid, unexpired session cookie, else None."""
    try:
        rater, expires, sig = value.rsplit("|", 2)
        good = hmac.new(_session_key(passcode), f"{rater}|{expires}".encode(), hashlib.sha256).hexdigest()
        return rater if hmac.compare_digest(sig, good) and int(expires) > time.time() else None
    except ValueError:
        return None


def make_handler(store: Store, rater: str | None = None, passcode: str | None = None):
    """rater set: a laptop dashboard for that rater. rater None: the website, signed in with passcode."""
    if rater is None and not passcode:
        raise ValueError("the website needs a team passcode")
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

    def state(rater: str) -> dict:
        calls = store.calls()
        assigned = store.assignments()
        tags = store.tags(rater)
        labels = {r["call_id"]: {k: (r[k] or "") for k in TARGETS + ["notes"]} for r in tags.to_dict("records")}
        sources = list(dict.fromkeys(c["source"] for c in calls))
        ai = store.ai_info() | {"file": "the team database"}
        return {"config": {"rater": rater, "raters": sorted({r for rs in assigned.values() for r in rs}), "items": TRANSCRIPT_ITEMS, "definitions": ITEM_DEFINITIONS,
                           "sentiments": SENTIMENTS, "outcomes": OUTCOMES, "ai": ai,
                           "sources": [{"key": s, "label": SOURCE_LABELS.get(s, s)} for s in sources],
                           "has_logo": _logo_path() is not None, "website": passcode is not None},
                "calls": [{"id": c["call_id"], "file_id": c["file_id"], "source": c["source"],
                           "assignees": assigned.get(c["call_id"], [])} for c in calls],
                "labels": labels}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200, cookie: str | None = None):
            body = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if cookie is not None:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Robots-Tag", "noindex, nofollow")
            super().end_headers()

        def _rater(self) -> str | None:
            """Who is asking: the fixed laptop rater, or the rater in a valid website session."""
            if rater is not None:
                return rater
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            return read_session(passcode, jar[SESSION_COOKIE].value) if SESSION_COOKIE in jar else None

        def _cookie(self, value: str, max_age: int) -> str:
            return (f"{SESSION_COOKIE}={value}; Max-Age={max_age}; Path=/; HttpOnly; Secure; SameSite=Strict")

        def _login(self, body: dict):
            if not hmac.compare_digest(str(body.get("passcode", "")).encode(), passcode.encode()):
                time.sleep(1)  # slow down guessing
                return self._json({"error": "Wrong passcode"}, 401)
            raters = sorted({r for rs in store.assignments().values() for r in rs})
            name = rater_name(str(body.get("rater", "")))
            if not name:
                return self._json({"raters": raters})
            if name not in raters:
                return self._json({"error": "Pick your name from the list"}, 400)
            self._json({"ok": True}, cookie=self._cookie(make_session(passcode, name), SESSION_DAYS * 86400))

        def do_GET(self):
            url = urlparse(self.path)
            cid = parse_qs(url.query).get("id", [""])[0]
            me = self._rater()
            if url.path.startswith("/api/") and me is None:
                return self._json({"error": "sign in", "login": True}, 401)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, page.encode(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(state(me))
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
            path = urlparse(self.path).path
            if path not in ("/api/label", "/api/login", "/api/logout"):
                return self._json({"error": "not found"}, 404)
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json({"error": "expected JSON"}, 415)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if path == "/api/login":
                    if passcode is None:
                        return self._json({"error": "not found"}, 404)
                    return self._login(body)
                if path == "/api/logout":
                    return self._json({"ok": True}, cookie=self._cookie("", 0))
                me = self._rater()
                if me is None:
                    return self._json({"error": "sign in", "login": True}, 401)
                store.save_tag(me, body["id"], body["values"], body.get("llm"))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, 400)
            except RuntimeError as e:
                return self._json({"error": str(e)}, 503)
            self._json({"ok": True})

        def log_message(self, *args):
            pass

    return Handler


def serve(rater: str, port: int = 8765, store: Store | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(store or Store(), rater))
