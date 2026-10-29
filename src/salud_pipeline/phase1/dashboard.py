"""The agent dashboard, backed by the team's Neon database: pick a call or paste a new one and see its sentiment,
QA checklist results, summary, outcome and coaching tips (static/analyze.html).

    salud-phase1 dashboard --rater avni

Uploading a call (pasted or .txt files, saved as source 'new') starts its analysis at once: sentiment scored in
this process, and the LLM review when an API key is set, both saved to Neon. The list shows it as processing
(live.call_status) and refreshes until it's done. Opening a call shows what is stored (/api/result) and fills in
anything still missing (/api/analyze/run). Re-analyze redoes both and replaces the stored results. The manual tagging page is gone;
tags saved during Phase 1 stay in phase1.reviewed_tags for `salud-phase1 compare`.

Two ways to run it:
- On a laptop (`salud-phase1 dashboard --rater avni`): 127.0.0.1 only, the user is fixed, no login. Uploads
  are analyzed in a server thread, and start_sweeper checks every 15 seconds for transcripts added any other way.
- As the team website on Vercel (api/index.py). Every data request needs a session cookie, which /api/login
  sets after the team passcode (TEAM_PASSCODE) and a name from the assignments. The cookie is signed with a
  key derived from the passcode, so changing the passcode signs everyone out. Nothing runs after a response
  there, so the page starts an upload's analysis itself. A daily Vercel cron calls /api/cron/process
  (vercel.json) with the CRON_SECRET bearer token for transcripts added any other way.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import os
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..paths import REPO_ROOT
from .assign import rater_name
from . import live
from .checklist import ITEM_DEFINITIONS, TRANSCRIPT_ITEMS
from .prompt import split_turns
from .store import Store

BRANDING_DIR = REPO_ROOT / "branding"
LOGO_NAMES = ["logo.svg", "logo.png", "logo.webp", "logo.jpg", "logo.jpeg"]
SESSION_COOKIE = "qa_session"
SESSION_DAYS = 30
CRON_SECONDS = 45   # stay inside the website's 60 s function limit (vercel.json maxDuration)


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


def analyze_in_background(store: Store):
    """on_upload for a long-running server (the laptop): analyze the uploaded calls in a thread, one at a time."""
    def start(ids: list[str]):
        def run():
            for cid in ids:
                try:
                    live.analyze(store, cid)
                except (RuntimeError, ValueError):   # the sweep or the page retries it
                    pass
        threading.Thread(target=run, name="salud-upload", daemon=True).start()
    return start


def make_handler(store: Store, rater: str | None = None, passcode: str | None = None, on_upload=None):
    """rater set: a laptop dashboard for that rater. rater None: the website, signed in with passcode.
    on_upload(ids) starts the analysis of uploaded calls on the server. On the website, where nothing may run
    after the response, it is None and the page starts it (/api/analyze/run) right after the upload."""
    if rater is None and not passcode:
        raise ValueError("the website needs a team passcode")
    page = files("salud_pipeline.phase1").joinpath("static/analyze.html").read_text()
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

        def _cron(self):
            """Vercel cron: analyze stored transcripts that are missing results. Needs the CRON_SECRET bearer token."""
            secret = os.environ.get("CRON_SECRET", "")
            given = self.headers.get("Authorization", "").encode()
            if len(secret) < 16 or not hmac.compare_digest(given, f"Bearer {secret}".encode()):
                return self._json({"error": "not found"}, 404)
            self._json(live.process_pending(store, seconds=CRON_SECONDS))

        def do_GET(self):
            url = urlparse(self.path)
            cid = parse_qs(url.query).get("id", [""])[0]
            if url.path == "/api/cron/process":
                try:
                    return self._cron()
                except RuntimeError as e:
                    return self._json({"error": str(e)}, 503)
            me = self._rater()
            if url.path.startswith("/api/") and me is None:
                return self._json({"error": "sign in", "login": True}, 401)
            try:
                if url.path in ("/", "/index.html", "/analyze"):
                    self._send(200, page.encode(), "text/html; charset=utf-8")
                elif url.path == "/api/analyze/list":
                    llm = live.llm_ready()
                    calls = [c | {"status": live.call_status(c, llm)} for c in store.analysis_list()]
                    self._json({"me": me, "website": passcode is not None, "llm": llm,
                                "items": TRANSCRIPT_ITEMS, "definitions": ITEM_DEFINITIONS, "calls": calls})
                elif url.path == "/api/call":
                    t = transcript(cid)
                    self._json(t) if t else self._json({"error": "unknown call"}, 404)
                elif url.path == "/api/result":
                    self._json(live.results(store, cid))
                elif url.path == "/brand/logo":
                    p = _logo_path()
                    if p is None:
                        return self._json({"error": "no logo"}, 404)
                    self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as e:    # unknown call
                self._json({"error": str(e)}, 404)
            except RuntimeError as e:  # e.g. database unreachable
                self._json({"error": str(e)}, 503)

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/api/login", "/api/logout", "/api/analyze/run", "/api/analyze/new"):
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
                if path == "/api/analyze/new":   # upload: save now, analyze in the background (see on_upload)
                    texts = body["transcripts"] if "transcripts" in body else [body.get("transcript", "")]
                    if not isinstance(texts, list) or not 1 <= len(texts) <= live.MAX_UPLOAD_CALLS:
                        raise ValueError(f"upload 1 to {live.MAX_UPLOAD_CALLS} transcripts at a time")
                    texts = [str(t) for t in texts]
                    if any(len(t) > live.MAX_TRANSCRIPT_CHARS for t in texts):
                        raise ValueError(f"a transcript is too long (max {live.MAX_TRANSCRIPT_CHARS:,} characters)")
                    if any(not t.strip() for t in texts):
                        raise ValueError("a transcript is empty")
                    ids = [store.add_new_call(t, me) for t in texts]
                    if on_upload:
                        on_upload(ids)
                    return self._json({"ids": ids})
                return self._json(live.analyze(store, str(body["id"]), redo=bool(body.get("redo"))))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, 400)
            except RuntimeError as e:
                return self._json({"error": str(e)}, 503)

        def log_message(self, *args):
            pass

    return Handler


def serve(rater: str, port: int = 8765, store: Store | None = None, auto: bool = False) -> ThreadingHTTPServer:
    """auto: uploads are analyzed by the server in the background (the page's own request then just waits)."""
    store = store or Store()
    return ThreadingHTTPServer(("127.0.0.1", port),
                               make_handler(store, rater, on_upload=analyze_in_background(store) if auto else None))


def start_sweeper(store: Store, every: float = 15, log=print) -> threading.Event:
    """Analyze new transcripts in the database now and then every `every` seconds, in a background thread.
    Set the returned event to stop it."""
    stop = threading.Event()

    def run():
        while not stop.is_set():
            try:
                r = live.process_pending(store, log=lambda cid, _: log(f"  analyzed {cid}"))
                if r["stopped"]:
                    log(f"  automatic analysis paused: {r['stopped'][:160]} ({r['left']} calls left)")
            except RuntimeError as e:   # database unreachable: try again next round
                log(f"  automatic analysis skipped: {str(e)[:160]}")
            stop.wait(every)

    threading.Thread(target=run, name="salud-sweeper", daemon=True).start()
    return stop
