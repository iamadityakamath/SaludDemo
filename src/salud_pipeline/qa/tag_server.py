"""Local call review page for building QA ground truth, with LLM pre-fill and coaching tips.

    salud-qa tag --rater AK

One page lists every call from every source (the real redacted calls and a synthetic run) with
filters for source, tagging status and ID. Serves on 127.0.0.1 only.

Each source keeps its own labels CSV, so synthetic practice tags never mix into the real ground
truth. When an LLM is configured (.env / environment, see llm_client.py), the Analyze button pre-fills
the checklist from one review that cites the transcript turns behind each answer and suggests
coaching tips. The LLM runs only when the rater presses Analyze; reopening an analyzed call
shows the cached result. Saved rows record which model pre-filled them and which items the rater
overrode.

Real and synthetic calls are analyzed the same way. Real calls are redacted, but Analyze still sends
them to the configured LLM (e.g. Groq); the page asks the rater to confirm once per browser session
before the first real call is sent to a non-local provider.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ..llm_client import LLMEndpoint
from ..schema import REPO_ROOT
from .assist import CallAssistant, split_turns
from .targets import (ITEM_DEFINITIONS, LABEL_COLUMNS, OUTCOMES, SENTIMENTS, TARGETS, TRANSCRIPT_ITEMS,
                      clean_transcript)

BRANDING_DIR = REPO_ROOT / "branding"
LOGO_NAMES = ["logo.svg", "logo.png", "logo.webp", "logo.jpg", "logo.jpeg"]


@dataclass
class CallSource:
    key: str               # short id used in call ids, e.g. "real" / "synthetic"
    label: str             # shown in the page
    transcripts_dir: Path
    labels_csv: Path
    real: bool             # real client data: subject to the HIPAA guard
    include: frozenset[str] | None = None  # contact IDs (file stems) to list; None = every transcript

    def files(self) -> list[Path]:
        paths = sorted(self.transcripts_dir.glob("*.txt")) if self.transcripts_dir.exists() else []
        return [p for p in paths if self.include is None or p.stem in self.include]


class LabelStore:
    """All calls across sources. Call id = '<source key>/<file name>'."""

    def __init__(self, sources: list[CallSource]):
        self.sources = {s.key: s for s in sources}
        self.locks = {s.key: threading.Lock() for s in sources}
        self._calls: list[dict] = []
        for s in sources:
            for p in s.files():
                self._calls.append({"id": f"{s.key}/{p.name}", "file_id": p.name, "source": s.key})
        self._by_id = {c["id"]: c for c in self._calls}
        self._text_cache: dict[str, dict] = {}

    def calls(self) -> list[dict]:
        return self._calls

    def call(self, call_id: str) -> dict | None:
        return self._by_id.get(call_id)

    def source_of(self, call_id: str) -> CallSource:
        return self.sources[self._by_id[call_id]["source"]]

    def transcript(self, call_id: str) -> dict:
        """Cleaned text and numbered turns, read on first request."""
        if call_id not in self._text_cache:
            c = self._by_id[call_id]
            text = clean_transcript((self.sources[c["source"]].transcripts_dir / c["file_id"]).read_text())
            self._text_cache[call_id] = {"text": text, "turns": split_turns(text)}
        return self._text_cache[call_id]

    @staticmethod
    def _read(csv: Path) -> pd.DataFrame:
        if not csv.exists():
            return pd.DataFrame(columns=LABEL_COLUMNS)
        df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        return df.reindex(columns=LABEL_COLUMNS, fill_value="")

    def labels_for(self, rater: str) -> dict[str, dict]:
        out = {}
        for key, src in self.sources.items():
            with self.locks[key]:
                df = self._read(src.labels_csv)
            for r in df[df["rater"] == rater].to_dict("records"):
                out[f"{key}/{r['file_id']}"] = {c: r[c] for c in TARGETS + ["notes"]}
        return out

    def save(self, rater: str, call_id: str, values: dict, llm: dict | None = None) -> None:
        if call_id not in self._by_id:
            raise ValueError(f"unknown call {call_id}")
        c, src = self._by_id[call_id], self.source_of(call_id)
        row = {"file_id": c["file_id"], "rater": rater}
        for t in TRANSCRIPT_ITEMS:
            v = values.get(t, "")
            if v not in ("Yes", "No", "N/A", ""):
                raise ValueError(f"{t}: {v!r}")
            row[t] = v
        for t, allowed in (("sentiment", SENTIMENTS), ("outcome", OUTCOMES)):
            v = values.get(t, "")
            if v and v not in allowed:
                raise ValueError(f"{t}: {v!r}")
            row[t] = v
        row["notes"] = str(values.get("notes", ""))[:2000]
        # Provenance: which model pre-filled the form and which targets the rater changed.
        suggested = (llm or {}).get("answers") or {}
        row["llm_model"] = str((llm or {}).get("model") or "")[:100] if suggested else ""
        row["overrides"] = ";".join(t for t in TARGETS if suggested.get(t) and row[t] and row[t] != suggested[t])
        with self.locks[src.key]:
            df = self._read(src.labels_csv)
            drop = (df["file_id"] == c["file_id"]) & ((df["rater"] == rater) | (df["rater"] == ""))
            df = pd.concat([df[~drop], pd.DataFrame([row])], ignore_index=True)
            df = df.sort_values(["file_id", "rater"], kind="stable")
            src.labels_csv.parent.mkdir(parents=True, exist_ok=True)
            tmp = src.labels_csv.with_suffix(".tmp")
            df[LABEL_COLUMNS].to_csv(tmp, index=False)
            tmp.replace(src.labels_csv)


class Suggester:
    """CallAssistant with single-flight requests. Never calls the LLM unless asked."""

    def __init__(self, endpoint: LLMEndpoint | None, cache_dir: Path):
        self.endpoint = endpoint
        self.assistant = CallAssistant(endpoint, cache_dir) if endpoint and endpoint.model else None
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def blocked_reason(self, real: bool) -> str | None:
        if self.assistant is None:
            return "No LLM configured. Add GROQ_API_KEY to .env and restart."
        return None

    def info(self) -> dict:
        base = {"model": self.endpoint.model if self.assistant else None,
                "provider": self.endpoint.provider if self.assistant else None,
                "remote": bool(self.assistant) and not self.endpoint.is_local}
        return base | {"real": self.blocked_reason(True), "synthetic": self.blocked_reason(False)}

    def cached(self, text: str, real: bool) -> dict | None:
        """Previous analysis from the local cache; never calls the LLM."""
        if self.blocked_reason(real):
            return None
        return self.assistant.cached(text)

    def review(self, text: str, real: bool, force: bool = False) -> dict:
        """Runs only on an explicit Analyze click. Concurrent clicks share one LLM call."""
        reason = self.blocked_reason(real)
        if reason:
            return {"disabled": reason}
        key = str(self.assistant._path(text))
        with self._lock:
            ev = self._inflight.get(key)
            owner = ev is None
            if owner:
                ev = self._inflight[key] = threading.Event()
        if not owner:
            ev.wait(timeout=180)
            return self._safe(text, force=False)
        try:
            return self._safe(text, force)
        finally:
            ev.set()
            with self._lock:
                self._inflight.pop(key, None)

    def _safe(self, text: str, force: bool) -> dict:
        try:
            return self.assistant.review(text, force=force)
        except (RuntimeError, ValueError) as e:
            return {"error": str(e)}


def _logo_path() -> Path | None:
    return next((BRANDING_DIR / n for n in LOGO_NAMES if (BRANDING_DIR / n).exists()), None)


def make_handler(store: LabelStore, rater: str, suggester: Suggester):
    page = files("salud_pipeline.qa").joinpath("static/tagger.html").read_text()
    config = {"rater": rater, "items": TRANSCRIPT_ITEMS, "definitions": ITEM_DEFINITIONS,
              "sentiments": SENTIMENTS, "outcomes": OUTCOMES,
              "sources": [{"key": s.key, "label": s.label, "real": s.real} for s in store.sources.values()]}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200):
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json({"config": config | {"llm": suggester.info(), "has_logo": _logo_path() is not None},
                            "calls": store.calls(), "labels": store.labels_for(rater)})
            elif url.path == "/api/call":
                cid = q.get("id", [""])[0]
                if store.call(cid) is None:
                    return self._json({"error": "unknown call"}, 404)
                self._json(store.transcript(cid))
            elif url.path == "/api/analysis":
                # Cached analysis only (shown when a call opens); never calls the LLM.
                cid = q.get("id", [""])[0]
                if store.call(cid) is None:
                    return self._json({"error": "unknown call"}, 404)
                cached = suggester.cached(store.transcript(cid)["text"], store.source_of(cid).real)
                self._json(cached or {"none": True})
            elif url.path == "/brand/logo":
                p = _logo_path()
                if p is None:
                    return self._json({"error": "no logo"}, 404)
                self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/analyze":
                # The only route that calls the LLM: the page's Analyze button.
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    cid = body["id"]
                except (KeyError, json.JSONDecodeError) as e:
                    return self._json({"error": str(e)}, 400)
                if store.call(cid) is None:
                    return self._json({"error": "unknown call"}, 404)
                return self._json(suggester.review(store.transcript(cid)["text"], store.source_of(cid).real,
                                                   force=bool(body.get("force"))))
            if path != "/api/label":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                store.save(rater, body["id"], body["values"], body.get("llm"))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, 400)
            self._json({"ok": True})

        def log_message(self, *args):
            pass

    return Handler


def serve(sources: list[CallSource], rater: str, port: int = 8765, endpoint: LLMEndpoint | None = None,
          cache_dir: Path | None = None) -> ThreadingHTTPServer:
    store = LabelStore([s for s in sources if s.transcripts_dir.exists()])
    suggester = Suggester(endpoint, cache_dir or REPO_ROOT / "reports" / "qa" / "assist_cache")
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, rater, suggester))
