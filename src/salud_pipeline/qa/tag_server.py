"""Local tagging tool for building QA ground truth, with LLM pre-fill and coaching tips.

    salud-qa tag --rater AK                                   # real calls
    salud-qa tag --rater AK --run data/synthetic/run_default  # synthetic practice calls

Serves one page on 127.0.0.1 only. When an LLM is configured (.env / environment, see
llm_client.py), opening a call pre-fills the checklist from one LLM review that also cites the
transcript turns behind each answer and suggests agent coaching tips. The rater can change any
answer. Saved rows record which model pre-filled them and which items the rater overrode, so
evaluation can tell human-only labels from LLM-assisted ones.

Real client transcripts are sent to a non-local LLM (e.g. Groq) only with --include-real.
"""

from __future__ import annotations

import json
import mimetypes
import threading
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


class LabelStore:
    def __init__(self, csv_path: Path, transcripts_dir: Path):
        self.csv_path, self.transcripts_dir = csv_path, transcripts_dir
        self.lock = threading.Lock()
        self._calls = [{"file_id": p.name, "text": clean_transcript(p.read_text())}
                       for p in sorted(transcripts_dir.glob("*.txt"))]
        for c in self._calls:
            c["turns"] = split_turns(c["text"])
        self._by_id = {c["file_id"]: c for c in self._calls}

    def calls(self) -> list[dict]:
        return self._calls

    def call(self, file_id: str) -> dict | None:
        return self._by_id.get(file_id)

    def _read(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame(columns=LABEL_COLUMNS)
        df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
        return df.reindex(columns=LABEL_COLUMNS, fill_value="")

    def labels_for(self, rater: str) -> dict[str, dict]:
        with self.lock:
            df = self._read()
        mine = df[df["rater"] == rater]
        return {r["file_id"]: {c: r[c] for c in TARGETS + ["notes"]} for r in mine.to_dict("records")}

    def save(self, rater: str, file_id: str, values: dict, llm: dict | None = None) -> None:
        if file_id not in self._by_id:
            raise ValueError(f"unknown call {file_id}")
        row = {"file_id": file_id, "rater": rater}
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
        with self.lock:
            df = self._read()
            drop = (df["file_id"] == file_id) & ((df["rater"] == rater) | (df["rater"] == ""))
            df = pd.concat([df[~drop], pd.DataFrame([row])], ignore_index=True)
            df = df.sort_values(["file_id", "rater"], kind="stable")
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.csv_path.with_suffix(".tmp")
            df[LABEL_COLUMNS].to_csv(tmp, index=False)
            tmp.replace(self.csv_path)


class Suggester:
    """Wraps CallAssistant with the real-data guard and background prefetch."""

    def __init__(self, endpoint: LLMEndpoint | None, cache_dir: Path, real_data: bool, include_real: bool):
        self.disabled_reason = None
        if endpoint is None or not endpoint.model:
            self.disabled_reason = "No LLM configured. Add GROQ_API_KEY to .env and restart."
        elif real_data and not endpoint.is_local and not include_real:
            self.disabled_reason = (f"AI suggestions are off for real calls: {endpoint.provider} is outside the "
                                    "approved environment. Restart with --include-real once Salud approves it.")
        self.assistant = CallAssistant(endpoint, cache_dir) if not self.disabled_reason else None
        self.endpoint = endpoint
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    @property
    def info(self) -> dict:
        if self.disabled_reason:
            return {"enabled": False, "reason": self.disabled_reason}
        return {"enabled": True, "model": self.endpoint.model, "provider": self.endpoint.provider}

    def review(self, text: str) -> dict:
        if self.assistant is None:
            return {"disabled": self.disabled_reason}
        key = str(self.assistant._path(text))
        with self._lock:
            ev = self._inflight.get(key)
            owner = ev is None
            if owner:
                ev = self._inflight[key] = threading.Event()
        if not owner:  # another thread (e.g. prefetch) is already asking; reuse its cached result
            ev.wait(timeout=180)
            return self._safe(text)
        try:
            return self._safe(text)
        finally:
            ev.set()
            with self._lock:
                self._inflight.pop(key, None)

    def _safe(self, text: str) -> dict:
        try:
            return self.assistant.review(text)
        except (RuntimeError, ValueError) as e:
            return {"error": str(e)}

    def prefetch(self, text: str) -> None:
        if self.assistant is not None and not self.assistant._path(text).exists():
            threading.Thread(target=self.review, args=(text,), daemon=True).start()


def _logo_path() -> Path | None:
    return next((BRANDING_DIR / n for n in LOGO_NAMES if (BRANDING_DIR / n).exists()), None)


def make_handler(store: LabelStore, rater: str, suggester: Suggester):
    page = files("salud_pipeline.qa").joinpath("static/tagger.html").read_text()
    config = {"rater": rater, "items": TRANSCRIPT_ITEMS, "definitions": ITEM_DEFINITIONS,
              "sentiments": SENTIMENTS, "outcomes": OUTCOMES, "source": store.transcripts_dir.name}

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
            if url.path in ("/", "/index.html"):
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json({"config": config | {"llm": suggester.info, "has_logo": _logo_path() is not None},
                            "calls": store.calls(), "labels": store.labels_for(rater)})
            elif url.path == "/api/suggest":
                fid = parse_qs(url.query).get("file_id", [""])[0]
                call = store.call(fid)
                if call is None:
                    return self._json({"error": "unknown call"}, 404)
                result = suggester.review(call["text"])
                idx = store.calls().index(call)
                if idx + 1 < len(store.calls()):
                    suggester.prefetch(store.calls()[idx + 1]["text"])
                self._json(result)
            elif url.path == "/brand/logo":
                p = _logo_path()
                if p is None:
                    return self._json({"error": "no logo"}, 404)
                self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/label":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                store.save(rater, body["file_id"], body["values"], body.get("llm"))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, 400)
            self._json({"ok": True})

        def log_message(self, *args):
            pass

    return Handler


def serve(csv_path: Path, transcripts_dir: Path, rater: str, port: int = 8765,
          endpoint: LLMEndpoint | None = None, real_data: bool = True, include_real: bool = False,
          cache_dir: Path | None = None) -> ThreadingHTTPServer:
    store = LabelStore(csv_path, transcripts_dir)
    suggester = Suggester(endpoint, cache_dir or REPO_ROOT / "reports" / "qa" / "assist_cache",
                          real_data, include_real)
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, rater, suggester))
