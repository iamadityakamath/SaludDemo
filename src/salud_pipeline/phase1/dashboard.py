"""Phase 1 route 1: the dashboard for reviewing and tagging calls.

    salud-phase1 dashboard --rater AK

Lists the 60 Phase 1 calls (calls.py) with filters and serves on 127.0.0.1 only. Each call opens pre-filled
with the AI analysis that `salud-phase1 analyze` saved in reports/phase1/ai_analyses.json: checklist answers,
the transcript quotes behind them, sentiment, outcome and coaching tips. The rater checks and corrects the
answers and saves. The dashboard never calls the LLM and never writes to ai_analyses.json.

Saved tags go to data/phase1/reviewed_tags/ (one row per call per rater). Each row records the model whose
answers pre-filled it (llm_model) and which items the rater changed (overrides).
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

from ..paths import AI_ANALYSES, REPO_ROOT
from .calls import CallSource
from .checklist import (ITEM_DEFINITIONS, LABEL_COLUMNS, OUTCOMES, SENTIMENTS, TARGETS, TRANSCRIPT_ITEMS,
                        clean_transcript)
from .prompt import split_turns

BRANDING_DIR = REPO_ROOT / "branding"
LOGO_NAMES = ["logo.svg", "logo.png", "logo.webp", "logo.jpg", "logo.jpeg"]


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


class AIAnalyses:
    """Read-only view of the script's AI analyses (ai_analyses.json). The dashboard never writes to it.

    The file is re-read when it changes on disk, so calls added later by `salud-phase1 analyze` show up
    after a page reload without restarting the dashboard.
    """

    def __init__(self, path: Path):
        self.path = path
        self._mtime: float | None = None
        self._data: dict = {}
        self._lock = threading.Lock()

    def _load(self) -> dict:
        with self._lock:
            mtime = self.path.stat().st_mtime if self.path.exists() else None
            if mtime != self._mtime:
                self._data = json.loads(self.path.read_text()) if mtime else {}
                self._mtime = mtime
            return self._data

    def get(self, call_id: str) -> dict | None:
        entry = self._load().get("calls", {}).get(call_id)
        return entry["analysis"] if entry else None

    def info(self) -> dict:
        d = self._load()
        return {"model": d.get("model"), "prompt_version": d.get("prompt_version"),
                "analyzed": len(d.get("calls", {})), "file": self.path.name}


def _logo_path() -> Path | None:
    return next((BRANDING_DIR / n for n in LOGO_NAMES if (BRANDING_DIR / n).exists()), None)


def make_handler(store: LabelStore, rater: str, ai: AIAnalyses):
    page = files("salud_pipeline.phase1").joinpath("static/dashboard.html").read_text()
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
                self._json({"config": config | {"ai": ai.info(), "has_logo": _logo_path() is not None},
                            "calls": store.calls(), "labels": store.labels_for(rater)})
            elif url.path == "/api/call":
                cid = q.get("id", [""])[0]
                if store.call(cid) is None:
                    return self._json({"error": "unknown call"}, 404)
                self._json(store.transcript(cid))
            elif url.path == "/api/analysis":
                # The script's saved analysis for this call (read-only); the dashboard never calls the LLM.
                cid = q.get("id", [""])[0]
                if store.call(cid) is None:
                    return self._json({"error": "unknown call"}, 404)
                self._json(ai.get(cid) or {"none": True})
            elif url.path == "/brand/logo":
                p = _logo_path()
                if p is None:
                    return self._json({"error": "no logo"}, 404)
                self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
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


def serve(sources: list[CallSource], rater: str, port: int = 8765,
          ai_file: Path = AI_ANALYSES) -> ThreadingHTTPServer:
    store = LabelStore([s for s in sources if s.transcripts_dir.exists()])
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, rater, AIAnalyses(ai_file)))
