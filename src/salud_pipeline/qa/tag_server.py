"""Local tagging tool for building QA ground truth.

    salud-qa tag --rater AK            # real calls -> data/ground_truth/real_calls_labels.csv

Serves one page on 127.0.0.1 only, so client transcripts never leave the machine. Each rater
sees only their own labels (blind tagging keeps the inter-rater check honest). Saving replaces
that rater's row for the call in the CSV that `salud-qa ml/llm` already read.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

import pandas as pd

from .targets import ITEM_DEFINITIONS, OUTCOMES, SENTIMENTS, TARGETS, TRANSCRIPT_ITEMS, clean_transcript

COLUMNS = ["file_id", "rater"] + TARGETS + ["notes"]


class LabelStore:
    def __init__(self, csv_path: Path, transcripts_dir: Path):
        self.csv_path, self.transcripts_dir = csv_path, transcripts_dir
        self.lock = threading.Lock()

    def calls(self) -> list[dict]:
        return [{"file_id": p.name, "text": clean_transcript(p.read_text())}
                for p in sorted(self.transcripts_dir.glob("*.txt"))]

    def _read(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame(columns=COLUMNS)
        df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
        return df.reindex(columns=COLUMNS, fill_value="")

    def labels_for(self, rater: str) -> dict[str, dict]:
        with self.lock:
            df = self._read()
        mine = df[df["rater"] == rater]
        return {r["file_id"]: {c: r[c] for c in TARGETS + ["notes"]} for r in mine.to_dict("records")}

    def save(self, rater: str, file_id: str, values: dict) -> None:
        known = {c["file_id"] for c in self.calls()}
        if file_id not in known:
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
        with self.lock:
            df = self._read()
            # Drop this rater's previous row and the blank template row for this call.
            drop = (df["file_id"] == file_id) & ((df["rater"] == rater) | (df["rater"] == ""))
            df = pd.concat([df[~drop], pd.DataFrame([row])], ignore_index=True)
            df = df.sort_values(["file_id", "rater"], kind="stable")
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.csv_path.with_suffix(".tmp")
            df[COLUMNS].to_csv(tmp, index=False)
            tmp.replace(self.csv_path)


def make_handler(store: LabelStore, rater: str):
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
            if self.path in ("/", "/index.html"):
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json({"config": config, "calls": store.calls(), "labels": store.labels_for(rater)})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/label":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                store.save(rater, body["file_id"], body["values"])
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, 400)
            self._json({"ok": True})

        def log_message(self, *args):
            pass

    return Handler


def serve(csv_path: Path, transcripts_dir: Path, rater: str, port: int = 8765) -> ThreadingHTTPServer:
    store = LabelStore(csv_path, transcripts_dir)
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, rater))
