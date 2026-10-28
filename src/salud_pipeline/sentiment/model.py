"""cardiffnlp/twitter-roberta-base-sentiment-latest, as an int8 ONNX model, run locally.

The same model runs on a laptop (salud-sentiment run) and on the team website (live Analyze), so a call's
scores don't depend on where it was scored. The int8 ONNX copy (Xenova's conversion, 126 MB) needs only
onnxruntime + tokenizers (~80 MB), small enough for a Vercel function; PyTorch would not fit.

The model gives P(negative), P(neutral), P(positive) per text (softmax, they sum to 1). We report
    score     = P(positive) - P(negative)   in [-1, 1]
    magnitude = 1 - P(neutral)              in [0, 1]
    label     = the most likely class
The files are pinned to one revision and downloaded once to MODEL_CACHE (on Vercel: /tmp, per instance).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ONNX_REPO = "Xenova/twitter-roberta-base-sentiment-latest"
REVISION = "f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445"   # main as of 2026-10
FILES = ["config.json", "tokenizer.json", "onnx/model_quantized.onnx"]
MODEL_NAME = f"{MODEL_ID} (int8 onnx)"
MAX_TOKENS = 512
PAD_ID = 1   # RoBERTa <pad>


def _cache_dir() -> Path:
    if os.environ.get("SALUD_MODEL_CACHE"):
        return Path(os.environ["SALUD_MODEL_CACHE"])
    if os.environ.get("VERCEL"):
        return Path(tempfile.gettempdir()) / "salud-sentiment"
    return Path.home() / ".cache" / "salud-sentiment"


def download(folder: Path | None = None) -> Path:
    """The pinned model files, downloaded on first use. Returns their folder."""
    folder = (folder or _cache_dir()) / REVISION[:12]
    for name in FILES:
        dest = folder / name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{ONNX_REPO}/resolve/{REVISION}/{name}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(dest)   # only complete files get the real name
    return folder


def from_probs(neg: float, neu: float, pos: float) -> dict:
    probs = {"negative": neg, "neutral": neu, "positive": pos}
    return {"score": pos - neg, "magnitude": 1 - neu, "label": max(probs, key=probs.get),
            "p_negative": neg, "p_neutral": neu, "p_positive": pos}


@dataclass
class SentimentModel:
    batch_size: int = 16
    device: str = "cpu"

    def __post_init__(self):
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            raise SystemExit("Sentiment needs the optional install: pip install -e '.[sentiment]'") from None
        self._np = np
        folder = download()
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=PAD_ID, pad_token="<pad>")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(folder / "onnx/model_quantized.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.session.get_inputs()}
        labels = {v.lower(): int(k) for k, v in json.loads((folder / "config.json").read_text())["id2label"].items()}
        self._idx = [labels["negative"], labels["neutral"], labels["positive"]]

    def score(self, texts: list[str]) -> list[dict]:
        np, out = self._np, []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer.encode_batch(texts[i:i + self.batch_size])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            logits = self.session.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            out += [from_probs(*(float(p[j]) for j in self._idx)) for p in probs]
        return out


_shared: SentimentModel | None = None
_shared_lock = threading.Lock()


def shared_model() -> SentimentModel:
    """One model per process (the website loads it on the first Analyze, then reuses it)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = SentimentModel()
        return _shared
