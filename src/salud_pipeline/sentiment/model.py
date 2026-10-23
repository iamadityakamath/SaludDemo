"""cardiffnlp/twitter-roberta-base-sentiment-latest, run locally.

The model gives P(negative), P(neutral), P(positive) per text (softmax, they sum to 1). We report
    score     = P(positive) - P(negative)   in [-1, 1]
    magnitude = 1 - P(neutral)              in [0, 1]
    label     = the most likely class
The revision is pinned so a rerun gives the same numbers. The first run downloads ~500 MB to the
Hugging Face cache (~/.cache/huggingface); after that it runs offline.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"   # main as of 2025-08-04
MAX_TOKENS = 512


def from_probs(neg: float, neu: float, pos: float) -> dict:
    probs = {"negative": neg, "neutral": neu, "positive": pos}
    return {"score": pos - neg, "magnitude": 1 - neu, "label": max(probs, key=probs.get),
            "p_negative": neg, "p_neutral": neu, "p_positive": pos}


@dataclass
class SentimentModel:
    model_id: str = MODEL_ID
    revision: str = REVISION
    batch_size: int = 32
    device: str | None = None   # None: Apple GPU (mps) or CUDA when available, else CPU

    def __post_init__(self):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from transformers.utils import logging as hf_logging
        except ImportError:
            raise SystemExit("Sentiment needs the optional install: pip install -e '.[sentiment]'") from None
        hf_logging.set_verbosity_error()   # the unused pooler weights trigger a harmless load report
        self._torch = torch
        if self.device is None:
            self.device = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=self.revision)
        self.model.to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self._idx = [labels["negative"], labels["neutral"], labels["positive"]]

    def score(self, texts: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                   max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(**batch).logits.float(), dim=-1).cpu().tolist()
            out += [from_probs(*(p[j] for j in self._idx)) for p in probs]
        return out
