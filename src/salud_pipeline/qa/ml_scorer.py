"""Traditional ML QA scorer: TF-IDF n-grams + one logistic regression per target.

Trained on synthetic calls only (ten real calls cannot train anything). Agent turns and caller
turns are vectorized separately, because most checklist items are about what the agent says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .targets import TARGETS


def _speaker_text(texts, agent: bool):
    out = []
    for t in texts:
        lines = [ln.split(":", 1)[1] for ln in t.splitlines() if ":" in ln
                 and ln.startswith("Agent") == agent]
        out.append(" ".join(lines))
    return out


def _vectorizer() -> FeatureUnion:
    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    return FeatureUnion([
        ("agent", Pipeline([("pick", FunctionTransformer(_speaker_text, kw_args={"agent": True})), ("tfidf", tfidf())])),
        ("caller", Pipeline([("pick", FunctionTransformer(_speaker_text, kw_args={"agent": False})), ("tfidf", tfidf())])),
    ])


class MLScorer:
    def __init__(self, C: float = 4.0):
        self.C = C
        self.vec = _vectorizer()
        self.models: dict[str, LogisticRegression | str] = {}

    def fit(self, df: pd.DataFrame) -> "MLScorer":
        X = self.vec.fit_transform(df["text"])
        for t in TARGETS:
            y = df[t].to_numpy()
            if len(set(y)) < 2:  # degenerate small sample: predict the only class seen
                self.models[t] = y[0]
                continue
            self.models[t] = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced").fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.vec.transform(df["text"])
        out = {"contact_id": df["contact_id"].to_numpy()}
        for t, m in self.models.items():
            out[t] = np.full(len(df), m, dtype=object) if isinstance(m, str) else m.predict(X)
        return pd.DataFrame(out)
