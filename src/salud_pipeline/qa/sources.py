"""The calls Phase 1 works with, and where their human labels and AI analyses live.

- Real: the 10 redacted Salud calls.
- Synthetic: LLM-written calls (Qwen) from data/synthetic/run_llm; realistic wording, planted labels known.

The 4,152 template transcripts in run_default are not used in Phase 1. Their synthetic accounts, payments
and contact history are for Phase 2.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import REPO_ROOT
from .tag_server import CallSource

REAL_DIR = REPO_ROOT / "data" / "Redacted Transcripts with Inferred Speakers"
SYNTHETIC_RUN = REPO_ROOT / "data" / "synthetic" / "run_llm"
GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth"
REAL_LABELS = GROUND_TRUTH / "real_calls_labels.csv"
SYNTHETIC_LABELS = GROUND_TRUTH / "synthetic_calls_labels.csv"
ASSIST_CACHE = REPO_ROOT / "reports" / "qa" / "assist_cache"   # AI analyses, shared by page and reports


def phase1_sources(synthetic_run: Path = SYNTHETIC_RUN) -> list[CallSource]:
    return [
        CallSource("real", "Real", REAL_DIR, REAL_LABELS, real=True),
        CallSource("synthetic", "Synthetic", synthetic_run / "transcripts", SYNTHETIC_LABELS, real=False),
    ]
