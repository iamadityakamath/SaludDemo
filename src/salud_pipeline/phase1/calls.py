"""The 60 Phase 1 calls and where their manual tags are saved.

- Real: the 10 redacted Salud calls.
- Synthetic: 50 LLM-written calls from data/phase1/synthetic_calls, listed in configs/phase1_synthetic_calls.txt.
  The 50 were picked from 168 to cover every planted outcome and scenario, failures of every checklist item,
  and some all-pass calls.

Call IDs are '<source>/<file name>', e.g. 'real/0fd3661b-....txt'.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..paths import PHASE1_CALL_LIST, PHASE1_SYNTHETIC_CALLS, REAL_TAGS, REAL_TRANSCRIPTS, SYNTHETIC_TAGS


@dataclass
class CallSource:
    key: str               # used in call IDs: "real" or "synthetic"
    label: str             # shown on the dashboard
    transcripts_dir: Path
    labels_csv: Path       # where manual tags for these calls are saved
    real: bool             # real client data (the dashboard asks before sending it to a cloud LLM)
    include: frozenset[str] | None = None  # file stems to use; None = every transcript in the folder

    def files(self) -> list[Path]:
        paths = sorted(self.transcripts_dir.glob("*.txt")) if self.transcripts_dir.exists() else []
        return [p for p in paths if self.include is None or p.stem in self.include]


def read_call_list(path: Path = PHASE1_CALL_LIST) -> frozenset[str] | None:
    if not path.exists():
        return None
    return frozenset(ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#"))


def phase1_sources(synthetic_calls: Path = PHASE1_SYNTHETIC_CALLS) -> list[CallSource]:
    return [
        CallSource("real", "Real", REAL_TRANSCRIPTS, REAL_TAGS, real=True),
        CallSource("synthetic", "Synthetic", synthetic_calls / "transcripts", SYNTHETIC_TAGS, real=False,
                   include=read_call_list()),
    ]
