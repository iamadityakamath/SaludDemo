"""Every project location in one place.

Client-provided folders keep the names Salud sent them under. Everything the project generates lives
under data/phase1, data/phase2 and reports/phase1, all gitignored.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
CONFIGS = REPO_ROOT / "configs"

# Client-provided (as received)
REAL_TRANSCRIPTS = DATA / "Redacted Transcripts with Inferred Speakers"
SCHEMA_HANDOFF = DATA / "BIG_Schema_Only_Handoff"

# Phase 1: call analysis
PHASE1_DATA = DATA / "phase1"
PHASE1_SYNTHETIC_CALLS = PHASE1_DATA / "synthetic_calls"       # LLM-written calls (+ their synthetic world)
MANUAL_TAGS = PHASE1_DATA / "manual_tags"
REAL_TAGS = MANUAL_TAGS / "real_calls.csv"
SYNTHETIC_TAGS = MANUAL_TAGS / "synthetic_calls.csv"
PHASE1_CALL_LIST = CONFIGS / "phase1_synthetic_calls.txt"     # the 50 synthetic calls to tag (tracked)
PHASE1_REPORTS = REPO_ROOT / "reports" / "phase1"
LLM_CACHE = PHASE1_REPORTS / "llm_cache"                       # one file per analyzed transcript
AI_ANALYSES = PHASE1_REPORTS / "ai_analyses.json"              # script output: every call's AI analysis
COMPARISON = PHASE1_REPORTS / "comparison"                     # script output: AI vs manual tags

# Phase 2: propensity
PHASE2_DATA = DATA / "phase2"
PHASE2_SYNTHETIC_WORLD = PHASE2_DATA / "synthetic_world"       # synthetic accounts, payments, contacts
