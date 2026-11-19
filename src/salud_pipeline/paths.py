"""Every project location in one place.

Client-provided folders keep the names Salud sent them under and stay in the shared root data/ (both phases read
them). Everything a phase generates lives in its own folder: "Phase 1/data", "Phase 1/reports", "Phase 2/data",
"Phase 2/reports", all gitignored. Code stays shared at the root (src/, api/, tests/).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
CONFIGS = REPO_ROOT / "configs"

# Client-provided (as received)
REAL_TRANSCRIPTS = DATA / "Redacted Transcripts with Inferred Speakers"
SCHEMA_HANDOFF = DATA / "BIG_Schema_Only_Handoff"
FINAL_PACKAGE = DATA / "BIG_Final_Package_84"                  # real CRM extract + transcripts, loaded into schema 'final'

# Phase 1: call analysis
PHASE1_ROOT = REPO_ROOT / "Phase 1"
PHASE1_DATA = PHASE1_ROOT / "data"
PHASE1_SYNTHETIC_CALLS = PHASE1_DATA / "synthetic_calls"       # LLM-written calls (+ their synthetic world)
# Local files imported once into the team database by `salud-db setup` (Neon is the live source after that).
REVIEWED_TAGS = PHASE1_DATA / "reviewed_tags"
REAL_TAGS = REVIEWED_TAGS / "real_calls.csv"
SYNTHETIC_TAGS = REVIEWED_TAGS / "synthetic_calls.csv"
PHASE1_CALL_LIST = CONFIGS / "phase1_synthetic_calls.txt"     # the 50 synthetic calls to tag (tracked)
PHASE1_REPORTS = PHASE1_ROOT / "reports"
LLM_CACHE = PHASE1_REPORTS / "llm_cache"                       # local cache: one file per analyzed transcript
AI_ANALYSES = PHASE1_REPORTS / "ai_analyses.json"              # pre-Neon analyses, imported by salud-db setup
COMPARISON = PHASE1_REPORTS / "comparison"                     # compare output (local Excel/CSV)
AI_VERSIONS = PHASE1_REPORTS / "ai_versions"                   # trial runs: <prompt version>.json + comparison

# Phase 2: propensity
PHASE2_ROOT = REPO_ROOT / "Phase 2"
PHASE2_DATA = PHASE2_ROOT / "data"
PHASE2_SYNTHETIC_WORLD = PHASE2_DATA / "synthetic_world"       # synthetic accounts, payments, contacts (original)
# Synthetic worlds for the propensity work. "salud" is tuned to Salud's final package (configs/synthetic_world_salud.json,
# checked with salud-synth-calibrate); "original" is the first world, kept for comparison.
# "master" (configs/synthetic_world_master.json) covers Salud's time frame up to 2026-11-18; it feeds the master file
# "Phase 2/reports/pre_training_data.xlsx" (salud-propensity master).
PHASE2_WORLDS = {"original": PHASE2_SYNTHETIC_WORLD, "salud": PHASE2_DATA / "synthetic_world_salud",
                 "master": PHASE2_DATA / "synthetic_world_master"}
DEFAULT_WORLD = "salud"
PHASE2_REPORTS = PHASE2_ROOT / "reports"                       # master file, experiments, scores
