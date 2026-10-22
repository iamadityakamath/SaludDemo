"""Vercel entry point: the Phase 1 tagging dashboard as the team website (Agent Call Analysis).

Vercel routes every path here (vercel.json). Settings are Vercel environment variables:
  DATABASE_URL   the team's Neon connection string (pooled)
  TEAM_PASSCODE  shared team passcode; at least 12 characters. Changing it signs everyone out.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from salud_pipeline.phase1.dashboard import make_handler  # noqa: E402
from salud_pipeline.phase1.store import Store  # noqa: E402

_passcode = os.environ.get("TEAM_PASSCODE", "")
if len(_passcode) < 12:
    raise RuntimeError("Set TEAM_PASSCODE (at least 12 characters) in the Vercel project settings")

_Dashboard = make_handler(Store(max_size=2), passcode=_passcode)


class handler(_Dashboard):  # noqa: N801  Vercel looks for a top-level class named `handler`
    pass
