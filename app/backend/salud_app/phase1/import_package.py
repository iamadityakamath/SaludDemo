"""Add Salud's final-package calls to Agent Call Reviews (source 'salud').

    cd app/backend
    .venv/bin/python -m salud_app.phase1.import_package --dry-run   # what would be added, nothing written
    .venv/bin/python -m salud_app.phase1.import_package             # add the calls and their results

For every transcript in `final.transcripts` that isn't already a Phase 1 call (8 of the 84 are the earlier real calls):
- adds it to phase1.calls as `salud/<file name>`, with the call's date from Salud's call record as `added_at`;
- scores its sentiment with the local Hugging Face model (free, nothing leaves the machine);
- stores its AI review **from the local Groq cache only** (the Stage 1 QA run reviewed all 84 with the same prompt and
  model). This command never calls Groq; a call without a cached review is reported, and the app reviews it when it is
  opened (or the background sweep does).
Safe to run again: existing calls and results are kept.

Needs migration 010 (source 'salud') applied to the database first.
"""

from __future__ import annotations

import argparse
import json
from zoneinfo import ZoneInfo

from .. import db
from ..config import get_settings
from ..sentiment.model import MODEL_NAME, shared_model
from ..sentiment.scoring import score_call
from . import store
from .llm import LLMEndpoint
from .prompt import PROMPT_VERSION, CallAssistant

SOURCE = "salud"
SALUD_TZ = ZoneInfo("America/Chicago")   # Salud's timestamps are naive local times

Q_PACKAGE = '''
SELECT t.file_name, t.transcript, ch."salud_DateandTimeofContact" AS contacted_at
FROM final.transcripts t
LEFT JOIN final.contact_history ch ON ch."salud_TCNRecordID" = t."salud_TCNRecordID"
WHERE t.phase1_call_id IS NULL AND t.transcript IS NOT NULL
ORDER BY ch."salud_DateandTimeofContact", t.file_name
'''


def source_allowed() -> bool:
    row = db.fetch_all("SELECT pg_get_constraintdef(oid) AS d FROM pg_constraint "
                       "WHERE conrelid = 'phase1.calls'::regclass AND conname = 'calls_source_check'")
    return bool(row) and f"'{SOURCE}'" in row[0]["d"]


def run(dry_run: bool = False, log=print) -> dict:
    if not source_allowed():
        raise SystemExit("phase1.calls doesn't accept source 'salud' yet: apply db/migrations/010_salud_package_calls.sql first")
    rows = db.fetch_all(Q_PACKAGE)
    existing = {r["call_id"] for r in db.fetch_all("SELECT call_id FROM phase1.calls WHERE source = %s", (SOURCE,))}
    assistant = CallAssistant(LLMEndpoint(), get_settings().llm_cache)
    stats = {"package_calls": len(rows), "already_added": 0, "added": 0, "sentiment_scored": 0,
             "reviews_from_cache": 0, "reviews_missing": 0}
    model = None if dry_run else shared_model()
    for r in rows:
        cid = f"{SOURCE}/{r['file_name']}"
        cached = assistant._path(r["transcript"])
        if dry_run:
            stats["already_added" if cid in existing else "added"] += 1
            stats["reviews_from_cache" if cached.exists() else "reviews_missing"] += 1
            continue
        if cid in existing:
            stats["already_added"] += 1
        else:
            added = r["contacted_at"].replace(tzinfo=SALUD_TZ) if r["contacted_at"] else None
            with db.pool().connection() as c:
                c.execute("INSERT INTO phase1.calls (call_id, source, file_id, transcript, added_by, added_at) "
                          "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (call_id) DO NOTHING",
                          (cid, SOURCE, r["file_name"], r["transcript"], "Salud package", added))
            stats["added"] += 1
        if store.sentiment(cid) is None:
            rows_, summary = score_call(r["transcript"], model)
            store.save_sentiment(cid, rows_, summary, MODEL_NAME)
            stats["sentiment_scored"] += 1
        if store.ai_analysis(cid) is None:
            if cached.exists():
                review = json.loads(cached.read_text())
                store.add_ai_analysis(cid, review, review.get("model") or assistant.endpoint.model, PROMPT_VERSION)
                stats["reviews_from_cache"] += 1
            else:
                stats["reviews_missing"] += 1
    log(("Dry run, nothing written: " if dry_run else "") + json.dumps(stats))
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report what would be added; write nothing")
    run(ap.parse_args(argv).dry_run)


if __name__ == "__main__":
    main()
