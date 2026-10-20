"""The team's shared Neon Postgres database: schema as code and the one-time import.

    salud-db setup     # team lead: apply db/migrations, load the synthetic world, import local Phase 1 files
    salud-db status    # row counts per table

`setup` is safe to run again at any time. It only adds what is missing:
- applies migrations not yet recorded in public.schema_migrations
- loads big_synthetic (Salud's CRM tables, synthetic) from data/phase2/synthetic_world if it is empty
- imports the 60 Phase 1 calls with transcripts, the AI analyses from reports/phase1/ai_analyses.json,
  reviewed tags from data/phase1/reviewed_tags/ and the older manual tags, never overwriting rows teammates
  already saved in the database.

After setup, the dashboard and `salud-phase1 analyze` / `compare` work directly against the database.
Schema changes use the direct connection (DATABASE_URL_UNPOOLED).
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from .paths import (AI_ANALYSES, LEGACY_MANUAL_TAGS, PHASE1_SYNTHETIC_CALLS, PHASE2_SYNTHETIC_WORLD, REAL_TAGS,
                    REPO_ROOT, SYNTHETIC_TAGS)
from .phase1.calls import phase1_sources
from .phase1.checklist import LABEL_COLUMNS, TARGETS, clean_transcript, load_synthetic
from .phase1.store import Store, database_url
from .phase2 import schema as S

MIGRATIONS = REPO_ROOT / "db" / "migrations"
# Parents before children, so foreign keys hold during the load.
CRM_TABLES = [S.GUARANTOR, S.ACCOUNTS, S.CONTACTS, S.BRIDGE, S.RECEIPTS, S.QA_REVIEW]
PHASE1_TABLES = ["calls", "ai_analyses", "ai_checklist", "ai_coaching", "reviewed_tags", "legacy_manual_tags"]


def connect() -> psycopg.Connection:
    return psycopg.connect(database_url(direct=True))


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply migrations not yet recorded in public.schema_migrations, in file-name order."""
    conn.execute("CREATE TABLE IF NOT EXISTS public.schema_migrations "
                 "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
    done = {r[0] for r in conn.execute("SELECT name FROM public.schema_migrations")}
    applied = []
    for f in sorted(MIGRATIONS.glob("*.sql")):
        if f.name in done:
            continue
        conn.execute(f.read_text())
        conn.execute("INSERT INTO public.schema_migrations (name) VALUES (%s)", (f.name,))
        applied.append(f.name)
    return applied


def _value(v, pg_type: str):
    """CSV value -> Python value for COPY. Empty and NaN become NULL."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return None
    if pg_type == "integer":
        return int(float(v))
    if pg_type == "boolean":
        return bool(int(float(v)))
    return v


def _copy(conn: psycopg.Connection, schema: str, table: str, df: pd.DataFrame) -> int:
    types = dict(conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = %s AND table_name = %s",
        (schema, table)).fetchall())
    cols = [c for c in df.columns if c in types]
    col_sql = ", ".join(f'"{c}"' for c in cols)
    with conn.cursor().copy(f'COPY {schema}."{table}" ({col_sql}) FROM STDIN') as cp:
        for row in df[cols].itertuples(index=False):
            cp.write_row([_value(v, types[c]) for v, c in zip(row, cols)])
    return len(df)


def load_world_if_empty(conn: psycopg.Connection, world=PHASE2_SYNTHETIC_WORLD) -> dict[str, int]:
    if conn.execute(f'SELECT EXISTS (SELECT 1 FROM big_synthetic."{S.ACCOUNTS}")').fetchone()[0]:
        return {}
    counts = {}
    for t in CRM_TABLES:
        df = pd.read_csv(world / "crm" / f"{t}.csv", dtype=str, keep_default_na=False)
        counts[f"big_synthetic.{t}"] = _copy(conn, "big_synthetic", t, df.drop_duplicates() if t == S.BRIDGE else df)
    return counts


def import_calls(conn: psycopg.Connection) -> int:
    """Add the 60 Phase 1 calls with their transcripts (fills in transcripts missing on existing rows)."""
    planted = load_synthetic(PHASE1_SYNTHETIC_CALLS).set_index("contact_id") \
        if (PHASE1_SYNTHETIC_CALLS / "truth").exists() else pd.DataFrame()
    n = 0
    for src in phase1_sources():
        for p in src.files():
            stem, synthetic = p.stem, src.key == "synthetic"
            pl = planted.loc[stem] if synthetic and stem in planted.index else None
            conn.execute(
                "INSERT INTO phase1.calls (call_id, source, file_id, transcript, contact_id, planted, "
                "planted_scenario, planted_sentiment, planted_outcome) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (call_id) DO UPDATE SET transcript = COALESCE(phase1.calls.transcript, EXCLUDED.transcript), "
                "planted = COALESCE(phase1.calls.planted, EXCLUDED.planted)",
                (f"{src.key}/{p.name}", src.key, p.name, clean_transcript(p.read_text()),
                 stem if synthetic else None, Jsonb({t: pl[t] for t in TARGETS}) if pl is not None else None,
                 pl["scenario"] if pl is not None else None, pl["sentiment"] if pl is not None else None,
                 pl["outcome"] if pl is not None else None))
            n += 1
    return n


def import_ai_analyses(store: Store) -> int:
    if not AI_ANALYSES.exists():
        return 0
    data = json.loads(AI_ANALYSES.read_text())
    known = {c["call_id"] for c in store.calls()}
    return sum(store.add_ai_analysis(cid, e["analysis"], data["model"], data["prompt_version"])
               for cid, e in data.get("calls", {}).items() if cid in known)


def _import_tags(conn: psycopg.Connection, csv, source: str, table: str) -> int:
    if not csv.exists():
        return 0
    df = pd.read_csv(csv, dtype=str, keep_default_na=False).reindex(columns=LABEL_COLUMNS, fill_value="")
    df = df[(df["rater"] != "") & (df[TARGETS] != "").any(axis=1)]
    cols = ["call_id", "rater"] + TARGETS + ["notes", "llm_model", "overrides"]
    n = 0
    for r in df.to_dict("records"):
        vals = [f"{source}/{r['file_id']}", r["rater"]] + [r[t] or None for t in TARGETS] + \
               [r["notes"] or None, r["llm_model"] or None, [x for x in r["overrides"].split(";") if x]]
        n += conn.execute(f"INSERT INTO phase1.{table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
                          "ON CONFLICT (call_id, rater) DO NOTHING RETURNING 1", vals).fetchone() is not None
    return n


def setup() -> None:
    with connect() as conn:  # one transaction for schema, world, calls and tags
        applied = migrate(conn)
        world = load_world_if_empty(conn)
        calls = import_calls(conn)
        tags = {"reviewed (real)": _import_tags(conn, REAL_TAGS, "real", "reviewed_tags"),
                "reviewed (synthetic)": _import_tags(conn, SYNTHETIC_TAGS, "synthetic", "reviewed_tags"),
                "legacy manual": _import_tags(conn, LEGACY_MANUAL_TAGS, "real", "legacy_manual_tags")}
    store = Store()
    added = import_ai_analyses(store)
    store.close()
    print("Migrations applied:", ", ".join(applied) or "none (schema up to date)")
    print("Synthetic world:", "loaded " + ", ".join(f"{k} {v}" for k, v in world.items()) if world else "already loaded")
    print(f"Calls: {calls} imported or checked. AI analyses added: {added}.")
    print("Tags added:", ", ".join(f"{k} {v}" for k, v in tags.items()))
    status()


def status() -> None:
    with connect() as conn:
        for schema, tables in (("big_synthetic", CRM_TABLES), ("phase1", PHASE1_TABLES)):
            for t in tables:
                try:
                    n = conn.execute(f'SELECT count(*) FROM {schema}."{t}"').fetchone()[0]
                except psycopg.errors.UndefinedTable:
                    conn.rollback()
                    n = "missing (run salud-db setup)"
                print(f"  {schema + '.' + t:50s} {n}")
        try:
            rows = conn.execute("SELECT c.source, count(*), count(a.call_id) FROM phase1.calls c "
                                "LEFT JOIN phase1.ai_analyses a USING (call_id) GROUP BY 1 ORDER BY 1").fetchall()
            for source, n, analyzed in rows:
                print(f"  {source} calls: {n}, with AI analysis: {analyzed}")
        except psycopg.errors.UndefinedColumn:
            conn.rollback()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="salud-db", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["setup", "status"])
    args = ap.parse_args(argv)
    try:
        {"setup": setup, "status": status}[args.cmd]()
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
