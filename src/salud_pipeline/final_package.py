"""Load Salud's final package (84 real calls: CRM tabs + transcripts) into the `final` schema, as received.

    salud-db load-final --test      # into the Neon 'tests' branch (DATABASE_URL_TEST) first
    salud-db load-final             # into the team database
    salud-db load-final --replace   # empty the final tables and load again

One table per workbook tab, plus final.transcripts and final.option_codes (from the Oct 14 handoff). Every
check runs before anything is written; the load is one transaction; afterwards every row is read back and
compared field by field with the workbook. Applies only db/migrations/009_final_package.sql, never another
pending migration.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg

from .paths import FINAL_PACKAGE, REPO_ROOT, SCHEMA_HANDOFF
from .phase1.prompt import split_turns
from .sentiment.turns import scorable_turns

MIGRATION = REPO_ROOT / "db" / "migrations" / "009_final_package.sql"
WORKBOOK = "BIG_Combined_Data_Updated.xlsx"
HANDOFF_JSON = SCHEMA_HANDOFF / "BIG_Schema.json"

# Workbook tab -> table, parents before children (the order they are loaded in).
TABS = {
    "Accounts": "accounts",
    "Contact History": "contact_history",
    "Account Contact Links": "account_contact_links",
    "Transactions": "transactions",
    "QA Reviews": "qa_reviews",
    "Schema Fields": "schema_fields",
    "Relationships": "relationships",
}
# Reference tabs have plain headers; renamed to snake case. CRM tabs keep Salud's column names.
RENAME = {
    "Schema Fields": {"Data Sheet": "data_sheet", "Entity": "entity", "Column": "column_name",
                      "Display Name": "display_name", "Source SQL Type": "source_sql_type",
                      "Description": "description", "Schema Context": "schema_context"},
    "Relationships": {"Child Sheet": "child_sheet", "Child Key": "child_key", "Parent Sheet": "parent_sheet",
                      "Parent Key": "parent_key", "Context": "context"},
}
KEYS = {
    "accounts": ["salud_gphpvaccountsId"],
    "contact_history": ["salud_gphpvcontacthistoryId"],
    "account_contact_links": ["salud_gphpvaccountsid", "salud_gphpvcontacthistoryid"],
    "transactions": ["salud_gphpvreceiptsId"],
    "qa_reviews": ["salud_gphpvqualityreviewId"],
    "schema_fields": ["data_sheet", "column_name"],
    "relationships": ["child_sheet", "child_key"],
    "option_codes": ["entity", "field", "code"],
    "transcripts": ["salud_TCNRecordID"],
}
# (child table, child column, parent table, parent column): enforced by the schema, so checked before writing.
LINKS = [
    ("account_contact_links", "salud_gphpvaccountsid", "accounts", "salud_gphpvaccountsId"),
    ("account_contact_links", "salud_gphpvcontacthistoryid", "contact_history", "salud_gphpvcontacthistoryId"),
    ("transactions", "salud_ReceiptsId", "accounts", "salud_gphpvaccountsId"),
    ("qa_reviews", "salud_QualityReviewId", "contact_history", "salud_gphpvcontacthistoryId"),
    ("transcripts", "salud_TCNRecordID", "contact_history", "salud_TCNRecordID"),
]
MULTI_SELECT = {"salud_ActionsTaken"}  # several labels in one cell, joined by '; '
LOAD_ORDER = list(TABS.values()) + ["option_codes", "transcripts"]
OTHER_SCHEMAS = ("phase1", "big_synthetic")


class PackageError(RuntimeError):
    pass


# ---------- read ----------

def read_package(package: Path = FINAL_PACKAGE, handoff_json: Path = HANDOFF_JSON) -> dict[str, pd.DataFrame]:
    """Every table to load, as DataFrames with the database's column names."""
    wb = package / WORKBOOK
    if not wb.exists():
        raise PackageError(f"Workbook not found: {wb}")
    xl = pd.ExcelFile(wb)
    missing = [t for t in TABS if t not in xl.sheet_names]
    if missing:
        raise PackageError(f"Workbook is missing tabs: {missing}")
    out = {table: xl.parse(tab).rename(columns=RENAME.get(tab, {})) for tab, table in TABS.items()}

    if not handoff_json.exists():
        raise PackageError(f"Schema handoff not found (needed for option codes): {handoff_json}")
    out["option_codes"] = pd.DataFrame(json.loads(handoff_json.read_text())["option_codes"])[
        ["entity", "field", "code", "label", "project_use"]]

    rows = []
    for p in sorted((package / "transcripts").glob("*.txt")):
        text = p.read_bytes().decode("utf-8").replace("\r\n", "\n")
        rows.append({"salud_TCNRecordID": p.stem, "file_name": p.name, "transcript": text,
                     "turns": len(split_turns(text)),
                     "has_caller": any(t["role"] == "caller" for t in scorable_turns(text))})
    out["transcripts"] = pd.DataFrame(rows, columns=["salud_TCNRecordID", "file_name", "transcript", "turns",
                                                     "has_caller"])
    return out


# ---------- values ----------

def _missing(v) -> bool:
    return v is None or v is pd.NaT or (isinstance(v, float) and math.isnan(v))


def convert(v, pg_type: str):
    """Workbook value -> Python value for the column's Postgres type. Text is kept exactly as given."""
    if _missing(v):
        return None
    if pg_type in ("text", "character varying"):
        return v if isinstance(v, str) else str(v)
    if pg_type.startswith("timestamp"):
        return pd.Timestamp(v).to_pydatetime()
    if pg_type == "numeric":
        return Decimal(str(v))
    if pg_type == "integer":
        if float(v) != int(float(v)):
            raise PackageError(f"Not a whole number: {v!r}")
        return int(float(v))
    if pg_type == "boolean":
        return bool(v)
    raise PackageError(f"No conversion for type {pg_type}")


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(b, Decimal):
        return Decimal(a) == b
    if isinstance(a, datetime) and isinstance(b, datetime):
        return a == b
    return a == b


# ---------- checks (before writing) ----------

def check(pkg: dict[str, pd.DataFrame], columns: dict[str, dict[str, str]]) -> None:
    """Raise PackageError listing every problem; nothing is written if any check fails."""
    errors = []
    for table in LOAD_ORDER:
        df, cols = pkg[table], columns[table]
        extra = [c for c in df.columns if c not in cols]
        if extra:
            errors.append(f"{table}: columns not in the schema: {extra}")
        if table in TABS.values():
            absent = [c for c in cols if c not in df.columns]
            if absent:
                errors.append(f"{table}: schema columns missing from the workbook: {absent}")
        key = KEYS[table]
        if df[key].isna().any().any():
            errors.append(f"{table}: empty key in {int(df[key].isna().any(axis=1).sum())} rows")
        if df.duplicated(key).any():
            errors.append(f"{table}: {int(df.duplicated(key).sum())} duplicate keys {key}")
        for c in df.columns:
            if c in cols:
                try:
                    df[c].map(lambda v, t=cols[c]: convert(v, t))
                except (PackageError, ValueError, ArithmeticError) as e:
                    errors.append(f"{table}.{c}: {e}")

    for child, ccol, parent, pcol in LINKS:
        vals = pkg[child][ccol].dropna()
        bad = vals[~vals.isin(pkg[parent][pcol])]
        if len(bad):
            errors.append(f"{child}.{ccol}: {len(bad)} values not found in {parent}.{pcol}")

    ch, tr = pkg["contact_history"], pkg["transcripts"]
    no_file = set(ch["salud_TCNRecordID"].dropna()) - set(tr["salud_TCNRecordID"])
    if no_file:
        errors.append(f"{len(no_file)} contact_history rows have no transcript file")

    # Every option label must be one of the handoff's labels for that field.
    sheet_entity = dict(zip(pkg["schema_fields"]["data_sheet"], pkg["schema_fields"]["entity"]))
    oc = pkg["option_codes"]
    for tab, table in TABS.items():
        entity = sheet_entity.get(tab)
        for c in pkg[table].columns:
            labels = set(oc[(oc["entity"] == entity) & (oc["field"] == c.lower())]["label"])
            if not labels:
                continue
            vals = pkg[table][c].dropna().astype(str)
            if c in MULTI_SELECT:
                vals = pd.Series([p.strip() for v in vals for p in v.split(";")])
            unknown = sorted(set(vals) - labels)
            if unknown:
                errors.append(f"{table}.{c}: labels not in the handoff's option codes: {unknown}")

    if errors:
        raise PackageError("The package failed its checks; nothing was written:\n  - " + "\n  - ".join(errors))


# ---------- database ----------

def apply_migration(conn: psycopg.Connection) -> bool:
    """Apply 009 only (not other pending migrations). True if it was applied now."""
    conn.execute("CREATE TABLE IF NOT EXISTS public.schema_migrations "
                 "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
    if conn.execute("SELECT 1 FROM public.schema_migrations WHERE name = %s", (MIGRATION.name,)).fetchone():
        return False
    conn.execute(MIGRATION.read_text())
    conn.execute("INSERT INTO public.schema_migrations (name) VALUES (%s)", (MIGRATION.name,))
    return True


def column_types(conn: psycopg.Connection) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {t: {} for t in LOAD_ORDER}
    for t, c, typ in conn.execute("SELECT table_name, column_name, data_type FROM information_schema.columns "
                                  "WHERE table_schema = 'final' ORDER BY table_name, ordinal_position"):
        if t in out:
            out[t][c] = typ
    return out


def counts(conn: psycopg.Connection, schema: str) -> dict[str, int]:
    tables = [r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables "
                                         "WHERE table_schema = %s AND table_type = 'BASE TABLE'", (schema,))]
    return {t: conn.execute(f'SELECT count(*) FROM {schema}."{t}"').fetchone()[0] for t in sorted(tables)}


def _copy(conn: psycopg.Connection, table: str, df: pd.DataFrame, types: dict[str, str]) -> None:
    cols = [c for c in df.columns if c in types]
    col_sql = ", ".join(f'"{c}"' for c in cols)
    with conn.cursor().copy(f'COPY final."{table}" ({col_sql}) FROM STDIN') as cp:
        for row in df[cols].itertuples(index=False):
            cp.write_row([convert(v, types[c]) for v, c in zip(row, cols)])


def verify(conn: psycopg.Connection, pkg: dict[str, pd.DataFrame], types: dict[str, dict[str, str]]) -> list[str]:
    """Read every row back and compare it field by field with the package. Returns the problems found."""
    problems = []
    for table in LOAD_ORDER:
        df, key = pkg[table], KEYS[table]
        cols = [c for c in df.columns if c in types[table]]
        select = ", ".join(f'"{c}"' for c in key + cols)
        stored = {tuple(r[:len(key)]): r[len(key):] for r in conn.execute(f'SELECT {select} FROM final."{table}"')}
        if len(stored) != len(df):
            problems.append(f"{table}: {len(stored)} rows stored, {len(df)} in the package")
        bad = 0
        for row in df[key + cols].itertuples(index=False):
            got = stored.get(tuple(convert(v, types[table][c]) for v, c in zip(row[:len(key)], key)))
            want = [convert(v, types[table][c]) for v, c in zip(row[len(key):], cols)]
            if got is None or not all(_same(w, g) for w, g in zip(want, got)):
                bad += 1
        if bad:
            problems.append(f"{table}: {bad} rows differ from the package")
    for table, col in (("transactions", "salud_Amount"), ("contact_history", "salud_PaymentAmount")):
        want = sum((Decimal(str(v)) for v in pkg[table][col].dropna()), Decimal(0))
        got = conn.execute(f'SELECT coalesce(sum("{col}"), 0) FROM final.{table}').fetchone()[0]
        if want != got:
            problems.append(f"{table}.{col}: total {got} stored, {want} in the package")
    return problems


def load(url: str, package: Path = FINAL_PACKAGE, replace: bool = False) -> None:
    pkg = read_package(package)
    with psycopg.connect(url, prepare_threshold=None) as conn:
        before = {s: counts(conn, s) for s in OTHER_SCHEMAS}
        applied = apply_migration(conn)
        types = column_types(conn)
        check(pkg, types)

        existing = counts(conn, "final")
        if any(existing.get(t) for t in LOAD_ORDER) and not replace:
            expected = {t: len(pkg[t]) for t in LOAD_ORDER}
            if {t: existing.get(t, 0) for t in LOAD_ORDER} != expected:
                raise PackageError(f"final already holds different data ({existing}); rerun with --replace")
            print("final is already loaded; checking it against the package.")
        else:
            if replace:
                conn.execute("TRUNCATE " + ", ".join(f'final."{t}"' for t in LOAD_ORDER))
            phase1_ids = {r[0] for r in conn.execute("SELECT call_id FROM phase1.calls WHERE source = 'real'")} \
                if "calls" in before["phase1"] else set()
            pkg["transcripts"]["phase1_call_id"] = [f"real/{n}" if f"real/{n}" in phase1_ids else None
                                                    for n in pkg["transcripts"]["file_name"]]
            types["transcripts"]["phase1_call_id"] = "text"
            for t in LOAD_ORDER:
                _copy(conn, t, pkg[t], types[t])

        problems = verify(conn, pkg, types)
        after = {s: counts(conn, s) for s in OTHER_SCHEMAS}
        if after != before:
            problems.append("phase1 / big_synthetic row counts changed")
        if problems:
            conn.rollback()
            raise PackageError("Verification failed; the load was rolled back:\n  - " + "\n  - ".join(problems))

        print(f"Migration {MIGRATION.name}: {'applied' if applied else 'already applied'}")
        for t, n in counts(conn, "final").items():
            print(f"  final.{t:24s} {n}")
        linked = conn.execute("SELECT count(phase1_call_id) FROM final.transcripts").fetchone()[0]
        print(f"Verified: every row matches the package; money totals match; {linked} transcripts linked to "
              f"phase1; phase1 and big_synthetic unchanged.")
