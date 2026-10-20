import re

from salud_pipeline.db import MIGRATIONS, _value
from salud_pipeline.phase2 import schema as S


def test_value_conversion():
    assert _value("", "integer") is None and _value(float("nan"), "text") is None
    assert _value("785570001.0", "integer") == 785570001
    assert _value("1", "boolean") is True and _value("0.0", "boolean") is False
    assert _value("2026-04-01 10:00:00", "timestamp without time zone") == "2026-04-01 10:00:00"


def test_big_synthetic_schema_matches_salud_columns():
    sql = (MIGRATIONS / "001_big_synthetic.sql").read_text()
    code = "\n".join(l for l in sql.splitlines() if not l.lstrip().startswith("--"))
    assert not re.search(r"uniqueidentifier|\bmoney\b|\bdatetime\b|nvarchar|\[", code)  # no SQL Server syntax
    for table, cols in S.COLUMNS.items():
        block = re.search(rf'CREATE TABLE IF NOT EXISTS big_synthetic\."{table}" \((.*?)\n\);', code, re.S)
        assert block, table
        assert re.findall(r'^\s+"(\w+)" ', block.group(1), re.M) == cols, table
    assert code.count("FOREIGN KEY") == 8


def test_shared_db_migrations_allow_real_calls_with_guards():
    sql = (MIGRATIONS / "003_shared_team_db.sql").read_text()
    for t in ("calls", "ai_analyses", "ai_checklist", "ai_coaching", "reviewed_tags"):
        assert f"DROP CONSTRAINT IF EXISTS {t}_call_id_check" in sql
    assert "CHECK (call_id = source || '/' || file_id)" in sql
    assert "CHECK (source = 'synthetic' OR contact_id IS NULL)" in sql
    assert "planted IS NULL" in (MIGRATIONS / "004_planted_labels.sql").read_text()


def test_live_schema_is_fully_migrated(test_db_url):
    import psycopg

    with psycopg.connect(test_db_url) as c:
        applied = {r[0] for r in c.execute("SELECT name FROM public.schema_migrations")}
        assert applied == {f.name for f in MIGRATIONS.glob("*.sql")}
        cols = {r[0] for r in c.execute("SELECT column_name FROM information_schema.columns "
                                        "WHERE table_schema = 'phase1' AND table_name = 'calls'")}
        assert {"source", "file_id", "transcript", "planted"} <= cols
