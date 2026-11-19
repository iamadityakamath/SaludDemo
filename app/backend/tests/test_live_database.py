"""Runs only when DATABASE_URL is configured: the connection must refuse writes."""
import psycopg
import pytest

from salud_app.config import get_settings
from salud_app.db import read_only_connection

pytestmark = pytest.mark.skipif(not get_settings().database_url, reason="DATABASE_URL not configured")


def test_connection_is_read_only():
    with read_only_connection() as conn:
        assert conn.execute("SHOW default_transaction_read_only").fetchone()["default_transaction_read_only"] == "on"
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute("CREATE TEMP TABLE app_write_probe (a int)")
