"""Read-only access to the team database.

Every connection is opened with `default_transaction_read_only=on`, so the app cannot change the shared database even by
mistake: any INSERT / UPDATE / DELETE fails at the database.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from .config import get_settings


class DatabaseNotConfigured(RuntimeError):
    pass


@contextmanager
def read_only_connection() -> Iterator[psycopg.Connection]:
    url = get_settings().database_url
    if not url:
        raise DatabaseNotConfigured("DATABASE_URL is not set (environment or repository root .env)")
    with psycopg.connect(url, options="-c default_transaction_read_only=on", row_factory=dict_row,
                         connect_timeout=10) as conn:
        yield conn


def fetch_all(sql: str, params: tuple | dict | None = None) -> list[dict]:
    with read_only_connection() as conn:
        return conn.execute(sql, params).fetchall()


def ping() -> bool:
    try:
        return fetch_all("SELECT 1 AS ok")[0]["ok"] == 1
    except Exception:  # noqa: BLE001 - health check reports any failure as "not reachable"
        return False
