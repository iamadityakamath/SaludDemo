"""Connection pool to the team database (Neon, pooled URL). Statements run in autocommit; multi-statement writes use
explicit transactions. Server-side prepared statements are off (PgBouncer pooling doesn't keep them)."""

from __future__ import annotations

import threading

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import get_settings


class DatabaseNotConfigured(RuntimeError):
    pass


_pool: ConnectionPool | None = None
_lock = threading.Lock()


def pool() -> ConnectionPool:
    global _pool
    with _lock:
        if _pool is None:
            url = get_settings().database_url
            if not url:
                raise DatabaseNotConfigured("DATABASE_URL is not set (environment or repository root .env)")
            _pool = ConnectionPool(url, min_size=0, max_size=4, open=True, check=ConnectionPool.check_connection,
                                   kwargs={"row_factory": dict_row, "prepare_threshold": None, "autocommit": True})
        return _pool


def fetch_all(sql: str, params=()) -> list[dict]:
    with pool().connection() as c:
        return c.execute(sql, params).fetchall()


def ping() -> bool:
    try:
        return fetch_all("SELECT 1 AS ok")[0]["ok"] == 1
    except Exception:  # noqa: BLE001 - the health check reports any failure as "not reachable"
        return False
