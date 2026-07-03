"""SQLite-backed cache with TTL for data fetcher results.

Stores serialized DataFrames and JSON blobs with expiry timestamps.
Avoids hammering yfinance / Screener.in on repeated calls.
"""

import json
import sqlite3
import time
from io import StringIO
from pathlib import Path

import pandas as pd

from app.config import settings


def _cache_db() -> Path:
    return settings.cache_db_path


def _get_conn() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(str(_cache_db()), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                kind  TEXT NOT NULL,
                ts    REAL NOT NULL
            )
            """
        )
        return conn
    except sqlite3.DatabaseError:
        # Corrupt DB — delete and recreate
        _cache_db().unlink(missing_ok=True)
        conn = sqlite3.connect(str(_cache_db()), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                kind  TEXT NOT NULL,
                ts    REAL NOT NULL
            )
            """
        )
        return conn


def _is_expired(ts: float, ttl: int) -> bool:
    return (time.time() - ts) > ttl


# ── DataFrame cache ──────────────────────────────────────────────


def get_dataframe(key: str, ttl: int) -> pd.DataFrame | None:
    """Retrieve a cached DataFrame if it exists and hasn't expired."""
    try:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT value, ts FROM cache WHERE key = ? AND kind = 'df'", (key,)
            ).fetchone()
            if row is None or _is_expired(row[1], ttl):
                return None
            return pd.read_json(StringIO(row[0]), orient="split")
        finally:
            conn.close()
    except Exception:
        return None


def set_dataframe(key: str, df: pd.DataFrame) -> None:
    """Store a DataFrame in cache."""
    try:
        conn = _get_conn()
        try:
            payload = df.to_json(orient="split", date_format="iso")
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, kind, ts) VALUES (?, ?, 'df', ?)",
                (key, payload, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # Cache write failure is non-fatal


# ── JSON cache ───────────────────────────────────────────────────


def get_json(key: str, ttl: int) -> dict | list | None:
    """Retrieve a cached JSON value if it exists and hasn't expired."""
    try:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT value, ts FROM cache WHERE key = ? AND kind = 'json'", (key,)
            ).fetchone()
            if row is None or _is_expired(row[1], ttl):
                return None
            return json.loads(row[0])
        finally:
            conn.close()
    except Exception:
        return None


def set_json(key: str, data: dict | list) -> None:
    """Store a JSON-serializable value in cache."""
    try:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, kind, ts) VALUES (?, ?, 'json', ?)",
                (key, json.dumps(data), time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # Cache write failure is non-fatal


# ── Maintenance ──────────────────────────────────────────────────


def clear_expired(max_age_seconds: int = 86400 * 7) -> int:
    """Remove entries older than max_age_seconds. Returns count deleted."""
    try:
        conn = _get_conn()
        try:
            cutoff = time.time() - max_age_seconds
            cursor = conn.execute("DELETE FROM cache WHERE ts < ?", (cutoff,))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
    except Exception:
        return 0


def clear_all() -> None:
    """Wipe entire cache. Use during development only."""
    try:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM cache")
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
