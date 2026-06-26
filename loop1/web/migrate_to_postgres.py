"""
Migration: copy data from SQLite (or raw JSON files) → PostgreSQL.

Usage:
    DATABASE_URL=postgresql://... python loop1/web/migrate_to_postgres.py

Set DATABASE_URL to the Railway Postgres connection string.
Run once before going live. Safe to re-run — existing rows are overwritten.
"""
import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SQLITE_DB = DATA_DIR / "diffdx.db"

COLLECTIONS = [
    ("users",           DATA_DIR / "users.json",           {}),
    ("doctors",         DATA_DIR / "doctors.json",          []),
    ("appointments",    DATA_DIR / "appointments.json",     {}),
    ("session_uploads", DATA_DIR / "session_uploads.json",  {}),
    ("messages",        DATA_DIR / "messages.json",         []),
    ("waitlist",        DATA_DIR / "waitlist.json",         []),
    ("blocked_dates",   DATA_DIR / "blocked_dates.json",    {}),
    ("second_opinions", DATA_DIR / "second_opinions.json",  []),
]


def _load_from_sqlite(collection: str, default):
    import sqlite3
    if not SQLITE_DB.exists():
        return None
    try:
        conn = sqlite3.connect(str(SQLITE_DB))
        row = conn.execute(
            "SELECT data FROM store WHERE collection=?", (collection,)
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def _load_from_json(json_path: Path, default):
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("ERROR: DATABASE_URL environment variable is not set.")
        print("Usage: DATABASE_URL=postgresql://... python migrate_to_postgres.py")
        sys.exit(1)

    import psycopg2
    # Parse URL manually to handle '@' characters in passwords.
    raw = url
    for prefix in ("postgresql://", "postgres://"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    at = raw.rfind("@")
    creds, hostpart = raw[:at], raw[at + 1:]
    colon = creds.find(":")
    pg_user = creds[:colon]
    pg_password = creds[colon + 1:]
    if "/" in hostpart:
        hostport, dbname = hostpart.split("/", 1)
    else:
        hostport, dbname = hostpart, "postgres"
    if ":" in hostport:
        host, port_str = hostport.rsplit(":", 1)
        pg_port = int(port_str)
    else:
        host, pg_port = hostport, 5432
    conn = psycopg2.connect(
        host=host, port=pg_port, dbname=dbname,
        user=pg_user, password=pg_password, sslmode="require"
    )
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS store (
            collection TEXT PRIMARY KEY,
            data       TEXT NOT NULL
        )
    """)
    conn.commit()

    print(f"Connected to PostgreSQL. Migrating {len(COLLECTIONS)} collections...\n")

    for collection, json_path, default in COLLECTIONS:
        # Prefer SQLite → JSON fallback → empty default
        data = _load_from_sqlite(collection, default)
        source = "SQLite"
        if data is None:
            data = _load_from_json(json_path, default)
            source = json_path.name if json_path.exists() else "empty default"
        if data is None:
            data = default
            source = "empty default"

        count = len(data) if isinstance(data, (dict, list)) else "?"
        print(f"  {collection:20s} ← {source:30s} ({count} records)")

        cur.execute("""
            INSERT INTO store (collection, data) VALUES (%s, %s)
            ON CONFLICT (collection) DO UPDATE SET data = EXCLUDED.data
        """, (collection, json.dumps(data, ensure_ascii=False)))

    conn.commit()
    cur.close()
    conn.close()
    print("\nDone — all data is now in PostgreSQL.")
    print("Deploy your app and set DATABASE_URL in Railway environment variables.")


if __name__ == "__main__":
    main()
