"""
One-time migration: read existing JSON flat files → diffdx.db

Run once before starting the server with the new SQLite backend:
    python loop1/web/migrate_to_sqlite.py

Safe to re-run — existing DB rows are overwritten, not duplicated.
"""
import json
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_FILE  = DATA_DIR / "diffdx.db"

MIGRATIONS = [
    ("users",           DATA_DIR / "users.json",           {}),
    ("doctors",         DATA_DIR / "doctors.json",          []),
    ("appointments",    DATA_DIR / "appointments.json",     {}),
    ("session_uploads", DATA_DIR / "session_uploads.json",  {}),
    ("messages",        DATA_DIR / "messages.json",         []),
    ("waitlist",        DATA_DIR / "waitlist.json",         []),
    ("blocked_dates",   DATA_DIR / "blocked_dates.json",    {}),
    ("second_opinions", DATA_DIR / "second_opinions.json",  []),
]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS store (
            collection TEXT PRIMARY KEY,
            data       TEXT NOT NULL
        )
    """)
    conn.commit()

    for collection, json_path, default in MIGRATIONS:
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                count = len(data) if isinstance(data, (dict, list)) else "?"
                print(f"  {collection:20s} ← {json_path.name}  ({count} records)")
            except Exception as e:
                print(f"  {collection:20s} ← {json_path.name}  ERROR reading: {e}")
                data = default
        else:
            print(f"  {collection:20s}   (no JSON file found, initialising empty)")
            data = default

        conn.execute(
            "INSERT OR REPLACE INTO store (collection, data) VALUES (?, ?)",
            (collection, json.dumps(data, ensure_ascii=False)),
        )

    conn.commit()
    conn.close()
    print(f"\nDone — database written to {DB_FILE}")
    print("You can now start the server. The JSON files are no longer used.")


if __name__ == "__main__":
    main()
