from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loop1.config import config

_SCHEMA_VERSION = config["logging"]["schema_version"]


def _session_path(session_id: str) -> Path:
    log_dir = Path(config["logging"]["session_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"session_{session_id}.jsonl"


def log_event(session_id: str, event_type: str, data: dict[str, Any]) -> None:
    entry = {
        "schema_version": _SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "session_id": session_id,
        **data,
    }
    path = _session_path(session_id)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def write_final_record(session_id: str, record: dict[str, Any]) -> Path:
    final_dir = Path(config["logging"]["final_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    path = final_dir / f"final_{session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.flush()
    return path


def read_session_events(session_id: str) -> list[dict[str, Any]]:
    path = _session_path(session_id)
    if not path.exists():
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
