from __future__ import annotations

import json
import logging
from pathlib import Path

from loop3.routing.router import route
from loop3.routing.routing_schema import RoutingDecision

_log = logging.getLogger(__name__)


def _extract_from_session(events: list[dict]) -> tuple[list[tuple[str, float]], float, str]:
    """
    Extract final_differential, final_confidence, and session_id from session events.
    Reads the last session_end or turn_complete event with a doctor_output.
    """
    session_id = ""
    final_diff: list[tuple[str, float]] = []
    final_confidence = 0.0

    for event in events:
        if not session_id:
            session_id = event.get("session_id", "")

        if event.get("event_type") in ("session_end", "turn_complete"):
            doc = event.get("doctor_output", {})
            raw_diff = doc.get("current_differential", [])
            if raw_diff:
                final_diff = [(d["dx"], float(d["prob"])) for d in raw_diff]
                final_confidence = float(doc.get("confidence_to_stop", 0.0))

    return final_diff, final_confidence, session_id


def run_routing_from_session(
    session_jsonl_path: str | Path,
    patient_id: str | None = None,
) -> RoutingDecision:
    """
    Read a completed Loop 1 session JSONL.
    Extract final_differential and final_confidence from the last doctor turn.
    Return a RoutingDecision.
    """
    path = Path(session_jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        raise ValueError(f"Session file is empty: {path}")

    final_diff, final_confidence, session_id = _extract_from_session(events)

    if not final_diff:
        raise ValueError(f"No differential found in session: {path}")

    return route(
        session_id=session_id,
        final_differential=final_diff,
        final_confidence=final_confidence,
        patient_id=patient_id,
    )
