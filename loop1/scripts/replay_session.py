"""
Replay a logged session with a different doctor prompt.

Usage:
    python scripts/replay_session.py \
        --session <session_id> \
        --prompt <prompt_file> \
        --out <output_jsonl>

What it does:
  - Reads the original session JSONL (from logs/sessions/session_<id>.jsonl)
  - For each turn: uses the logged profile_state and reconstructs the history
    from preceding turn_complete events
  - Calls the doctor LLM with the new prompt at temperature=0
  - Writes a new JSONL with replay_of and prompt_version fields on each record
  - Patient inputs are taken verbatim from the original log
  - Profile updater and compression are NOT re-run — the logged profile_state is used

Constraints:
  - Requires turn_complete events with profile_state (Phase 6+ logs only)
  - Uses logged retrieved_exemplar_ids from the original log (not re-retrieved)
  - Temperature forced to 0 for determinism
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is on the path when run from the loop1/ directory
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from loop1.config import config
from loop1.doctor import build_doctor_prompt
from loop1.llm import call_llm
from loop1.logging_utils import read_session_events
from loop1.retrieval import get_exemplar_by_id
from loop1.schemas import (
    Demographics,
    DiagnosisEntry,
    DoctorTurnOutput,
    History,
    PatientProfile,
    Symptom,
    TurnRecord,
)

import re
from pydantic import ValidationError


def _load_prompt(prompt_file: Path) -> str:
    return prompt_file.read_text(encoding="utf-8").strip()


def _events_to_turn_records(turn_events: list[dict]) -> list[TurnRecord]:
    records: list[TurnRecord] = []
    for ev in turn_events:
        doc_out = DoctorTurnOutput(**ev["doctor_output"])
        records.append(
            TurnRecord(
                turn_index=ev["turn_index"],
                doctor_output=doc_out,
                patient_answer=ev["patient_answer"],
                retrieved_exemplar_ids=ev.get("retrieved_exemplar_ids", []),
                timestamp=ev["timestamp"],
            )
        )
    return records


def _profile_from_state(state: dict) -> PatientProfile:
    return PatientProfile(**state)


def _recent_history(history: list[TurnRecord], profile: PatientProfile, keep_recent: int) -> list[TurnRecord]:
    if profile.running_summary:
        return history[-keep_recent:]
    return history


def _call_doctor(
    system_prompt: str,
    profile: PatientProfile,
    history: list[TurnRecord],
    turn_index: int,
    exemplar_ids: list[str],
    model: str,
    max_retries: int = 3,
) -> DoctorTurnOutput | None:
    from loop1.doctor import _render_exemplars

    # Load exemplars by ID from the original log
    exemplars = []
    for eid in exemplar_ids:
        try:
            exemplars.append(get_exemplar_by_id(eid))
        except ValueError:
            pass

    # Build prompt but swap in the new system prompt
    messages = build_doctor_prompt(profile, history, turn_index, exemplars or None)
    messages[0]["content"] = system_prompt  # replace system prompt

    last_err: Exception | None = None
    for attempt in range(max_retries):
        raw = call_llm(model=model, messages=messages, temperature=0)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_err = exc
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Invalid JSON (attempt {attempt + 1}): {exc}. Reply JSON only."},
            ]
            continue
        try:
            return DoctorTurnOutput(**data)
        except (ValidationError, TypeError) as exc:
            last_err = exc
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Schema error (attempt {attempt + 1}): {exc}. Fix and reply JSON only."},
            ]
    print(f"  [WARN] Turn {turn_index} failed after {max_retries} attempts: {last_err}", file=sys.stderr)
    return None


def replay(
    session_id: str,
    prompt_file: Path,
    out_path: Path,
    model: str | None = None,
) -> None:
    if model is None:
        model = config["models"]["doctor"]

    events = read_session_events(session_id)
    if not events:
        print(f"No events found for session {session_id}", file=sys.stderr)
        sys.exit(1)

    turn_events = [e for e in events if e["event_type"] == "turn_complete"]
    if not turn_events:
        print("No turn_complete events found — nothing to replay.", file=sys.stderr)
        sys.exit(1)

    # Check that profile_state is present (Phase 6+ logs)
    if "profile_state" not in turn_events[0]:
        print(
            "ERROR: turn_complete events do not have profile_state. "
            "This session was logged before Phase 6. Cannot replay.",
            file=sys.stderr,
        )
        sys.exit(1)

    system_prompt = _load_prompt(prompt_file)
    keep_recent = config["thresholds"]["compression_keep_recent"]
    prompt_version = prompt_file.stem  # e.g. "doctor_v0_5"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    replayed_records: list[dict] = []

    print(f"Replaying session {session_id} with prompt {prompt_file.name} ({len(turn_events)} turns)")

    preceding_records: list[TurnRecord] = []

    for i, ev in enumerate(turn_events):
        turn_index = ev["turn_index"]
        profile = _profile_from_state(ev["profile_state"])
        exemplar_ids = ev.get("retrieved_exemplar_ids", [])
        patient_answer = ev["patient_answer"]

        history_slice = _recent_history(preceding_records, profile, keep_recent)

        print(f"  Turn {turn_index}: replaying...")
        replayed_output = _call_doctor(
            system_prompt=system_prompt,
            profile=profile,
            history=history_slice,
            turn_index=turn_index,
            exemplar_ids=exemplar_ids,
            model=model,
        )

        record: dict = {
            "schema_version": "0.5.0",
            "replay_of": session_id,
            "prompt_version": prompt_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn_index": turn_index,
            "doctor_output": replayed_output.model_dump() if replayed_output else None,
            "patient_answer": patient_answer,
            "retrieved_exemplar_ids": exemplar_ids,
            "original_question": ev["doctor_output"]["chosen_question"],
        }
        replayed_records.append(record)

        # Build a TurnRecord for history reconstruction using the ORIGINAL patient answer
        # and the REPLAYED doctor output (so subsequent turns see the new prompt's questions)
        if replayed_output is not None:
            preceding_records.append(
                TurnRecord(
                    turn_index=turn_index,
                    doctor_output=replayed_output,
                    patient_answer=patient_answer,
                    retrieved_exemplar_ids=exemplar_ids,
                    timestamp=ev["timestamp"],
                )
            )

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in replayed_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nReplay complete. {len(replayed_records)} turns written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a logged session with a new doctor prompt.")
    parser.add_argument("--session", required=True, help="Session ID to replay")
    parser.add_argument("--prompt", required=True, type=Path, help="Path to new prompt file")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--model", default=None, help="Override doctor model")
    args = parser.parse_args()

    replay(
        session_id=args.session,
        prompt_file=args.prompt,
        out_path=args.out,
        model=args.model,
    )


if __name__ == "__main__":
    main()
