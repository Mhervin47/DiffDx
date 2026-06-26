from __future__ import annotations

import json
import re
from pathlib import Path

from loop2.ddxplus.loader import is_in_exemplar_pool
from loop2.ddxplus.schemas import DDXPlusEvalResult, DDXPlusPatient


def _normalize(dx: str) -> str:
    return re.sub(r"\s+", " ", dx.strip().lower())


def _top_differential(session_events: list[dict]) -> list[tuple[str, float]]:
    """Return the final differential from the last turn_complete or session_end event."""
    final_diff: list[tuple[str, float]] = []

    for event in reversed(session_events):
        # session_end carries the final doctor_output
        if event.get("event_type") in ("session_end", "turn_complete"):
            doc_out = event.get("doctor_output", {})
            raw = doc_out.get("current_differential", [])
            if raw:
                final_diff = [(d["dx"], float(d["prob"])) for d in raw]
                break

    return final_diff


def _final_confidence(session_events: list[dict]) -> float:
    for event in reversed(session_events):
        if event.get("event_type") in ("session_end", "turn_complete"):
            doc_out = event.get("doctor_output", {})
            val = doc_out.get("confidence_to_stop")
            if val is not None:
                return float(val)
    return 0.0


def _terminated_on_confidence(session_events: list[dict]) -> bool:
    for event in session_events:
        if event.get("event_type") == "session_end":
            return event.get("termination_reason") == "confidence_threshold"
    return False


def _total_turns(session_events: list[dict]) -> int:
    return sum(1 for e in session_events if e.get("event_type") == "turn_complete")


def _turns_to_correct_diagnosis(
    session_events: list[dict], ground_truth: str
) -> int | None:
    """First turn_index at which ground_truth was the top-1 diagnosis. None if never."""
    gt = _normalize(ground_truth)
    for event in session_events:
        if event.get("event_type") != "turn_complete":
            continue
        doc_out = event.get("doctor_output", {})
        diff = doc_out.get("current_differential", [])
        if diff and _normalize(diff[0]["dx"]) == gt:
            return int(doc_out.get("turn_index", event.get("turn_index", 0)))
    return None


def _jaccard_overlap(
    doctor_diff: list[tuple[str, float]],
    gt_diff: list[tuple[str, float]],
) -> float:
    """Jaccard overlap between the disease-name sets of the two differentials."""
    doc_set = {_normalize(dx) for dx, _ in doctor_diff}
    gt_set = {_normalize(dx) for dx, _ in gt_diff}
    if not doc_set and not gt_set:
        return 1.0
    intersection = len(doc_set & gt_set)
    union = len(doc_set | gt_set)
    return intersection / union if union else 0.0


def evaluate_session(
    session_jsonl_path: str | Path,
    ddxplus_record: DDXPlusPatient,
    session_id: str | None = None,
) -> DDXPlusEvalResult:
    """
    Read a session JSONL log and compare it against the DDXPlus ground truth.

    No LLM calls — pure comparison logic.
    """
    path = Path(session_jsonl_path)
    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if session_id is None:
        # Infer from the first event
        session_id = events[0].get("session_id", path.stem) if events else path.stem

    final_diff = _top_differential(events)
    gt_pathology = ddxplus_record.ground_truth_pathology
    gt_norm = _normalize(gt_pathology)

    top1_correct = bool(
        final_diff and _normalize(final_diff[0][0]) == gt_norm
    )
    top3_contains = any(
        _normalize(dx) == gt_norm for dx, _ in final_diff[:3]
    )
    overlap = _jaccard_overlap(final_diff, ddxplus_record.ground_truth_differential)
    turns_correct = _turns_to_correct_diagnosis(events, gt_pathology)
    confidence = _final_confidence(events)
    terminated_conf = _terminated_on_confidence(events)
    total = _total_turns(events)
    in_pool = is_in_exemplar_pool(gt_pathology)

    return DDXPlusEvalResult(
        patient_id=ddxplus_record.patient_id,
        session_id=str(session_id),
        leading_diagnosis_correct=top1_correct,
        top3_contains_truth=top3_contains,
        differential_overlap=overlap,
        turns_to_correct_diagnosis=turns_correct,
        final_confidence=confidence,
        terminated_on_confidence=terminated_conf,
        total_turns=total,
        in_exemplar_pool=in_pool,
    )
