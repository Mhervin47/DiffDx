from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from loop2.ddxplus.evaluator import (
    _jaccard_overlap,
    _normalize,
    evaluate_session,
)
from loop2.ddxplus.schemas import DDXPlusPatient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_patient(pathology: str = "Pulmonary embolism") -> DDXPlusPatient:
    return DDXPlusPatient(
        patient_id="test_000001",
        age=45,
        sex="F",
        initial_evidence="Do you have a cough?",
        initial_evidence_code="E_201",
        symptoms={"cough": 1},
        antecedents={},
        ground_truth_pathology=pathology,
        ground_truth_differential=[
            (pathology, 0.50),
            ("Pneumonia", 0.25),
            ("GERD", 0.10),
        ],
    )


def _make_turn_event(
    session_id: str,
    turn_index: int,
    top_dx: str,
    top_prob: float = 0.60,
    confidence: float = 0.45,
    secondary_dxs: list[tuple[str, float]] | None = None,
) -> dict:
    diff = [{"dx": top_dx, "prob": top_prob}]
    if secondary_dxs:
        diff += [{"dx": dx, "prob": p} for dx, p in secondary_dxs]
    return {
        "schema_version": "0.5.0",
        "event_type": "turn_complete",
        "session_id": session_id,
        "turn_index": turn_index,
        "doctor_output": {
            "turn_index": turn_index,
            "current_differential": diff,
            "biggest_uncertainty": "test",
            "candidate_questions": ["q1"],
            "chosen_question": "q1",
            "rationale": "test",
            "confidence_to_stop": confidence,
            "should_stop": False,
            "safety_flags": [],
        },
        "patient_answer": "Yes.",
        "retrieved_exemplar_ids": [],
        "timestamp": "2026-01-01T00:00:00Z",
    }


def _make_session_end_event(
    session_id: str,
    reason: str,
    top_dx: str,
    top_prob: float = 0.75,
    confidence: float = 0.75,
    secondary_dxs: list[tuple[str, float]] | None = None,
) -> dict:
    diff = [{"dx": top_dx, "prob": top_prob}]
    if secondary_dxs:
        diff += [{"dx": dx, "prob": p} for dx, p in secondary_dxs]
    return {
        "schema_version": "0.5.0",
        "event_type": "session_end",
        "session_id": session_id,
        "termination_reason": reason,
        "doctor_output": {
            "turn_index": 5,
            "current_differential": diff,
            "biggest_uncertainty": "test",
            "candidate_questions": ["q1"],
            "chosen_question": "q1",
            "rationale": "test",
            "confidence_to_stop": confidence,
            "should_stop": True,
            "safety_flags": [],
        },
    }


def _write_session_jsonl(events: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for e in events:
        tmp.write(json.dumps(e) + "\n")
    tmp.flush()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("Pulmonary Embolism") == "pulmonary embolism"

    def test_strip_whitespace(self):
        assert _normalize("  GERD  ") == "gerd"

    def test_collapse_spaces(self):
        assert _normalize("Possible  NSTEMI") == "possible nstemi"


class TestJaccardOverlap:
    def test_identical_sets(self):
        a = [("PE", 0.5), ("Pneumonia", 0.3)]
        b = [("PE", 0.6), ("Pneumonia", 0.2)]
        assert _jaccard_overlap(a, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = [("PE", 0.5)]
        b = [("GERD", 0.5)]
        assert _jaccard_overlap(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = [("PE", 0.5), ("Pneumonia", 0.3), ("GERD", 0.2)]
        b = [("PE", 0.6), ("Bronchitis", 0.3)]
        # Intersection: {"pe"}, Union: {"pe","pneumonia","gerd","bronchitis"}
        assert _jaccard_overlap(a, b) == pytest.approx(1 / 4)

    def test_empty_both(self):
        assert _jaccard_overlap([], []) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Session-level tests (5 hand-constructed fake sessions)
# ---------------------------------------------------------------------------

class TestEvaluateSession:
    def test_case1_correct_top1_on_first_turn(self):
        """Doctor nails the diagnosis on turn 0 and confidence terminates."""
        patient = _make_patient("Pulmonary embolism")
        events = [
            _make_turn_event("sess1", 0, "Pulmonary Embolism", confidence=0.55),
            _make_session_end_event("sess1", "confidence_threshold", "Pulmonary Embolism"),
        ]
        path = _write_session_jsonl(events)
        result = evaluate_session(path, patient, session_id="sess1")

        assert result.leading_diagnosis_correct is True
        assert result.top3_contains_truth is True
        assert result.turns_to_correct_diagnosis == 0
        assert result.terminated_on_confidence is True
        assert result.total_turns == 1
        assert result.in_exemplar_pool is True

    def test_case2_correct_in_top3_not_top1(self):
        """Doctor's top-1 is wrong but ground truth is in top-3."""
        patient = _make_patient("Pulmonary embolism")
        events = [
            _make_turn_event("sess2", 0, "Pneumonia", confidence=0.40,
                             secondary_dxs=[("Pulmonary Embolism", 0.30), ("GERD", 0.10)]),
            _make_session_end_event("sess2", "max_turns", "Pneumonia",
                                    secondary_dxs=[("Pulmonary Embolism", 0.28), ("GERD", 0.10)]),
        ]
        path = _write_session_jsonl(events)
        result = evaluate_session(path, patient, session_id="sess2")

        assert result.leading_diagnosis_correct is False
        assert result.top3_contains_truth is True
        assert result.turns_to_correct_diagnosis is None  # never reached top-1

    def test_case3_ground_truth_never_mentioned(self):
        """Doctor never considers the ground truth at all — zero Jaccard overlap."""
        # GT differential: [Pulmonary embolism, Pneumonia, GERD]
        # Doctor only considers Influenza + Bronchitis → no overlap with GT set
        patient = _make_patient("Pulmonary embolism")
        events = [
            _make_turn_event("sess3", 0, "Influenza", confidence=0.30,
                             secondary_dxs=[("Bronchitis", 0.20)]),
            _make_turn_event("sess3", 1, "Influenza", confidence=0.35,
                             secondary_dxs=[("Bronchitis", 0.18)]),
            _make_session_end_event("sess3", "max_turns", "Influenza",
                                    secondary_dxs=[("Bronchitis", 0.18)]),
        ]
        path = _write_session_jsonl(events)
        result = evaluate_session(path, patient, session_id="sess3")

        assert result.leading_diagnosis_correct is False
        assert result.top3_contains_truth is False
        assert result.differential_overlap == pytest.approx(0.0)
        assert result.turns_to_correct_diagnosis is None
        assert result.total_turns == 2

    def test_case4_turns_to_correct_diagnosis_mid_session(self):
        """Ground truth first appears as top-1 at turn 2 out of 4."""
        patient = _make_patient("Pericarditis")
        events = [
            _make_turn_event("sess4", 0, "GERD", confidence=0.30),
            _make_turn_event("sess4", 1, "GERD", confidence=0.32),
            _make_turn_event("sess4", 2, "Pericarditis", confidence=0.48),
            _make_turn_event("sess4", 3, "Pericarditis", confidence=0.55),
            _make_session_end_event("sess4", "confidence_threshold", "Pericarditis"),
        ]
        path = _write_session_jsonl(events)
        result = evaluate_session(path, patient, session_id="sess4")

        assert result.leading_diagnosis_correct is True
        assert result.turns_to_correct_diagnosis == 2
        assert result.total_turns == 4
        assert result.terminated_on_confidence is True

    def test_case5_jaccard_overlap_partial(self):
        """Doctor's differential partially overlaps with ground truth differential."""
        patient = _make_patient("SLE")
        # GT differential: SLE, Pneumonia, GERD
        # Doctor final: Rheumatoid Arthritis, Pneumonia, GERD (SLE missing)
        events = [
            _make_session_end_event(
                "sess5", "max_turns", "Rheumatoid Arthritis", top_prob=0.40,
                secondary_dxs=[("Pneumonia", 0.30), ("GERD", 0.20)],
            ),
        ]
        path = _write_session_jsonl(events)
        result = evaluate_session(path, patient, session_id="sess5")

        assert result.leading_diagnosis_correct is False
        assert result.top3_contains_truth is False
        # Doctor has: {rheumatoid arthritis, pneumonia, gerd}
        # GT has: {sle, pneumonia, gerd}
        # Intersection: {pneumonia, gerd} = 2
        # Union: {rheumatoid arthritis, pneumonia, gerd, sle} = 4
        assert result.differential_overlap == pytest.approx(2 / 4)
        assert result.in_exemplar_pool is True  # SLE is in pool
