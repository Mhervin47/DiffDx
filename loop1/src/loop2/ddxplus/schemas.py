from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.7.0"


class DDXPlusPatient(BaseModel):
    patient_id: str
    age: int
    sex: str
    initial_evidence: str                              # decoded human-readable chief complaint
    initial_evidence_code: str                         # raw E_ code (e.g. "E_201")
    symptoms: dict[str, Any]                           # question_en -> decoded value
    antecedents: dict[str, Any]                        # question_en -> decoded value
    ground_truth_pathology: str
    ground_truth_differential: list[tuple[str, float]]


class DDXPlusEvalResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    patient_id: str
    session_id: str
    leading_diagnosis_correct: bool
    top3_contains_truth: bool
    differential_overlap: float                        # Jaccard overlap with GT differential
    turns_to_correct_diagnosis: int | None             # None if never reached
    final_confidence: float
    terminated_on_confidence: bool
    total_turns: int
    in_exemplar_pool: bool                             # was disease covered by an exemplar?
