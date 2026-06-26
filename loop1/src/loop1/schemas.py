from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "0.5.0"


class Symptom(BaseModel):
    name: str
    onset: str = ""
    severity: str = ""
    notes: str = ""


class Demographics(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    other: dict[str, Any] = Field(default_factory=dict)


class History(BaseModel):
    medical: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    family: list[str] = Field(default_factory=list)
    social: list[str] = Field(default_factory=list)


class PatientProfile(BaseModel):
    session_id: str
    demographics: Demographics = Field(default_factory=Demographics)
    chief_complaint: str = ""
    symptoms: list[Symptom] = Field(default_factory=list)
    history: History = Field(default_factory=History)
    ruled_out: list[str] = Field(default_factory=list)
    ruled_in: list[str] = Field(default_factory=list)
    free_notes: str = ""
    running_summary: str = ""


class SymptomUpdate(BaseModel):
    name: str
    onset: str = ""
    severity: str = ""
    notes: str = ""


class HistoryAdditions(BaseModel):
    medical: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    family: list[str] = Field(default_factory=list)
    social: list[str] = Field(default_factory=list)


class ProfileDelta(BaseModel):
    new_symptoms: list[Symptom] = Field(default_factory=list)
    symptom_updates: list[SymptomUpdate] = Field(default_factory=list)
    history_additions: HistoryAdditions = Field(default_factory=HistoryAdditions)
    append_ruled_out: list[str] = Field(default_factory=list)
    append_ruled_in: list[str] = Field(default_factory=list)
    free_notes_append: str = ""


class DiagnosisEntry(BaseModel):
    dx: str
    prob: float = Field(ge=0.0, le=1.0)

    @field_validator("dx")
    @classmethod
    def normalize_dx(cls, v: str) -> str:
        return v.title()


class DoctorTurnOutput(BaseModel):
    turn_index: int = Field(ge=0)
    current_differential: list[DiagnosisEntry]
    biggest_uncertainty: str
    candidate_questions: list[str] = Field(min_length=1)
    chosen_question: str
    rationale: str
    confidence_to_stop: float = Field(ge=0.0, le=1.0)
    should_stop: bool
    safety_flags: list[str] = Field(default_factory=list)


class TurnRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    turn_index: int = Field(ge=0)
    doctor_output: DoctorTurnOutput
    patient_answer: str
    retrieved_exemplar_ids: list[str] = Field(default_factory=list)
    timestamp: str


class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    prompt_template_version: str


class ClosingTurn(BaseModel):
    schema_version: str = SCHEMA_VERSION
    leading_diagnosis: str
    differential_summary: str
    recommended_next_steps: list[str]
    unresolved_patient_questions: list[str]
    generated_by: str


class SafetyEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    turn: int
    matched_phrase: str
    patient_input: str
    action: str = "session_terminated"


class FinalRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    started_at: str
    ended_at: str
    termination_reason: Literal[
        "confidence_threshold", "max_turns", "safety_stop", "user_quit"
    ]
    final_profile: PatientProfile
    final_differential: list[DiagnosisEntry]
    primary_diagnosis: str
    turn_history: list[TurnRecord]
    model_metadata: ModelMetadata
    closing_turn: Optional[ClosingTurn] = None


class Exemplar(BaseModel):
    exemplar_id: str
    tags: list[str] = Field(default_factory=list)
    context_summary: str
    good_next_question: str
    rationale: str
    frozen: bool = True
    embedding: Optional[list[float]] = None  # Phase 5 hook: populated by retrieval module
