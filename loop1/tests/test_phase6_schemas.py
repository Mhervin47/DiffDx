"""Phase 6 — ClosingTurn, SafetyEvent, schema_version field tests."""
import pytest
from pydantic import ValidationError

from loop1.schemas import (
    SCHEMA_VERSION,
    ClosingTurn,
    DiagnosisEntry,
    FinalRecord,
    ModelMetadata,
    PatientProfile,
    SafetyEvent,
    Symptom,
    TurnRecord,
    Demographics,
    History,
    DoctorTurnOutput,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _profile() -> PatientProfile:
    return PatientProfile(
        session_id="sess-p6",
        demographics=Demographics(age=30, sex="M"),
        chief_complaint="fever",
        symptoms=[Symptom(name="fever", onset="3 days ago")],
        history=History(),
    )


def _doctor_turn() -> DoctorTurnOutput:
    return DoctorTurnOutput(
        turn_index=0,
        current_differential=[DiagnosisEntry(dx="flu", prob=0.7)],
        biggest_uncertainty="exposure",
        candidate_questions=["Have you traveled recently?"],
        chosen_question="Have you traveled recently?",
        rationale="Travel raises infectious disease probability.",
        confidence_to_stop=0.3,
        should_stop=False,
    )


# ── schema_version constant ───────────────────────────────────────────────────

def test_schema_version_constant():
    assert SCHEMA_VERSION == "0.5.0"


# ── TurnRecord schema_version ─────────────────────────────────────────────────

def test_turn_record_has_schema_version():
    rec = TurnRecord(
        turn_index=0,
        doctor_output=_doctor_turn(),
        patient_answer="No travel.",
        retrieved_exemplar_ids=[],
        timestamp="2026-06-01T10:00:00+00:00",
    )
    assert rec.schema_version == SCHEMA_VERSION


def test_turn_record_schema_version_in_dump():
    rec = TurnRecord(
        turn_index=0,
        doctor_output=_doctor_turn(),
        patient_answer="No.",
        retrieved_exemplar_ids=[],
        timestamp="2026-06-01T10:00:00+00:00",
    )
    dumped = rec.model_dump()
    assert "schema_version" in dumped
    assert dumped["schema_version"] == SCHEMA_VERSION


# ── FinalRecord schema_version ────────────────────────────────────────────────

def test_final_record_has_schema_version():
    record = FinalRecord(
        session_id="sess-p6",
        started_at="2026-06-01T10:00:00+00:00",
        ended_at="2026-06-01T10:15:00+00:00",
        termination_reason="confidence_threshold",
        final_profile=_profile(),
        final_differential=[DiagnosisEntry(dx="flu", prob=0.8)],
        primary_diagnosis="Flu",
        turn_history=[],
        model_metadata=ModelMetadata(
            model_name="groq/test",
            model_version="0",
            prompt_template_version="v0.4",
        ),
    )
    assert record.schema_version == SCHEMA_VERSION


def test_final_record_closing_turn_defaults_none():
    record = FinalRecord(
        session_id="sess-p6",
        started_at="2026-06-01T10:00:00+00:00",
        ended_at="2026-06-01T10:15:00+00:00",
        termination_reason="max_turns",
        final_profile=_profile(),
        final_differential=[],
        primary_diagnosis="undetermined",
        turn_history=[],
        model_metadata=ModelMetadata(
            model_name="x", model_version="0", prompt_template_version="v0.4"
        ),
    )
    assert record.closing_turn is None


# ── ClosingTurn ───────────────────────────────────────────────────────────────

def test_closing_turn_valid():
    ct = ClosingTurn(
        leading_diagnosis="Most likely you have influenza.",
        differential_summary="Influenza (70%), COVID-19 (20%), common cold (10%).",
        recommended_next_steps=["Rest at home", "See GP if fever lasts more than 5 days"],
        unresolved_patient_questions=["Will antibiotics help?"],
        generated_by="groq/llama-3.3-70b-versatile",
    )
    assert ct.schema_version == SCHEMA_VERSION
    assert len(ct.recommended_next_steps) == 2


def test_closing_turn_empty_lists():
    ct = ClosingTurn(
        leading_diagnosis="Unclear.",
        differential_summary="Insufficient information.",
        recommended_next_steps=[],
        unresolved_patient_questions=[],
        generated_by="groq/test",
    )
    assert ct.unresolved_patient_questions == []


def test_final_record_with_closing_turn():
    ct = ClosingTurn(
        leading_diagnosis="Most likely influenza.",
        differential_summary="Influenza (70%).",
        recommended_next_steps=["Stay hydrated"],
        unresolved_patient_questions=[],
        generated_by="groq/test",
    )
    record = FinalRecord(
        session_id="sess-p6",
        started_at="2026-06-01T10:00:00+00:00",
        ended_at="2026-06-01T10:15:00+00:00",
        termination_reason="confidence_threshold",
        final_profile=_profile(),
        final_differential=[DiagnosisEntry(dx="flu", prob=0.7)],
        primary_diagnosis="Flu",
        turn_history=[],
        model_metadata=ModelMetadata(
            model_name="groq/test", model_version="0", prompt_template_version="v0.4"
        ),
        closing_turn=ct,
    )
    assert record.closing_turn is not None
    assert record.closing_turn.leading_diagnosis == "Most likely influenza."


def test_final_record_closing_turn_in_dump():
    ct = ClosingTurn(
        leading_diagnosis="Most likely influenza.",
        differential_summary="Influenza (70%).",
        recommended_next_steps=[],
        unresolved_patient_questions=[],
        generated_by="groq/test",
    )
    record = FinalRecord(
        session_id="sess-p6",
        started_at="2026-06-01T10:00:00+00:00",
        ended_at="2026-06-01T10:15:00+00:00",
        termination_reason="confidence_threshold",
        final_profile=_profile(),
        final_differential=[],
        primary_diagnosis="Flu",
        turn_history=[],
        model_metadata=ModelMetadata(
            model_name="x", model_version="0", prompt_template_version="v0.4"
        ),
        closing_turn=ct,
    )
    dumped = record.model_dump()
    assert "closing_turn" in dumped
    assert dumped["closing_turn"]["leading_diagnosis"] == "Most likely influenza."


# ── SafetyEvent ───────────────────────────────────────────────────────────────

def test_safety_event_valid():
    ev = SafetyEvent(
        session_id="sess-p6",
        turn=2,
        matched_phrase="chest pain",
        patient_input="I have severe chest pain and can't breathe",
    )
    assert ev.schema_version == SCHEMA_VERSION
    assert ev.action == "session_terminated"
    assert ev.matched_phrase == "chest pain"


def test_safety_event_in_dump():
    ev = SafetyEvent(
        session_id="sess-p6",
        turn=0,
        matched_phrase="suicidal",
        patient_input="I'm feeling suicidal",
    )
    dumped = ev.model_dump()
    assert dumped["schema_version"] == SCHEMA_VERSION
    assert dumped["action"] == "session_terminated"
