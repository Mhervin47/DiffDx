"""Phase 0 — validate every Pydantic model in schemas.py."""
import pytest
from pydantic import ValidationError

from loop1.schemas import (
    Demographics,
    DiagnosisEntry,
    DoctorTurnOutput,
    Exemplar,
    FinalRecord,
    History,
    ModelMetadata,
    PatientProfile,
    Symptom,
    TurnRecord,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_profile(session_id: str = "sess-001") -> PatientProfile:
    return PatientProfile(
        session_id=session_id,
        demographics=Demographics(age=34, sex="F"),
        chief_complaint="headache",
        symptoms=[Symptom(name="headache", onset="3 days ago", severity="moderate")],
        history=History(medical=["hypertension"], medications=["lisinopril"]),
        ruled_out=["cluster headache"],
        ruled_in=[],
        free_notes="Patient appears anxious.",
    )


def make_doctor_turn(turn_index: int = 0) -> DoctorTurnOutput:
    return DoctorTurnOutput(
        turn_index=turn_index,
        current_differential=[
            DiagnosisEntry(dx="migraine", prob=0.55),
            DiagnosisEntry(dx="tension headache", prob=0.25),
        ],
        biggest_uncertainty="No aura history yet",
        candidate_questions=["Do you see flashing lights?", "Is it one-sided?"],
        chosen_question="Do you see any visual disturbances before the headache starts?",
        rationale="Aura presence would strongly support migraine over tension headache.",
        confidence_to_stop=0.32,
        should_stop=False,
        safety_flags=[],
    )


# ── Symptom ───────────────────────────────────────────────────────────────────

def test_symptom_minimal():
    s = Symptom(name="fever")
    assert s.name == "fever"
    assert s.onset == ""
    assert s.severity == ""


def test_symptom_full():
    s = Symptom(name="cough", onset="1 week", severity="mild", notes="dry")
    assert s.notes == "dry"


# ── Demographics ──────────────────────────────────────────────────────────────

def test_demographics_defaults():
    d = Demographics()
    assert d.age is None
    assert d.sex is None
    assert d.other == {}


def test_demographics_with_other():
    d = Demographics(age=45, sex="M", other={"ethnicity": "Asian"})
    assert d.other["ethnicity"] == "Asian"


# ── History ───────────────────────────────────────────────────────────────────

def test_history_all_lists_default_empty():
    h = History()
    for field in ("medical", "medications", "allergies", "family", "social"):
        assert getattr(h, field) == []


# ── PatientProfile ────────────────────────────────────────────────────────────

def test_patient_profile_round_trip():
    profile = make_profile()
    data = profile.model_dump()
    restored = PatientProfile(**data)
    assert restored.session_id == profile.session_id
    assert restored.symptoms[0].name == "headache"


def test_patient_profile_requires_session_id():
    with pytest.raises(ValidationError):
        PatientProfile()  # type: ignore[call-arg]


# ── DiagnosisEntry ────────────────────────────────────────────────────────────

def test_diagnosis_prob_bounds():
    DiagnosisEntry(dx="flu", prob=0.0)
    DiagnosisEntry(dx="flu", prob=1.0)


def test_diagnosis_prob_out_of_bounds():
    with pytest.raises(ValidationError):
        DiagnosisEntry(dx="flu", prob=1.1)
    with pytest.raises(ValidationError):
        DiagnosisEntry(dx="flu", prob=-0.1)


# ── DoctorTurnOutput ──────────────────────────────────────────────────────────

def test_doctor_turn_valid():
    turn = make_doctor_turn()
    assert turn.should_stop is False
    assert len(turn.candidate_questions) >= 1


def test_doctor_turn_confidence_bounds():
    with pytest.raises(ValidationError):
        DoctorTurnOutput(
            turn_index=0,
            current_differential=[DiagnosisEntry(dx="flu", prob=0.9)],
            biggest_uncertainty="x",
            candidate_questions=["q"],
            chosen_question="q",
            rationale="r",
            confidence_to_stop=1.5,  # invalid
            should_stop=False,
        )


def test_doctor_turn_negative_index_rejected():
    with pytest.raises(ValidationError):
        DoctorTurnOutput(
            turn_index=-1,
            current_differential=[],
            biggest_uncertainty="x",
            candidate_questions=["q"],
            chosen_question="q",
            rationale="r",
            confidence_to_stop=0.5,
            should_stop=False,
        )


def test_doctor_turn_safety_flags_default_empty():
    turn = make_doctor_turn()
    assert turn.safety_flags == []


# ── TurnRecord ────────────────────────────────────────────────────────────────

def test_turn_record_valid():
    rec = TurnRecord(
        turn_index=0,
        doctor_output=make_doctor_turn(),
        patient_answer="Yes, I do see zigzag lines.",
        retrieved_exemplar_ids=["ex_001"],
        timestamp="2026-05-22T10:00:00+00:00",
    )
    assert rec.patient_answer.startswith("Yes")


# ── FinalRecord ───────────────────────────────────────────────────────────────

def test_final_record_valid():
    profile = make_profile()
    turn = TurnRecord(
        turn_index=0,
        doctor_output=make_doctor_turn(),
        patient_answer="Three days.",
        retrieved_exemplar_ids=[],
        timestamp="2026-05-22T10:00:00+00:00",
    )
    record = FinalRecord(
        session_id="sess-001",
        started_at="2026-05-22T10:00:00+00:00",
        ended_at="2026-05-22T10:15:00+00:00",
        termination_reason="confidence_threshold",
        final_profile=profile,
        final_differential=[DiagnosisEntry(dx="migraine", prob=0.78)],
        primary_diagnosis="migraine",
        turn_history=[turn],
        model_metadata=ModelMetadata(
            model_name="groq/llama-3.3-70b-versatile",
            model_version="1.0",
            prompt_template_version="v0.1",
        ),
    )
    assert record.termination_reason == "confidence_threshold"


def test_final_record_invalid_termination_reason():
    profile = make_profile()
    with pytest.raises(ValidationError):
        FinalRecord(
            session_id="sess-001",
            started_at="2026-05-22T10:00:00+00:00",
            ended_at="2026-05-22T10:15:00+00:00",
            termination_reason="timeout",  # not in Literal
            final_profile=profile,
            final_differential=[],
            primary_diagnosis="unknown",
            turn_history=[],
            model_metadata=ModelMetadata(
                model_name="x", model_version="1", prompt_template_version="v0.1"
            ),
        )


# ── Exemplar ──────────────────────────────────────────────────────────────────

def test_exemplar_frozen_default():
    ex = Exemplar(
        exemplar_id="ex_001",
        tags=["cardiac"],
        context_summary="65yo M with chest pain",
        good_next_question="Does the pain radiate to your arm?",
        rationale="Radiation pattern differentiates ACS from MSK.",
    )
    assert ex.frozen is True


def test_exemplar_can_be_unfrozen():
    ex = Exemplar(
        exemplar_id="ex_999",
        context_summary="test",
        good_next_question="q",
        rationale="r",
        frozen=False,
    )
    assert ex.frozen is False
