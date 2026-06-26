from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from loop3.routing.ambiguity_handler import get_routing_options, is_ambiguous
from loop3.routing.map_loader import load_disease_specialty_map, lookup_disease
from loop3.routing.router import route
from loop3.routing.routing_schema import (
    AppointmentType,
    RoutingDecision,
    RoutingOption,
    UrgencyLevel,
)
from loop3.runners.phase9_routing_runner import run_routing_from_session
from loop3.routing.urgency_classifier import classify_urgency


@pytest.fixture(scope="module")
def specialty_map():
    return load_disease_specialty_map()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestRoutingSchema:
    def test_schema_version(self):
        decision = RoutingDecision(
            session_id="s1",
            is_ambiguous=False,
            primary_routing=RoutingOption(
                specialty="Cardiology",
                urgency=UrgencyLevel.EMERGENCY,
                reasoning="Test.",
                appointment_type=AppointmentType.EMERGENCY,
                confidence_weight=0.8,
            ),
            final_differential=[("Pulmonary embolism", 0.8)],
            final_confidence=0.7,
        )
        assert decision.schema_version == "0.9.0"
        assert decision.routing_model_version == "disease_specialty_map_v1"

    def test_serialization(self):
        option = RoutingOption(
            specialty="Neurology",
            urgency=UrgencyLevel.URGENT,
            reasoning="Needs neuro.",
            appointment_type=AppointmentType.URGENT,
            confidence_weight=0.6,
        )
        data = option.model_dump()
        assert data["urgency"] == "urgent"
        assert data["appointment_type"] == "urgent_appointment_within_48h"

    def test_confidence_weight_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RoutingOption(
                specialty="X", urgency=UrgencyLevel.ROUTINE,
                reasoning="x", appointment_type=AppointmentType.ROUTINE,
                confidence_weight=1.5,
            )


# ---------------------------------------------------------------------------
# Map loader tests
# ---------------------------------------------------------------------------

class TestMapLoader:
    def test_loads_without_error(self, specialty_map):
        assert len(specialty_map) > 0

    def test_all_entries_have_required_keys(self, specialty_map):
        required = {"specialty", "urgency", "appointment_type", "reasoning"}
        for disease, entry in specialty_map.items():
            assert required <= set(entry.keys()), f"{disease!r} missing keys"

    def test_no_missing_urgency(self, specialty_map):
        valid_urgencies = {"emergency", "urgent", "routine"}
        for disease, entry in specialty_map.items():
            assert entry["urgency"] in valid_urgencies, f"{disease!r}: bad urgency"

    def test_no_missing_appointment_type(self, specialty_map):
        valid_types = {
            "emergency_visit", "same_day_appointment",
            "urgent_appointment_within_48h", "routine_appointment",
            "telehealth_consultation",
        }
        for disease, entry in specialty_map.items():
            assert entry["appointment_type"] in valid_types, f"{disease!r}: bad appt type"

    def test_lookup_case_insensitive(self, specialty_map):
        assert lookup_disease("pulmonary embolism", specialty_map) is not None
        assert lookup_disease("PULMONARY EMBOLISM", specialty_map) is not None

    def test_lookup_unknown_returns_none(self, specialty_map):
        assert lookup_disease("Totally Unknown Disease XYZ", specialty_map) is None

    def test_sle_key_correct(self, specialty_map):
        assert lookup_disease("SLE", specialty_map) is not None
        assert lookup_disease("SLE (Systemic Lupus Erythematosus)", specialty_map) is None

    def test_acute_dystonic_reactions_key_correct(self, specialty_map):
        assert lookup_disease("Acute dystonic reactions", specialty_map) is not None
        assert lookup_disease("Acute dystonia", specialty_map) is None

    def test_no_duplicate_entries(self, specialty_map):
        assert len(specialty_map) == len(set(specialty_map.keys()))


# ---------------------------------------------------------------------------
# Urgency classifier tests
# ---------------------------------------------------------------------------

class TestUrgencyClassifier:
    def test_emergency_stays_emergency_at_any_confidence(self, specialty_map):
        for conf in [0.1, 0.4, 0.9]:
            assert classify_urgency("Pulmonary embolism", conf, specialty_map) == UrgencyLevel.EMERGENCY

    def test_routine_at_high_confidence_stays_routine(self, specialty_map):
        assert classify_urgency("Influenza", 0.7, specialty_map) == UrgencyLevel.ROUTINE

    def test_routine_at_low_confidence_bumps_to_urgent(self, specialty_map):
        assert classify_urgency("Influenza", 0.3, specialty_map) == UrgencyLevel.URGENT

    def test_urgent_at_low_confidence_stays_urgent(self, specialty_map):
        assert classify_urgency("Pericarditis", 0.2, specialty_map) == UrgencyLevel.URGENT

    def test_unknown_disease_returns_routine(self, specialty_map):
        result = classify_urgency("Completely Unknown Disease", 0.8, specialty_map)
        assert result == UrgencyLevel.ROUTINE

    def test_ten_hand_checked_diseases(self, specialty_map):
        cases = [
            ("Possible NSTEMI / STEMI", 0.8, UrgencyLevel.EMERGENCY),
            ("Guillain-Barré syndrome", 0.7, UrgencyLevel.EMERGENCY),
            ("Anaphylaxis", 0.6, UrgencyLevel.EMERGENCY),
            ("Tuberculosis", 0.5, UrgencyLevel.URGENT),
            ("Atrial fibrillation", 0.6, UrgencyLevel.URGENT),
            ("GERD", 0.8, UrgencyLevel.ROUTINE),
            ("Influenza", 0.7, UrgencyLevel.ROUTINE),
            ("Bronchitis", 0.6, UrgencyLevel.ROUTINE),
            ("GERD", 0.2, UrgencyLevel.URGENT),   # low confidence bump
            ("SLE", 0.5, UrgencyLevel.URGENT),
        ]
        for disease, conf, expected in cases:
            result = classify_urgency(disease, conf, specialty_map)
            assert result == expected, f"{disease} at conf={conf}: expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Ambiguity handler tests
# ---------------------------------------------------------------------------

class TestAmbiguityHandler:
    def test_unambiguous_single_specialty(self, specialty_map):
        # All PE → Pulmonology
        diff = [("Pulmonary embolism", 0.7), ("Pneumonia", 0.2), ("Bronchitis", 0.1)]
        # Pulm: 0.7+0.2+0.1=1.0 → only one specialty → not ambiguous
        assert is_ambiguous(diff, specialty_map) is False

    def test_unambiguous_dominant_specialty(self, specialty_map):
        # Heavy Cardiology weight vs small Pulmonology — BUT both are emergency specialties,
        # so the emergency-surfacing rule marks this as ambiguous regardless of probability ratio.
        diff = [("Possible NSTEMI / STEMI", 0.7), ("Stable angina", 0.2), ("Pulmonary embolism", 0.1)]
        assert is_ambiguous(diff, specialty_map) is True

    def test_unambiguous_dominant_non_emergency(self, specialty_map):
        # Dominant Cardiology (urgent) vs small Pulmonology (urgent) — no emergency rule, not ambiguous
        diff = [("Pericarditis", 0.8), ("Stable angina", 0.1), ("Bronchitis", 0.1)]
        # Cardiology: 0.9, Internal Medicine/Pulmonology small → probability dominates, no emergency rule
        assert is_ambiguous(diff, specialty_map) is False

    def test_ambiguous_split_differential(self, specialty_map):
        # Even split between Cardiology and Pulmonology
        diff = [("Possible NSTEMI / STEMI", 0.45), ("Pulmonary embolism", 0.45), ("GERD", 0.1)]
        assert is_ambiguous(diff, specialty_map) is True

    def test_get_routing_options_unambiguous_returns_one(self, specialty_map):
        diff = [("Pulmonary embolism", 0.8), ("Pneumonia", 0.2)]
        options = get_routing_options(diff, specialty_map, final_confidence=0.7)
        assert len(options) == 1
        assert options[0].specialty == "Pulmonology"

    def test_get_routing_options_ambiguous_returns_multiple(self, specialty_map):
        diff = [("Possible NSTEMI / STEMI", 0.45), ("Pulmonary embolism", 0.45), ("GERD", 0.1)]
        options = get_routing_options(diff, specialty_map, final_confidence=0.5)
        assert len(options) >= 2

    def test_get_routing_options_empty_diff_returns_fallback(self, specialty_map):
        options = get_routing_options([], specialty_map, final_confidence=0.3)
        assert len(options) == 1
        assert options[0].specialty == "Internal Medicine"


# ---------------------------------------------------------------------------
# Core router tests
# ---------------------------------------------------------------------------

class TestRouter:
    def test_pe_routes_to_pulmonology_emergency(self):
        decision = route(
            session_id="test-1",
            final_differential=[("Pulmonary embolism", 0.7), ("Pneumonia", 0.3)],
            final_confidence=0.65,
        )
        assert decision.primary_routing.specialty == "Pulmonology"
        assert decision.primary_routing.urgency == UrgencyLevel.EMERGENCY
        assert decision.schema_version == "0.9.0"

    def test_emergency_surfaces_as_primary_over_higher_probability_routine(self):
        # GERD (routine, 0.6) vs Guillain-Barré (emergency, 0.4)
        # Emergency must surface as primary despite lower probability
        decision = route(
            session_id="test-2",
            final_differential=[("GERD", 0.6), ("Guillain-Barré syndrome", 0.4)],
            final_confidence=0.55,
        )
        assert decision.primary_routing.urgency == UrgencyLevel.EMERGENCY
        assert "Neurology" in decision.primary_routing.specialty

    def test_unknown_disease_falls_back_gracefully(self):
        decision = route(
            session_id="test-3",
            final_differential=[("Completely Unknown Disease XYZ", 1.0)],
            final_confidence=0.5,
        )
        assert decision.primary_routing.specialty == "Internal Medicine"
        assert decision.primary_routing.urgency == UrgencyLevel.ROUTINE

    def test_ambiguous_flag_set_correctly(self):
        ambiguous = route(
            session_id="test-4a",
            final_differential=[("Possible NSTEMI / STEMI", 0.45), ("Pulmonary embolism", 0.45), ("GERD", 0.1)],
            final_confidence=0.5,
        )
        assert ambiguous.is_ambiguous is True

        unambiguous = route(
            session_id="test-4b",
            final_differential=[("Pulmonary embolism", 0.9), ("GERD", 0.1)],
            final_confidence=0.8,
        )
        assert unambiguous.is_ambiguous is False

    def test_patient_id_passed_through(self):
        decision = route(
            session_id="s1",
            final_differential=[("Influenza", 1.0)],
            final_confidence=0.6,
            patient_id="test_000042",
        )
        assert decision.patient_id == "test_000042"


# ---------------------------------------------------------------------------
# Phase 9 runner test — case_01.jsonl (Phase 6 demo session)
# ---------------------------------------------------------------------------

class TestPhase9Runner:
    _FIXTURE = Path(__file__).parent / "fixtures" / "critic_calibration" / "case_01_session.jsonl"

    def test_runs_on_case_01(self):
        decision = run_routing_from_session(self._FIXTURE)
        # case_01 final differential: ACS=0.55, PE=0.30
        # Both Cardiology (ACS) and Pulmonology (PE) are emergency → either is valid primary
        assert decision.primary_routing.urgency == UrgencyLevel.EMERGENCY
        assert decision.session_id == "d7f3a9b2-6c8e-4e93-b4a0-1a4f8f0f2101"
        assert decision.schema_version == "0.9.0"

    def test_pe_in_differential_triggers_emergency_routing(self):
        # Even though ACS ranks above PE in case_01, PE is in the differential
        # The router must produce an emergency routing (both ACS and PE are emergency)
        decision = run_routing_from_session(self._FIXTURE)
        assert decision.primary_routing.appointment_type == AppointmentType.EMERGENCY

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            run_routing_from_session("/nonexistent/path.jsonl")

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = Path(f.name)
        with pytest.raises(ValueError, match="empty"):
            run_routing_from_session(tmp)
