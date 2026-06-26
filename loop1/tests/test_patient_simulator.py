from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from loop2.ddxplus.patient_simulator import (
    SimulatedPatient,
    _check_leakage,
    _leakage_tokens,
)
from loop2.ddxplus.schemas import DDXPlusPatient


def _make_patient(pathology: str = "Pulmonary embolism") -> DDXPlusPatient:
    return DDXPlusPatient(
        patient_id="test_000001",
        age=45,
        sex="F",
        initial_evidence="Do you have a cough?",
        initial_evidence_code="E_201",
        symptoms={
            "Do you have a cough?": 1,
            "Do you feel short of breath?": 1,
            "Do you have chest pain?": 1,
            "Do you have leg swelling?": 1,
        },
        antecedents={
            "Did you recently travel?": 1,
            "Do you smoke?": 0,
        },
        ground_truth_pathology=pathology,
        ground_truth_differential=[
            ("Pulmonary embolism", 0.45),
            ("Pneumonia", 0.20),
            ("GERD", 0.10),
        ],
    )


class TestLeakageDetection:
    def test_no_leakage_clean_response(self):
        assert _check_leakage("I have a cough and shortness of breath.", "Pulmonary embolism") is False

    def test_leakage_exact_word(self):
        assert _check_leakage("Could this be a pulmonary embolism?", "Pulmonary embolism") is True

    def test_leakage_case_insensitive(self):
        assert _check_leakage("PULMONARY sounds scary.", "Pulmonary embolism") is True

    def test_leakage_tokens_filters_short_words(self):
        # "SLE" is 3 chars, should be filtered out
        tokens = _leakage_tokens("SLE")
        assert tokens == []

    def test_leakage_tokens_single_long_word(self):
        tokens = _leakage_tokens("Pericarditis")
        assert "pericarditis" in tokens

    def test_leakage_tokens_multi_word(self):
        tokens = _leakage_tokens("Pulmonary embolism")
        assert "pulmonary" in tokens
        assert "embolism" in tokens

    def test_leakage_short_pathology_name_is_still_flagged(self):
        # "GERD" is 4 chars (> 3), so it IS in the token list and leakage IS flagged
        assert _check_leakage("I have been diagnosed with gerd before.", "GERD") is True

    def test_leakage_very_short_pathology_not_blocked(self):
        # Pathology names shorter than 4 chars (e.g. "HIV") produce no tokens → not flagged
        tokens = _leakage_tokens("HIV")
        assert tokens == []
        assert _check_leakage("I had HIV exposure.", "HIV") is False


class TestSimulatedPatientUnit:
    def test_render_history_empty(self):
        patient = _make_patient()
        sim = SimulatedPatient(patient)
        assert "(none yet)" in sim._render_history()

    def test_render_history_with_turns(self):
        patient = _make_patient()
        sim = SimulatedPatient(patient)
        sim._conversation_history.append(("Do you have a fever?", "Yes, mild one."))
        history = sim._render_history()
        assert "Doctor: Do you have a fever?" in history
        assert "Patient: Yes, mild one." in history

    def test_build_patient_record_no_ground_truth(self):
        from loop2.ddxplus.patient_simulator import _build_patient_record_json
        import json
        patient = _make_patient()
        record_json = _build_patient_record_json(patient)
        data = json.loads(record_json)
        assert "ground_truth_pathology" not in data
        assert "ground_truth_differential" not in data
        assert "symptoms" in data
        assert "antecedents" in data

    def test_update_disclosed_tracks_mentioned_symptoms(self):
        patient = _make_patient()
        sim = SimulatedPatient(patient)
        sim._update_disclosed("Yes, I have some cough and shortness of breath.")
        assert any("cough" in s.lower() for s in sim.disclosed_symptoms)

    @patch("loop2.ddxplus.patient_simulator.call_llm")
    def test_initial_complaint_no_leakage(self, mock_llm):
        mock_llm.return_value = "I've been having chest pain and trouble breathing."
        patient = _make_patient()
        sim = SimulatedPatient(patient)
        result = sim.initial_complaint()
        assert "chest pain" in result
        assert len(sim._conversation_history) == 1

    @patch("loop2.ddxplus.patient_simulator.call_llm")
    def test_initial_complaint_leakage_triggers_retry(self, mock_llm):
        # First response leaks, second is clean
        mock_llm.side_effect = [
            "I might have a pulmonary embolism.",  # leaks
            "I have leg swelling and chest pain.",  # clean
        ]
        patient = _make_patient()
        sim = SimulatedPatient(patient)
        result = sim.initial_complaint()
        assert "embolism" not in result.lower()
        assert mock_llm.call_count == 2

    @patch("loop2.ddxplus.patient_simulator.call_llm")
    def test_initial_complaint_all_leak_uses_fallback(self, mock_llm):
        mock_llm.return_value = "I think I have pulmonary embolism issues."
        patient = _make_patient()
        sim = SimulatedPatient(patient)
        result = sim.initial_complaint()
        # After max retries, fallback is used — should not leak
        assert "embolism" not in result.lower()
        assert len(sim._conversation_history) == 1

    @patch("loop2.ddxplus.patient_simulator.call_llm")
    def test_answer_appends_to_history(self, mock_llm):
        mock_llm.return_value = "Yes, I have some chest pain."
        patient = _make_patient()
        sim = SimulatedPatient(patient)
        sim.answer("Do you have chest pain?")
        assert len(sim._conversation_history) == 1
        assert sim._conversation_history[0][0] == "Do you have chest pain?"

    @patch("loop2.ddxplus.patient_simulator.call_llm")
    def test_answer_consistency_across_turns(self, mock_llm):
        mock_llm.side_effect = [
            "Yes, I have chest pain on the right side.",
            "Yes, still on the right side, it's a sharp pain.",
        ]
        patient = _make_patient()
        sim = SimulatedPatient(patient)
        sim.answer("Where is your chest pain?")
        sim.answer("Can you describe the chest pain again?")
        # Both answered; history has 2 turns
        assert len(sim._conversation_history) == 2


# ---------------------------------------------------------------------------
# Integration tests (require GROQ_API_KEY)
# ---------------------------------------------------------------------------

INTEGRATION = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live simulator integration tests",
)


@INTEGRATION
class TestSimulatedPatientIntegration:
    """5 live sessions verifying consistent, non-leaking responses."""

    def _run_session(self, pathology: str, questions: list[str]) -> list[str]:
        from loop2.ddxplus.loader import load_ddxplus

        patients = load_ddxplus(split="test", max_rows=5000)
        matching = [p for p in patients if p.ground_truth_pathology == pathology]
        assert matching, f"No patients found for {pathology}"
        record = matching[0]
        sim = SimulatedPatient(record)
        complaint = sim.initial_complaint()
        assert complaint, "Initial complaint should not be empty"
        assert _check_leakage(complaint, record.ground_truth_pathology) is False, \
            f"Initial complaint leaked pathology: {complaint}"

        answers = []
        for q in questions:
            answer = sim.answer(q)
            assert answer, f"Empty answer for: {q}"
            assert _check_leakage(answer, record.ground_truth_pathology) is False, \
                f"Answer leaked pathology '{record.ground_truth_pathology}': {answer}"
            answers.append(answer)
        return answers

    def test_pe_session(self):
        self._run_session("Pulmonary embolism", [
            "Do you have any leg pain or swelling?",
            "Have you traveled recently?",
            "Do you feel short of breath?",
        ])

    def test_pericarditis_session(self):
        self._run_session("Pericarditis", [
            "Does the chest pain change when you lean forward?",
            "Have you had a recent infection or fever?",
        ])

    def test_sle_session(self):
        self._run_session("SLE", [
            "Do you have any joint pain?",
            "Have you noticed any rashes?",
        ])

    def test_influenza_session(self):
        self._run_session("Influenza", [
            "Do you have a fever?",
            "Are you having muscle aches?",
        ])

    def test_panic_attack_session(self):
        self._run_session("Panic attack", [
            "Do you feel your heart racing?",
            "Are you having trouble breathing?",
        ])
