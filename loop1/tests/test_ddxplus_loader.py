from __future__ import annotations

import pytest

from loop2.ddxplus.loader import decode_evidence, is_in_exemplar_pool, load_ddxplus
from loop2.ddxplus.schemas import DDXPlusPatient


def _make_ontology() -> dict:
    return {
        "E_91": {
            "question_en": "Do you have a fever?",
            "is_antecedent": False,
            "data_type": "B",
            "value_meaning": {},
        },
        "E_54": {
            "question_en": "Characterize your pain:",
            "is_antecedent": False,
            "data_type": "M",
            "value_meaning": {
                "V_112": {"en": "haunting", "fr": "lancinante"},
                "V_161": {"en": "burning", "fr": "brûlante"},
            },
        },
        "E_10": {
            "question_en": "Pain severity (1-10):",
            "is_antecedent": False,
            "data_type": "C",
            "value_meaning": {},
        },
        "E_201": {
            "question_en": "Do you have a cough?",
            "is_antecedent": False,
            "data_type": "B",
            "value_meaning": {},
        },
        "E_ANT": {
            "question_en": "Do you smoke?",
            "is_antecedent": True,
            "data_type": "B",
            "value_meaning": {},
        },
    }


class TestDecodeEvidence:
    def test_binary_present(self):
        ont = _make_ontology()
        name, value = decode_evidence("E_91", ont)
        assert name == "Do you have a fever?"
        assert value == 1

    def test_categorical(self):
        ont = _make_ontology()
        name, value = decode_evidence("E_10_@_7", ont)
        assert name == "Pain severity (1-10):"
        assert value == "7"

    def test_multichoice_decoded(self):
        ont = _make_ontology()
        name, value = decode_evidence("E_54_@_V_112", ont)
        assert name == "Characterize your pain:"
        assert value == "haunting"

    def test_multichoice_unknown_value_falls_back_to_code(self):
        ont = _make_ontology()
        name, value = decode_evidence("E_54_@_V_999", ont)
        assert name == "Characterize your pain:"
        assert value == "V_999"

    def test_unknown_code_falls_back_to_code(self):
        ont = _make_ontology()
        name, value = decode_evidence("E_UNKNOWN", ont)
        assert name == "E_UNKNOWN"
        assert value == 1


class TestIsInExemplarPool:
    def test_known_disease_match(self):
        assert is_in_exemplar_pool("Pulmonary embolism") is True

    def test_case_insensitive(self):
        assert is_in_exemplar_pool("pulmonary embolism") is True
        assert is_in_exemplar_pool("APPENDICITIS") is True

    def test_unknown_disease(self):
        assert is_in_exemplar_pool("GERD") is False

    def test_empty_string(self):
        assert is_in_exemplar_pool("") is False


class TestLoadDDXPlus:
    def test_load_100_patients_no_crash(self):
        patients = load_ddxplus(split="test", max_rows=100)
        assert len(patients) == 100

    def test_schema_parses_cleanly(self):
        patients = load_ddxplus(split="test", max_rows=100)
        for p in patients:
            assert isinstance(p, DDXPlusPatient)
            assert p.patient_id.startswith("test_")
            assert isinstance(p.age, int) and p.age > 0
            assert p.sex in ("M", "F")
            assert isinstance(p.ground_truth_pathology, str) and p.ground_truth_pathology
            assert isinstance(p.ground_truth_differential, list)
            assert len(p.ground_truth_differential) > 0
            assert isinstance(p.symptoms, dict)
            assert isinstance(p.antecedents, dict)

    def test_differential_is_list_of_tuples(self):
        patients = load_ddxplus(split="test", max_rows=10)
        for p in patients:
            for name, prob in p.ground_truth_differential:
                assert isinstance(name, str)
                assert 0.0 <= prob <= 1.0

    def test_symptoms_use_readable_names(self):
        patients = load_ddxplus(split="test", max_rows=10)
        for p in patients:
            for key in p.symptoms:
                assert not key.startswith("E_"), f"Raw code leaked into symptoms: {key}"
            for key in p.antecedents:
                assert not key.startswith("E_"), f"Raw code leaked into antecedents: {key}"

    def test_initial_evidence_is_readable(self):
        patients = load_ddxplus(split="test", max_rows=10)
        for p in patients:
            assert not p.initial_evidence.startswith("E_"), (
                f"Raw code leaked into initial_evidence: {p.initial_evidence}"
            )
            assert p.initial_evidence_code.startswith("E_")

    def test_patient_ids_unique(self):
        patients = load_ddxplus(split="test", max_rows=100)
        ids = [p.patient_id for p in patients]
        assert len(ids) == len(set(ids))

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="Unknown split"):
            load_ddxplus(split="bogus")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_ddxplus(ddxplus_dir="/nonexistent/path", split="test")
