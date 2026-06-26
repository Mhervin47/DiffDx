"""Phase 6 — extended symptom dedup (Pass 4: >80% content-word similarity)."""
import pytest

from loop1.compressor import _dedup_symptoms, _similarity_ratio, _content_words
from loop1.schemas import Symptom


# ── _similarity_ratio ─────────────────────────────────────────────────────────

def test_identical_sets_give_1():
    w = frozenset({"fever", "high"})
    assert _similarity_ratio(w, w) == 1.0


def test_disjoint_sets_give_0():
    a = frozenset({"fever"})
    b = frozenset({"cough"})
    assert _similarity_ratio(a, b) == 0.0


def test_partial_overlap():
    a = frozenset({"fever", "high"})
    b = frozenset({"fever", "persistent"})
    # |intersection|=1, |union|=3  → 1/3 ≈ 0.333
    ratio = _similarity_ratio(a, b)
    assert abs(ratio - 1 / 3) < 1e-9


def test_empty_sets_give_0():
    assert _similarity_ratio(frozenset(), frozenset({"fever"})) == 0.0
    assert _similarity_ratio(frozenset({"fever"}), frozenset()) == 0.0


# ── _dedup_symptoms Pass 4 — high-similarity variant merging ─────────────────

def test_fever_variants_merged():
    symptoms = [
        Symptom(name="fever", onset="3 days ago"),
        Symptom(name="high fever", onset=""),
        Symptom(name="persistent fever", onset=""),
    ]
    result = _dedup_symptoms(symptoms)
    names = [s.name.lower() for s in result]
    # All three should collapse to one entry
    fever_entries = [n for n in names if "fever" in n]
    assert len(fever_entries) == 1


def test_longer_name_kept_over_shorter_variant():
    symptoms = [
        Symptom(name="fever", onset=""),
        Symptom(name="persistent high fever", onset="3 days ago"),
    ]
    result = _dedup_symptoms(symptoms)
    assert len(result) == 1
    # The longer/more specific name should win
    assert "fever" in result[0].name.lower()


def test_onset_preserved_across_merge():
    symptoms = [
        Symptom(name="fever", onset="3 days ago"),
        Symptom(name="high fever", onset=""),
    ]
    result = _dedup_symptoms(symptoms)
    assert len(result) == 1
    assert result[0].onset == "3 days ago"


def test_unrelated_symptoms_not_merged():
    symptoms = [
        Symptom(name="fever", onset="3 days ago"),
        Symptom(name="cough", onset="1 week"),
        Symptom(name="rash", onset="yesterday"),
    ]
    result = _dedup_symptoms(symptoms)
    assert len(result) == 3


def test_existing_passes_still_work_after_pass4():
    # Pass 1: exact match
    symptoms = [
        Symptom(name="Headache", onset="2 days ago"),
        Symptom(name="headache", onset="", severity="severe"),
    ]
    result = _dedup_symptoms(symptoms)
    assert len(result) == 1
    assert result[0].severity == "severe"


def test_single_symptom_unchanged():
    symptoms = [Symptom(name="nausea", onset="today")]
    result = _dedup_symptoms(symptoms)
    assert len(result) == 1
    assert result[0].name == "nausea"


def test_empty_list_unchanged():
    assert _dedup_symptoms([]) == []
