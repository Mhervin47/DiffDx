"""Tests for the exemplar retrieval module (Phase 4 + Phase 5)."""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from loop1.retrieval import ExemplarPool, load_pool, select_mmr, select_random
from loop1.schemas import Exemplar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_EXEMPLAR = {
    "exemplar_id": "ex_test_001",
    "tags": ["test"],
    "context_summary": "A test patient.",
    "good_next_question": "Does it hurt?",
    "rationale": "Pain presence is the highest-value gap.",
    "frozen": True,
}


def write_exemplar(directory: Path, name: str, data: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_pool
# ---------------------------------------------------------------------------

def test_load_pool_returns_exemplar_objects():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        write_exemplar(d, "ex_a.json", MINIMAL_EXEMPLAR)
        write_exemplar(d, "ex_b.json", {**MINIMAL_EXEMPLAR, "exemplar_id": "ex_test_002"})
        pool = load_pool(d)
    assert len(pool) == 2
    assert all(isinstance(ex, Exemplar) for ex in pool)


def test_load_pool_sorted_order():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        write_exemplar(d, "ex_z.json", {**MINIMAL_EXEMPLAR, "exemplar_id": "z"})
        write_exemplar(d, "ex_a.json", {**MINIMAL_EXEMPLAR, "exemplar_id": "a"})
        pool = load_pool(d)
    assert pool[0].exemplar_id == "a"
    assert pool[1].exemplar_id == "z"


def test_load_pool_raises_on_missing_required_field():
    bad = {k: v for k, v in MINIMAL_EXEMPLAR.items() if k != "context_summary"}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        write_exemplar(d, "bad.json", bad)
        with pytest.raises(ValueError, match="bad.json"):
            load_pool(d)


def test_load_pool_raises_on_invalid_json():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "corrupt.json").write_text("{ not json }", encoding="utf-8")
        with pytest.raises(ValueError, match="corrupt.json"):
            load_pool(d)


def test_load_pool_raises_when_directory_empty():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="No exemplar files found"):
            load_pool(Path(tmp))


def test_load_pool_accepts_optional_embedding_field():
    data = {**MINIMAL_EXEMPLAR, "embedding": [0.1, 0.2, 0.3]}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        write_exemplar(d, "ex_embed.json", data)
        pool = load_pool(d)
    assert pool[0].embedding == [0.1, 0.2, 0.3]


def test_load_pool_accepts_missing_embedding_field():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        write_exemplar(d, "ex_no_embed.json", MINIMAL_EXEMPLAR)
        pool = load_pool(d)
    assert pool[0].embedding is None


# ---------------------------------------------------------------------------
# select_random
# ---------------------------------------------------------------------------

def _make_pool(n: int) -> ExemplarPool:
    return [
        Exemplar(
            exemplar_id=f"ex_{i:03d}",
            tags=[],
            context_summary=f"Patient {i}",
            good_next_question="Any pain?",
            rationale="Pain is informative.",
        )
        for i in range(n)
    ]


def test_select_random_returns_correct_count():
    pool = _make_pool(10)
    selected = select_random(pool, n=3)
    assert len(selected) == 3


def test_select_random_no_duplicates():
    pool = _make_pool(10)
    selected = select_random(pool, n=5)
    ids = [ex.exemplar_id for ex in selected]
    assert len(ids) == len(set(ids))


def test_select_random_clamps_to_pool_size():
    pool = _make_pool(2)
    selected = select_random(pool, n=10)
    assert len(selected) == 2


def test_select_random_deterministic_with_seeded_rng():
    pool = _make_pool(15)
    rng = random.Random(42)
    first = [ex.exemplar_id for ex in select_random(pool, n=3, rng=rng)]
    rng2 = random.Random(42)
    second = [ex.exemplar_id for ex in select_random(pool, n=3, rng=rng2)]
    assert first == second


def test_select_random_returns_exemplar_objects():
    pool = _make_pool(5)
    selected = select_random(pool, n=3)
    assert all(isinstance(ex, Exemplar) for ex in selected)


# ---------------------------------------------------------------------------
# Integration: load_pool against the real exemplars directory
# ---------------------------------------------------------------------------

def test_real_exemplars_load_and_validate():
    real_dir = Path(__file__).parent.parent / "exemplars"
    pool = load_pool(real_dir)
    assert len(pool) >= 10, "Expected at least 10 hand-curated exemplars"
    assert all(ex.frozen for ex in pool), "All seed exemplars should be frozen"
    assert all(ex.exemplar_id for ex in pool)


def test_real_exemplars_select_three():
    real_dir = Path(__file__).parent.parent / "exemplars"
    pool = load_pool(real_dir)
    selected = select_random(pool, n=3)
    assert len(selected) == 3
    ids = [ex.exemplar_id for ex in selected]
    assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# select_mmr helpers
# ---------------------------------------------------------------------------

DIM = 8  # small dimension for synthetic tests


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def _make_embedded_pool(n: int, dim: int = DIM) -> ExemplarPool:
    """Synthetic pool where each exemplar has a distinct unit-normalised embedding."""
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        items.append(
            Exemplar(
                exemplar_id=f"ex_{i:03d}",
                tags=[],
                context_summary=f"Patient {i}",
                good_next_question="Any pain?",
                rationale="Pain is informative.",
                embedding=vec.tolist(),
            )
        )
    return items


# ---------------------------------------------------------------------------
# select_mmr — unit tests
# ---------------------------------------------------------------------------

def test_select_mmr_returns_correct_count():
    pool = _make_embedded_pool(10)
    q = _unit([1.0] + [0.0] * (DIM - 1))
    result = select_mmr(pool, q, n=3)
    assert len(result) == 3


def test_select_mmr_no_duplicates():
    pool = _make_embedded_pool(10)
    q = _unit([1.0] + [0.0] * (DIM - 1))
    result = select_mmr(pool, q, n=5)
    ids = [ex.exemplar_id for ex in result]
    assert len(ids) == len(set(ids))


def test_select_mmr_n1_returns_one_exemplar():
    pool = _make_embedded_pool(10)
    q = _unit([1.0] + [0.0] * (DIM - 1))
    result = select_mmr(pool, q, n=1)
    assert len(result) == 1


def test_select_mmr_deterministic_same_inputs():
    pool = _make_embedded_pool(15)
    q = _unit([0.5, -0.3] + [0.1] * (DIM - 2))
    first = [ex.exemplar_id for ex in select_mmr(pool, q, n=3)]
    second = [ex.exemplar_id for ex in select_mmr(pool, q, n=3)]
    assert first == second


def test_select_mmr_raises_on_null_embedding():
    pool = _make_embedded_pool(5)
    pool[2] = Exemplar(
        exemplar_id="ex_no_embed",
        tags=[],
        context_summary="Missing embedding.",
        good_next_question="Any pain?",
        rationale="Pain is informative.",
        embedding=None,
    )
    q = _unit([1.0] + [0.0] * (DIM - 1))
    with pytest.raises(ValueError, match="ex_no_embed"):
        select_mmr(pool, q, n=3)


def test_select_mmr_returns_exemplar_objects():
    pool = _make_embedded_pool(10)
    q = _unit([1.0] + [0.0] * (DIM - 1))
    result = select_mmr(pool, q, n=3)
    assert all(isinstance(ex, Exemplar) for ex in result)


def test_select_mmr_clamps_to_pool_size():
    pool = _make_embedded_pool(2)
    q = _unit([1.0] + [0.0] * (DIM - 1))
    result = select_mmr(pool, q, n=10)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# select_mmr — retrieval quality tests (real exemplar embeddings)
# ---------------------------------------------------------------------------

_REAL_DIR = Path(__file__).parent.parent / "exemplars"


def _load_real_pool() -> ExemplarPool:
    return load_pool(_REAL_DIR)


def _exemplar_vec(pool: ExemplarPool, exemplar_id: str) -> np.ndarray:
    for ex in pool:
        if ex.exemplar_id == exemplar_id:
            return np.array(ex.embedding, dtype=np.float32)
    raise KeyError(exemplar_id)


def test_select_mmr_retrieves_measles_when_query_is_measles_embedding():
    pool = _load_real_pool()
    query = _exemplar_vec(pool, "ex_measles_001")
    # Perturb slightly so query != exact exemplar vector, but still close
    noise = np.random.default_rng(7).standard_normal(384).astype(np.float32) * 0.05
    query = query + noise
    query /= np.linalg.norm(query)

    result = select_mmr(pool, query, n=3)
    ids = [ex.exemplar_id for ex in result]
    assert "ex_measles_001" in ids, f"Expected measles exemplar in top-3, got: {ids}"


def test_select_mmr_retrieves_cvst_when_query_is_cvst_embedding():
    pool = _load_real_pool()
    query = _exemplar_vec(pool, "ex_cvst_001")
    noise = np.random.default_rng(13).standard_normal(384).astype(np.float32) * 0.05
    query = query + noise
    query /= np.linalg.norm(query)

    result = select_mmr(pool, query, n=3)
    ids = [ex.exemplar_id for ex in result]
    assert "ex_cvst_001" in ids, f"Expected CVST exemplar in top-3, got: {ids}"


# ---------------------------------------------------------------------------
# embed_patient_state
# ---------------------------------------------------------------------------

from loop1.retrieval import build_faiss_index, embed_patient_state, get_exemplars_for_profile  # noqa: E402
from loop1.schemas import Demographics, History, PatientProfile, Symptom  # noqa: E402


def _minimal_profile(**kwargs) -> PatientProfile:
    defaults = dict(
        session_id="test-session",
        chief_complaint="headache",
        demographics=Demographics(),
        symptoms=[],
        history=History(),
        running_summary="",
    )
    defaults.update(kwargs)
    return PatientProfile(**defaults)


def test_embed_patient_state_returns_384d_array():
    profile = _minimal_profile()
    vec = embed_patient_state(profile)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)


def test_embed_patient_state_unit_normalised():
    profile = _minimal_profile()
    vec = embed_patient_state(profile)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_embed_patient_state_no_symptoms_no_summary_does_not_crash():
    profile = _minimal_profile(chief_complaint="chest pain", symptoms=[], running_summary="")
    vec = embed_patient_state(profile)
    assert vec.shape == (384,)


def test_embed_patient_state_deterministic():
    profile = _minimal_profile(
        chief_complaint="fever and rash",
        symptoms=[Symptom(name="fever"), Symptom(name="rash")],
        running_summary="4-year-old with cephalocaudal rash progression.",
    )
    vec1 = embed_patient_state(profile)
    vec2 = embed_patient_state(profile)
    np.testing.assert_array_equal(vec1, vec2)


def test_embed_patient_state_caps_symptoms_at_5():
    symptoms = [Symptom(name=f"symptom_{i}") for i in range(10)]
    profile = _minimal_profile(symptoms=symptoms)
    vec = embed_patient_state(profile)
    assert vec.shape == (384,)


# ---------------------------------------------------------------------------
# build_faiss_index
# ---------------------------------------------------------------------------

def test_build_faiss_index_ntotal_equals_pool_size():
    pool = _load_real_pool()
    index = build_faiss_index(pool)
    assert index.ntotal == len(pool)


def test_build_faiss_index_raises_on_null_embedding():
    pool = _make_embedded_pool(5, dim=384)
    pool[1] = Exemplar(
        exemplar_id="ex_no_embed",
        tags=[],
        context_summary="Missing embedding.",
        good_next_question="Any pain?",
        rationale="Pain is informative.",
        embedding=None,
    )
    with pytest.raises(ValueError, match="ex_no_embed"):
        build_faiss_index(pool)


# ---------------------------------------------------------------------------
# get_exemplars_for_profile
# ---------------------------------------------------------------------------

def test_get_exemplars_for_profile_returns_correct_count():
    profile = _minimal_profile(chief_complaint="chest pain")
    result = get_exemplars_for_profile(profile, n=3)
    assert len(result) == 3
    assert all(isinstance(ex, Exemplar) for ex in result)


def test_get_exemplars_for_profile_no_duplicates():
    profile = _minimal_profile(chief_complaint="chest pain")
    result = get_exemplars_for_profile(profile, n=3)
    ids = [ex.exemplar_id for ex in result]
    assert len(ids) == len(set(ids))


def test_get_exemplars_for_profile_measles_profile_retrieves_measles():
    profile = _minimal_profile(
        chief_complaint="fever and rash in a child",
        symptoms=[
            Symptom(name="high fever"),
            Symptom(name="blotchy red rash starting at hairline"),
            Symptom(name="cough"),
            Symptom(name="runny nose"),
            Symptom(name="red watery eyes"),
        ],
        running_summary=(
            "9-month-old unvaccinated child with cephalocaudal rash after international travel. "
            "Three-day prodrome of cough, coryza, and conjunctivitis before rash onset."
        ),
    )
    result = get_exemplars_for_profile(profile, n=3)
    ids = [ex.exemplar_id for ex in result]
    assert "ex_measles_001" in ids, f"Expected measles in top-3, got: {ids}"


def test_get_exemplars_for_profile_cvst_profile_retrieves_cvst():
    profile = _minimal_profile(
        chief_complaint="severe headache in a young woman on the pill",
        symptoms=[
            Symptom(name="thunderclap headache"),
            Symptom(name="worsening with lying down"),
            Symptom(name="nausea"),
        ],
        running_summary=(
            "26-year-old woman on oral contraceptive pill presenting with progressive severe headache. "
            "Neurological symptoms raising concern for venous thrombosis."
        ),
    )
    result = get_exemplars_for_profile(profile, n=3)
    ids = [ex.exemplar_id for ex in result]
    assert "ex_cvst_001" in ids, f"Expected CVST in top-3, got: {ids}"
