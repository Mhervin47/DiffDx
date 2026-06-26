from __future__ import annotations

import json
import random
from pathlib import Path

from loop1.schemas import Exemplar, PatientProfile

# All heavy imports are deferred to first use to avoid slow startup.
_EMBED_MODEL = None
_FAISS_MODULE = None

# fastembed model name — equivalent to all-MiniLM-L6-v2, cached in ~/.cache/fastembed/
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_faiss():
    global _FAISS_MODULE
    if _FAISS_MODULE is None:
        import faiss as _faiss  # noqa: PLC0415
        _FAISS_MODULE = _faiss
    return _FAISS_MODULE


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from fastembed import TextEmbedding  # noqa: PLC0415
        _EMBED_MODEL = TextEmbedding(EMBED_MODEL_NAME)
    return _EMBED_MODEL

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ExemplarPool = list[Exemplar]

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_EXEMPLARS_DIR = Path(__file__).parent.parent.parent / "exemplars"


def load_pool(exemplars_dir: Path = _EXEMPLARS_DIR) -> ExemplarPool:
    """
    Load and validate every *.json file in exemplars_dir as an Exemplar.
    Raises ValueError (with the offending filename) if any file fails.
    """
    pool: ExemplarPool = []
    paths = sorted(exemplars_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No exemplar files found in {exemplars_dir}")
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pool.append(Exemplar.model_validate(data))
        except Exception as exc:
            raise ValueError(f"Failed to load exemplar '{path.name}': {exc}") from exc
    return pool


# ---------------------------------------------------------------------------
# Patient-state embedding
# ---------------------------------------------------------------------------

def embed_patient_state(profile: PatientProfile) -> np.ndarray:
    """
    Embed the current patient state as a unit-normalised 384d vector.

    String format:
        "{chief_complaint}. {running_summary}. Symptoms: {s1}, {s2}, ..."
    - running_summary omitted when empty (turn 1 fallback: chief complaint only)
    - Symptoms clause omitted when symptom list is empty
    - At most 5 symptoms included, in list order
    """
    import numpy as np  # noqa: PLC0415
    parts: list[str] = [profile.chief_complaint]

    if profile.running_summary:
        parts.append(profile.running_summary)

    if profile.symptoms:
        top5 = [s.name for s in profile.symptoms[:5]]
        parts.append("Symptoms: " + ", ".join(top5))

    text = ". ".join(parts)
    vec: np.ndarray = next(_get_embed_model().embed([text]))
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(pool: ExemplarPool):
    """
    Build a flat inner-product FAISS index from the pool's precomputed embeddings.
    Unit-normalised vectors make inner product equivalent to cosine similarity.
    Raises ValueError if any exemplar is missing its embedding field.
    """
    missing = [ex.exemplar_id for ex in pool if ex.embedding is None]
    if missing:
        raise ValueError(
            f"build_faiss_index requires pre-computed embeddings. "
            f"Missing on: {missing}. Run scripts/embed_exemplars.py first."
        )

    import numpy as np  # noqa: PLC0415
    faiss = _get_faiss()
    dim = len(pool[0].embedding)  # type: ignore[arg-type]
    index = faiss.IndexFlatIP(dim)
    vecs = np.array([ex.embedding for ex in pool], dtype=np.float32)
    index.add(vecs)
    return index


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_mmr(
    pool: ExemplarPool,
    query_vec: np.ndarray,
    n: int = 3,
    lambda_: float = 0.7,
) -> list[Exemplar]:
    """
    Maximal Marginal Relevance selection.

    Greedily picks n exemplars maximising:
        λ · sim(candidate, query) - (1-λ) · max_{s ∈ selected} sim(candidate, s)

    Vectors are assumed unit-normalised so dot product == cosine similarity.
    Raises ValueError if any exemplar is missing its embedding field.
    """
    import numpy as np  # noqa: PLC0415
    missing = [ex.exemplar_id for ex in pool if ex.embedding is None]
    if missing:
        raise ValueError(
            f"select_mmr requires pre-computed embeddings. "
            f"Missing on: {missing}. Run scripts/embed_exemplars.py first."
        )

    k = min(n, len(pool))
    pool_vecs = np.array([ex.embedding for ex in pool], dtype=np.float32)  # (P, D)
    q = query_vec.astype(np.float32)

    relevance = pool_vecs @ q  # (P,) cosine sim to query

    selected_indices: list[int] = []
    # similarity matrix between pool items, computed lazily as items are selected
    selected_sims = np.full((len(pool), k), -1.0, dtype=np.float32)

    for step in range(k):
        if step == 0:
            # No selected items yet — pure relevance
            scores = relevance.copy()
        else:
            last_idx = selected_indices[-1]
            # Update the running max-similarity-to-selected column
            selected_sims[:, step - 1] = pool_vecs @ pool_vecs[last_idx]
            max_sim_to_selected = selected_sims[:, :step].max(axis=1)
            scores = lambda_ * relevance - (1 - lambda_) * max_sim_to_selected

        # Mask already-selected
        for idx in selected_indices:
            scores[idx] = -np.inf

        selected_indices.append(int(np.argmax(scores)))

    return [pool[i] for i in selected_indices]


def select_random(
    pool: ExemplarPool,
    n: int = 3,
    rng: random.Random | None = None,
) -> list[Exemplar]:
    """
    Return n exemplars chosen uniformly at random (without replacement).
    Pass a seeded rng for deterministic behaviour in tests or replay mode.
    Phase 5 hook: this function will be joined by select_mmr() which accepts
    a query embedding and re-ranks for relevance + diversity.
    """
    _rng = rng or random
    k = min(n, len(pool))
    return _rng.sample(pool, k)


# ---------------------------------------------------------------------------
# Module-level singletons + public entry points
# ---------------------------------------------------------------------------

_pool: ExemplarPool | None = None
_index: object | None = None  # faiss.IndexFlatIP, typed as object to avoid eager import


def _get_pool() -> ExemplarPool:
    global _pool
    if _pool is None:
        _pool = load_pool()
    return _pool


def _get_index() -> faiss.IndexFlatIP:
    global _index
    if _index is None:
        _index = build_faiss_index(_get_pool())
    return _index


def get_exemplar_by_id(exemplar_id: str) -> Exemplar:
    """Return a specific exemplar by ID. Raises ValueError if not found."""
    pool = _get_pool()
    for ex in pool:
        if ex.exemplar_id == exemplar_id:
            return ex
    available = [ex.exemplar_id for ex in pool]
    raise ValueError(f"Exemplar '{exemplar_id}' not found. Available: {available}")


def get_exemplars(n: int = 3, rng: random.Random | None = None) -> list[Exemplar]:
    """
    Phase 4 random selection — kept for ablation testing and backward compatibility.
    Phase 5 sessions use get_exemplars_for_profile instead.
    """
    return select_random(_get_pool(), n, rng=rng)


def get_exemplars_for_profile(
    profile: PatientProfile,
    n: int = 3,
    lambda_: float = 0.7,
) -> list[Exemplar]:
    """
    Returns n exemplars for the current patient state.
    Uses random selection to avoid loading the embedding model on startup.
    """
    return select_random(_get_pool(), n=n)
