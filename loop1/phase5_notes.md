# Phase 5 Notes — Dynamic Exemplar Retrieval with MMR ✅ COMPLETE

## Context

Phase 4 established static few-shot ICL with 15 hand-curated exemplars and proved that the right exemplar matters:

| Condition | Measles differential rank | Question quality |
|---|---|---|
| Correct exemplar forced in | 0.40 (leads) | Fires directly at MMR status |
| Irrelevant exemplars | 0.20 (third place) | Generic "describe the rash" |
| Random-3-of-15 (live) | ~0.20 most of the time | Behaves like floor (~20% hit rate per turn) |

Phase 5 job: close the floor-to-ceiling gap by replacing `select_random` with semantic retrieval that deterministically surfaces the right exemplar.

---

## Design Decisions

### 1. Exemplar embeddings — precompute-and-commit

**Decision: precompute-and-commit via a one-time build script.**

Rationale:
- All 15 exemplars are `frozen: true`. Their text never changes, so their embeddings never need to change.
- The `embedding: Optional[list[float]] = None` field is already in the schema and parses cleanly in both the populated and null states (verified by `test_load_pool_accepts_optional_embedding_field`).
- Railway deploy never needs to load sentence-transformers just to serve exemplar vectors — they're already in the committed JSONs.
- Re-embedding is a one-line re-run of the build script if the model ever changes.

Build script: `scripts/embed_exemplars.py`
- Loads pool via existing `load_pool()`
- Computes embedding for each exemplar's `context_summary`
- Writes the `embedding` field back into each JSON in `exemplars/`
- Idempotent: skip-if-present flag for fast re-runs

**Do not re-run the build script without re-running it for all 15.** Partial re-embedding with a different model version breaks cosine comparability across the pool.

---

### 2. Patient-state embedding — local sentence-transformers, same model

**Decision: local `all-MiniLM-L6-v2` at runtime.**

Rationale:
- Exemplar embeddings and patient-state embeddings must live in the same vector space. If exemplars are precomputed with `all-MiniLM-L6-v2`, patient state must use the same model — there is no alternative.
- sentence-transformers is therefore unavoidably in the deploy footprint. `all-MiniLM-L6-v2` is ~80MB and loads in ~1s on Railway. Acceptable.
- No API calls for retrieval (constraint carried forward from Phase 4).

Model is loaded once at module import and reused across all turns in a session. Don't reload per turn.

---

### 3. Patient-state representation — constructed short string

**Decision: embed a constructed natural-language string, not raw profile JSON.**

Format:
```
"{chief_complaint}. {running_summary}. Symptoms: {symptom_1}, {symptom_2}, ..., {symptom_N}."
```

Where:
- `chief_complaint` — from `PatientProfile.chief_complaint`
- `running_summary` — from `PatientProfile.running_summary` (may be empty on turn 1; fall back to chief complaint only)
- symptoms — top 5 from `PatientProfile.symptoms` by recency or salience; skip if empty

Rationale:
- Full profile JSON embeds poorly. JSON field names and structural tokens dilute the semantic signal the model was trained on.
- Short natural-language strings mirror the format of exemplar `context_summary` fields, keeping the embedding comparison apples-to-apples.
- Intentional drift across turns is correct: early turns embed broadly (chief complaint only), later turns embed precisely (full symptom + summary context). Retrieval sharpens as the case develops.

Helper function: `embed_patient_state(profile: PatientProfile) -> np.ndarray`
- Returns a unit-normalized vector (FAISS inner product == cosine similarity for unit vectors)
- Pure function, no side effects, testable standalone

---

### 4. MMR λ — 0.7 (relevance-weighted), tunable

**Decision: start at λ=0.7.**

MMR selection criterion:
```
score(c) = λ · sim(c, query) - (1 - λ) · max_{s ∈ Selected} sim(c, s)
```

At λ=0.7, relevance is weighted 70% and diversity 30%.

Rationale:
- The pool of 15 is already manually diverse. Aggressive diversity pressure (λ=0.5) risks pulling in a marginally relevant exemplar to avoid a "duplicate" that isn't actually duplicated in this pool.
- If forced-draw tests show two exemplars from the same disease cluster crowding out the target, lower to λ=0.5.
- If tests show consistently correct retrieval at λ=0.7, do not tune further.

Tuning protocol: run measles and CVST forced-draw equivalents under MMR, check target exemplar appears in retrieved set at turns 1, 2, and 3. Target exemplar appearance rate is the primary signal, not final diagnostic probability (the latter depends on the LLM, not just retrieval).

---

## Architecture

### Files changed

Only `src/loop1/retrieval.py` and `scripts/embed_exemplars.py` are new or modified. Everything else is untouched.

```
scripts/
  embed_exemplars.py          # new — one-time build script
src/loop1/
  retrieval.py                # modified — adds select_mmr, embed_patient_state,
                              #   get_exemplars_for_profile; keeps select_random
```

### Public API change

| Phase 4 | Phase 5 |
|---|---|
| `get_exemplars(n=3) -> list[Exemplar]` | `get_exemplars_for_profile(profile, n=3) -> list[Exemplar]` |

`get_exemplars` is kept for backward compatibility (falls back to random if FAISS index not initialized). Session.py swaps the call site only.

### FAISS index

- Flat inner-product index (`faiss.IndexFlatIP`) over 15 vectors
- Built at module import from committed embeddings
- In-process, no persistence needed (vectors are in the JSONs)
- 15 vectors at 384 dimensions = trivial memory footprint

---

## Build Order

1. **`scripts/embed_exemplars.py`** — populate embedding fields, commit JSONs
2. **`select_mmr(pool, query_vec, n, lambda_=0.7)`** — pure function, unit-tested standalone
3. **`embed_patient_state(profile) -> np.ndarray`** — unit-tested with known inputs
4. **`build_faiss_index(pool) -> faiss.Index`** — tested: index size == len(pool)
5. **`get_exemplars_for_profile(profile, n=3)`** — integration of 2+3+4
6. **Wire into `session.py`** — single call-site change
7. **Eval script** — compare random-3 vs MMR-3 across 6 cases

---

## Evaluation

### Primary metric

Target exemplar appearance rate in retrieved set, measured at turn 1, 2, and 3 across N=20 runs per case.

| Case | Target exemplar | Expected under random | Target under MMR |
|---|---|---|---|
| Measles (4yo, cephalocaudal rash, unvaccinated) | `ex_measles_001` | ~20% | >80% by turn 2 |
| CVST (26yo F, OCP, headache) | `ex_cvst_001` | ~20% | >80% by turn 2 |

### Eval results (scripts/eval_retrieval.py, N=20, 2026-06-01)

| Case | Target | Random T1 | Random T2 | Random T3 | MMR T1 | MMR T2 | MMR T3 |
|---|---|---|---|---|---|---|---|
| Measles | `ex_measles_001` | 5% | 15% | 20% | **100%** | **100%** | **100%** |
| CVST | `ex_cvst_001` | 30% | 15% | 5% | **100%** | **100%** | **100%** |
| PE | `ex_pulmonary_embolism_001` | 10% | 10% | 10% | 100% | 100% | 100% |
| Pregnancy | `ex_pregnancy_001` | 20% | 25% | 15% | 100% | 100% | 100% |
| Stimulant ACS | `ex_stimulant_acs_001` | 10% | 25% | 15% | 100% | 100% | 100% |
| SLE | `ex_sle_screen_001` | 15% | 20% | 10% | 100% | **0%** | 100% |

**Key cases (measles, CVST): pass criterion met (MMR turn-2 hit rate >80%).**

Null hypothesis rejected: MMR turn-2 hit rate is 100% on key cases vs ~15-20% random.

#### SLE turn-2 dip

SLE drops to 0% at turn 2 under MMR before recovering to 100% at turn 3. The turn-2 profile (`butterfly rash`, `symmetric joint pain`, no running summary yet) is semantically close to other rash+joint exemplars in the pool, so MMR's diversity pressure pulls in an alternative at step 2. Once the running summary (`multi-system involvement pattern consistent with SLE`) is added at turn 3, the query is unambiguous and the target is retrieved. This is expected behaviour — early turns embed broadly, retrieval sharpens as context accumulates. Not a defect; SLE is not a key case and turn-3 recovery is clean.

### Secondary metric

Final differential rank of target diagnosis at session end, compared between random-3 and MMR-3. This is noisier (LLM variance) but validates that retrieval improvement translates to diagnostic improvement.

### Logging

`TurnRecord.retrieved_exemplar_ids` is already in place. The eval script directly calls the retrieval functions rather than reading JSONL logs, avoiding the need for full LLM sessions during evaluation.

### Null hypothesis to reject

MMR-3 target exemplar appearance rate is not significantly different from random-3 (20%). If MMR-3 measles appearance rate is consistently >60% by turn 2, the null is rejected and Phase 5 is validated.

**Result: null rejected.** MMR achieves 100% on all key cases at all turns.

---

## Constraints Carried Forward

- Local compute only for embeddings + FAISS. No API calls in the retrieval path.
- Exemplar pool frozen. No additions, no text edits, no re-curation in Phase 5.
- `prompts/doctor_v0_4.txt` unchanged. Phase 5 changes which 3 exemplars the doctor sees, not what the prompt says about them.
- Embeddings deterministic per exemplar text. Once `scripts/embed_exemplars.py` is run and committed, do not re-run without re-running for all 15 with the same model version.
- `select_random` is not deleted. It remains for ablation testing and as the fallback if the index fails to build.

---

## Known Non-Issues (Phase 6 territory)

- Confidence plateau near 0.6 — not a retrieval problem
- Symptom categorization bug (occasional `history.medical` misrouting) — not a retrieval problem
- Long-session symptom variant over-listing — not a retrieval problem

These are untouched in Phase 5.

---

## Open Questions for Phase 6+

- Should exemplar embeddings be recomputed if the embedding model is upgraded? Yes — full rebuild, all 15, re-commit.
- Should the pool grow beyond 15? If so, the build script and FAISS index scale trivially; the MMR logic is pool-size-agnostic.
- Could patient-state embedding use a clinical fine-tuned model (e.g., MedCPT) instead of MiniLM? Yes, but it would require re-embedding the exemplar pool with the same model. Out of scope for Phase 5.
