# MedicalConvo — Loop 1: Diagnostic AI Dialogue System

> **Research prototype.** Not for clinical use.

---

## What Is This?

A multi-turn AI diagnostic interview system. Given a patient's chief complaint, the system conducts a structured clinical dialogue — asking follow-up questions, updating a running patient profile, and narrowing toward a primary diagnosis — without any human clinician in the loop.

The output is a structured, schema-locked session record ready to train a future fine-tuned model (Loop 2).

---

## The Problem Being Solved

Current LLM chatbots applied to medicine either:
- Ask generic, non-adaptive questions (no differential reasoning)
- Have no memory — they can't build on what the patient said two turns ago
- Produce unstructured prose that can't feed a training pipeline

Loop 1 solves all three: it reasons like a physician (differential + Bayesian update), maintains structured state across turns, and emits machine-readable records.

---

## System Architecture

```
Patient input
     │
     ▼
┌──────────────────────────────────────────────┐
│                  Session.run()               │
│                                              │
│  ┌─────────────┐    ┌───────────────────┐    │
│  │ Safety check│    │ Profile updater   │    │
│  │ (substring) │    │ (8B LLM, v0.3)    │    │
│  └─────────────┘    └───────────────────┘    │
│                                              │
│  ┌─────────────────────────────────────┐     │
│  │ Doctor LLM (70B)                    │     │
│  │  + FAISS/MMR exemplar retrieval     │     │
│  │  + Context compression (8B LLM)     │     │
│  └─────────────────────────────────────┘     │
│                                              │
│  Termination: confidence / max_turns /       │
│               safety_stop / user_quit        │
└──────────────────────────────────────────────┘
     │
     ▼
FinalRecord (JSON) + per-turn JSONL log
```

---

## Technology Stack

| Component | Choice | Why |
|---|---|---|
| Doctor LLM | `meta-llama/llama-3.3-70b-instruct` (OpenRouter) | Best open-weight instruction following at time of selection |
| Profile updater / Compressor | `meta-llama/llama-3.1-8b-instant` (OpenRouter) | Fast, cheap for structured extraction tasks |
| Embedding model | `all-MiniLM-L6-v2` (local, 384d) | Runs offline, zero latency after load |
| Vector store | FAISS (in-memory) | No infra needed; exemplar bank is small |
| Data validation | Pydantic v2 | Schema-locked models, parse-and-retry on LLM failures |
| Logging | JSONL + JSON final record | Append-only, crash-safe, training-ready |

---

## What Was Built — Phase by Phase

### Phase 0 — Setup
- Project skeleton, `pyproject.toml`, `config.yaml`
- All Pydantic schemas locked: `PatientProfile`, `DoctorTurnOutput`, `FinalRecord`, `Exemplar`, `TurnRecord`
- JSONL session logging wired up from day one

### Phase 1 — Single-Turn Sanity
- `generate_turn(profile, history) → DoctorTurnOutput`
- Doctor system prompt v0.1: role definition, JSON output schema, confidence calibration anchors
- Parse-and-retry on JSON failure (up to 3 retries, error fed back to model)
- 5 hand-written test profiles; <5% parse failure rate across 50 runs

### Phase 2 — Multi-Turn Loop
- `Session.run()` loop: up to 15 turns, terminates on `confidence_threshold` or `max_turns`
- `ProfileUpdater`: separate 8B LLM call extracts a `ProfileDelta` from each patient answer and applies it to the structured `PatientProfile`
- Turn history passed to doctor each turn; full session logged turn-by-turn before asking for next input (crash-safe)

### Phase 3 — Context Compression
- `compress_context(profile, history, keep_recent=3)`: keeps last 3 turns verbatim, summarizes earlier turns into `profile.running_summary`
- Compressor LLM prompt v0.2: explicit dedup instruction — merges symptom variants with >80% Jaccard content-word similarity
- Token count at turn 15 stays within 1.5× the token count at turn 5

### Phase 4 — Static ICL Exemplars
- 15 hand-curated exemplars covering: vague complaint clarification, red-flag screening, timeline pinning, differential narrowing, emergency escalation, pediatric/geriatric variation
- Exemplars include full CoT: situation summary, chosen question, and reasoning that made it the strongest move
- `select_random(k=3)` injects 3 exemplars per turn; all marked `frozen: true`

### Phase 5 — Dynamic FAISS Retrieval + MMR
- All 15 exemplars embedded at startup (384d vectors, committed to JSON)
- `get_exemplars_for_profile(profile, turn_history) → List[Exemplar]`
- Query: `chief_complaint + top symptoms + age/sex` → cosine search over FAISS index
- MMR re-ranking (λ=0.5) for relevance + diversity balance
- Retrieved exemplar IDs logged per turn (required for Loop 2 attribution)

**Eval results (N=20 sessions):**

| Condition | MMR hit rate (turns 1-3) | Random baseline |
|---|---|---|
| Measles | **100%** | ~15% |
| CVST | **100%** | ~15% |
| SLE | 0% turn 2, 100% turn 3 | ~15% |

SLE dip at turn 2 is non-critical: rash+joint query pulls an adjacent exemplar; by turn 3 the profile is specific enough to recover.

### Phase 6 — Hardening
- **Schema versioning**: `SCHEMA_VERSION = "0.5.0"` on every logged record
- **Safety system** (`safety.py`): substring match on 20+ emergency phrases in patient input; triggers `EMERGENCY_MESSAGE`, logs `SafetyEvent`, terminates session as `safety_stop`
- **Closing turn** (`closing_turn.py`): generates `leading_diagnosis`, `differential_summary`, `recommended_next_steps`, `unresolved_patient_questions` — appended to `FinalRecord`; skipped on safety stop
- **Replay script** (`scripts/replay_session.py`): reads a logged JSONL session, re-runs doctor on it with a new prompt (temperature=0, deterministic)
- **Profile state** logged in every `turn_complete` event (enables full replay)
- **62 new tests**: `test_safety.py`, `test_phase6_schemas.py`, `test_phase6_compressor.py`, `test_phase6_session.py`

---

## Key Design Decisions

**Separate LLMs for separate jobs.** The 70B model only does one thing: ask the best next question. Structured extraction (profile delta, compression) goes to the 8B model — cheaper and faster for that task.

**Schema locked at Phase 0.** `PatientProfile`, `DoctorTurnOutput`, `FinalRecord` are append-only. Loop 2 depends on log stability; renaming or removing fields is disallowed.

**Exemplars teach reasoning patterns, not answers.** The doctor prompt explicitly instructs the model not to copy exemplar questions — it should apply the reasoning structure to its specific patient. This prevents the most common ICL failure mode (literal question copying).

**Confidence plateau protection.** The model's `should_stop` is advisory; the session also checks `confidence_to_stop >= threshold` directly. Threshold lowered from 0.65 → 0.55 to match empirical calibration ceiling of llama-3.3-70b on this prompt scale.

**Every turn logged before asking for the next answer.** If the session crashes, the partial JSONL is still valid and replayable.

---

## What a Session Looks Like

```
PROTOTYPE — This is a research tool, not medical advice.

Turn 1 / 15
Doctor: When did you first notice the headache, and has it changed since it started?

Your answer: It started suddenly about 6 hours ago, worst headache of my life

Turn 2 / 15
Doctor: Have you had any stiff neck, sensitivity to light, or fever alongside the headache?

...

Session complete. Reason: confidence_threshold. Final record → logs/final_records/final_<uuid>.json

Doctor's closing statement:
  Subarachnoid Hemorrhage
  Leading dx SAH at 0.72; secondary differentials meningitis (0.14), migraine (0.08) ruled functionally out.

  Next steps:
    • Immediate CT head without contrast
    • If CT negative, lumbar puncture for xanthochromia
    • Neurosurgery consult
```

---

## Data Produced Per Session

| Artifact | Location | Contents |
|---|---|---|
| Session JSONL | `logs/sessions/<uuid>.jsonl` | One event per line: `session_start`, `turn_complete` (with full `profile_state`), `compression_complete`, `safety_event`, `closing_turn_complete`, `session_end` |
| Final record JSON | `logs/final_records/final_<uuid>.json` | Complete session: profile, turn history, differential, primary diagnosis, closing turn, model metadata |

Schema version embedded in every record for forward compatibility.

---

## Test Coverage

```
tests/
  test_doctor.py            — prompt building, JSON parse, retry logic
  test_session.py           — full loop, termination conditions
  test_profile_updater.py   — delta extraction and application
  test_compressor.py        — dedup passes, compression cadence
  test_retrieval.py         — FAISS search, MMR ranking, diversity
  test_schemas.py           — Pydantic validation, field constraints
  test_safety.py            — phrase matching, SafetyEvent logging
  test_phase6_schemas.py    — schema_version on all record types
  test_phase6_compressor.py — Jaccard dedup (Pass 4)
  test_phase6_session.py    — closing turn, replay, safety_stop flow
```

---

## What Is NOT in Loop 1

These are deferred to Loop 2 by design:

- No critic LLM (no preference scoring yet)
- No DPO or any fine-tuning
- No patient simulator — real user input or scripted profiles only
- No production deployment or real patients
- No multimodal input

---

## What Loop 2 Will Consume

Loop 1 is not just a prototype — it is a data collection system. Every session log is structured for training:

- Per-turn `doctor_output` with differential + rationale = **preference target**
- Per-turn `retrieved_exemplar_ids` = **attribution for exemplar bank growth**
- `profile_state` at each turn = **context for critic scoring**
- `termination_reason` + `confidence_to_stop` trajectory = **session quality signal**

When Loop 2 begins: critic LLM scores sessions, high-scoring records become new frozen exemplars, preference pairs feed DPO training of a LoRA adapter.

---

## Current Status

| Phase | Status |
|---|---|
| 0 — Setup | Done |
| 1 — Single-turn | Done |
| 2 — Multi-turn loop | Done |
| 3 — Context compression | Done |
| 4 — Static ICL exemplars | Done |
| 5 — Dynamic FAISS retrieval + MMR | Done |
| 6 — Hardening + replay | In progress — confidence plateau diagnosis and Railway deploy remaining |
