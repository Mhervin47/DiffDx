# MedicalConvo — Complete System Documentation

**Current Phase**: 10 (Specialist Routing UI + Doctor Portal)  
**Language**: Python 3.11+  
**Schema Version**: 0.9.0  
**Last Updated**: 2026-07-02

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Configuration & Dependencies](#3-configuration--dependencies)
4. [Locked Schemas](#4-locked-schemas)
5. [Loop 1 — Core Diagnostic System](#5-loop-1--core-diagnostic-system)
6. [Loop 2 — Evaluation & Critic](#6-loop-2--evaluation--critic)
7. [Loop 3 — Routing & Specialist Assignment](#7-loop-3--routing--specialist-assignment)
8. [Web API & Frontend](#8-web-api--frontend)
9. [Prompts (Versioned)](#9-prompts-versioned)
10. [Exemplars & Test Cases](#10-exemplars--test-cases)
11. [Data Files & Datasets](#11-data-files--datasets)
12. [Scripts & Utilities](#12-scripts--utilities)
13. [Phase Progression](#13-phase-progression)
14. [Deployment & Environment](#14-deployment--environment)
15. [Key Architectural Decisions](#15-key-architectural-decisions)
16. [Common Workflows](#16-common-workflows)
17. [Troubleshooting](#17-troubleshooting)
18. [Glossary](#18-glossary)

---

## 1. Project Overview

MedicalConvo is a research sandbox for AI-assisted medical diagnosis. It is **not** for clinical use. The system:

- Conducts multi-turn diagnostic dialogues between an AI doctor and a patient
- Uses in-context learning (ICL) with hand-curated exemplars to improve question quality
- Scores turn quality via a critic LLM after the session
- Routes the final diagnosis to the appropriate medical specialist with urgency classification
- Exposes all of this via a web UI for patients, and a separate doctor portal

The architecture is organized in three loops:

| Loop | Responsibility |
|------|---------------|
| **Loop 1** | Diagnostic dialogue — profile building, question generation, session logging |
| **Loop 2** | Evaluation — patient simulation (DDXPlus), critic scoring, ground-truth comparison |
| **Loop 3** | Routing — disease → specialty mapping, urgency classification, ambiguity handling |

**Primary model**: Llama 3.3 70B via Groq API  
**Fallback chain**: OpenRouter → Cerebras → Gemini (automatic on 429)  
**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local)  
**Vector store**: FAISS (flat inner-product, cosine similarity via unit normalization)

---

## 2. Directory Structure

```
loop1/
├── src/
│   ├── loop1/                        # Core diagnostic runtime
│   │   ├── schemas.py                # Pydantic models (locked, versioned)
│   │   ├── config.py                 # YAML config loader
│   │   ├── llm.py                    # LLM calls, fallbacks, retry, JSON parsing
│   │   ├── doctor.py                 # Doctor turn generation + exemplar injection
│   │   ├── session.py                # Multi-turn session loop + termination
│   │   ├── retrieval.py              # FAISS embedding + MMR exemplar selection
│   │   ├── compressor.py             # Context compression (dedup + LLM summary)
│   │   ├── profile_updater.py        # Profile delta extraction from Q&A
│   │   ├── safety.py                 # Emergency phrase detection
│   │   ├── closing_turn.py           # Plain-language closing statement
│   │   └── logging_utils.py          # Session JSONL + final record I/O
│   │
│   ├── loop2/                        # Evaluation layer
│   │   ├── critic/
│   │   │   ├── critic.py             # Turn-level critic LLM call
│   │   │   ├── critique_schema.py    # TurnCritique Pydantic model
│   │   │   └── aggregator.py         # Aggregate critique stats
│   │   ├── ddxplus/
│   │   │   ├── schemas.py            # DDXPlusPatient, DDXPlusEvalResult
│   │   │   ├── loader.py             # DDXPlus dataset loader
│   │   │   ├── patient_simulator.py  # LLM-backed patient (grounded in DDXPlus)
│   │   │   └── evaluator.py          # Ground-truth comparison metrics
│   │   └── runners/
│   │       └── phase7_eval.py        # Full 20-patient eval pipeline
│   │
│   └── loop3/                        # Routing layer
│       ├── routing/
│       │   ├── router.py             # Main routing orchestrator
│       │   ├── routing_schema.py     # RoutingDecision, RoutingOption schemas
│       │   ├── urgency_classifier.py # Disease → urgency level
│       │   ├── ambiguity_handler.py  # Multi-specialty differential handling
│       │   └── map_loader.py         # Load disease_specialty_map.json
│       └── runners/
│           └── phase9_routing_runner.py  # Reads session → produces RoutingDecision
│
├── web/
│   ├── api.py                        # FastAPI app (all endpoints)
│   ├── api_session.py                # APISession: stateful web session wrapper
│   └── data/
│       ├── users.json                # Patient + doctor accounts
│       ├── doctors.json              # Mock doctor directory (specialty, slots)
│       ├── appointments.json         # Booked appointments
│       ├── medicalconvo.db           # SQLite (reserved for future use)
│       └── disease_specialty_map.json
│
├── web/static/                       # Vanilla HTML/CSS/JS frontend
│   ├── index.html                    # Case selection grid
│   ├── session.html                  # Live diagnostic chat UI
│   ├── report.html                   # Final diagnosis + specialist routing
│   ├── doctor-portal.html            # Doctor appointment queue + session detail
│   ├── login.html                    # Auth page (patient + doctor)
│   ├── history.html                  # Patient session history
│   ├── auth.js                       # JWT token storage + request headers
│   └── styles.css                    # Dark theme (bg #0f1117, accent teal #00b4d8)
│
├── prompts/                          # Versioned prompt templates
│   ├── doctor_v0_6.txt               # Current doctor prompt
│   ├── doctor_v0_{1,2,3,4,5}.txt    # Historical (for replay)
│   ├── compressor_v0_{1,2}.txt
│   ├── profile_updater_v0_{1,2,3}.txt
│   ├── critic_turn_v0_1.txt
│   └── simulated_patient_v0_1.txt
│
├── exemplars/                        # Hand-curated clinical examples (~50 files)
│   └── ex_*.json                     # All frozen: true (seed set)
│
├── test_cases/                       # Hand-written patient profiles
│   └── case_0{1,2,3,4,5,6}.json
│
├── data/
│   ├── DDXPlus_Raw/                  # GITIGNORED — download separately
│   ├── ddxplus_eval_set.json         # 20 hand-curated eval patients (committed)
│   └── disease_specialty_map.json    # 70+ disease → specialty mappings
│
├── logs/
│   ├── sessions/                     # Per-session JSONL (one per session_id)
│   └── final_records/                # FinalRecord JSON (one per session_id)
│
├── scripts/                          # Dev utilities
│   ├── hello_world.py
│   ├── run_session.py
│   ├── run_single_turn.py
│   ├── replay_session.py
│   ├── embed_exemplars.py
│   ├── eval_retrieval.py
│   ├── curate_eval_set.py
│   ├── diagnose_confidence.py
│   ├── validate_confidence.py
│   └── bake_off.py
│
├── tests/                            # pytest test suite
├── config.yaml                       # Global configuration
├── pyproject.toml                    # Package definition (uv)
├── Procfile                          # Heroku deployment
└── CLAUDE.md                         # Loop 1 original spec (authoritative)
```

---

## 3. Configuration & Dependencies

### 3.1 config.yaml

```yaml
logging:
  final_dir: logs/final_records
  schema_version: 0.5.0
  session_dir: logs/sessions

models:
  compressor: groq/llama-3.3-70b-versatile
  doctor: groq/llama-3.3-70b-versatile
  embedder: sentence-transformers/all-MiniLM-L6-v2
  profile_updater: groq/llama-3.3-70b-versatile

prompt_versions:
  compressor: v0.2
  doctor: v0.6
  profile_updater: v0.3

thresholds:
  compression_keep_recent: 3       # Keep last N turns verbatim; compress older
  compression_token_threshold: 3000
  confidence_to_stop: 0.55         # Doctor stops asking when confidence >= this
  exemplar_pool_size: 50
  llm_max_tokens: 2048
  llm_temperature: 0.3
  max_turns: 15                    # Hard cap on dialogue length
  mmr_lambda: 0.5                  # MMR: 0=pure diversity, 1=pure similarity
  top_k_exemplars: 3               # Exemplars injected per doctor prompt
```

### 3.2 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `litellm` | ≥1.40.0 | Multi-provider LLM abstraction (Groq, OpenRouter, etc.) |
| `pydantic` | ≥2.7.0 | Schema validation for all JSON outputs |
| `faiss-cpu` | ≥1.7.4 | FAISS vector similarity for exemplar retrieval |
| `sentence-transformers` | ≥3.0.0 | Embedding model (all-MiniLM-L6-v2) |
| `fastembed` | ≥0.8.0 | Alternative embedding backend (lazy-loaded) |
| `tenacity` | ≥8.3.0 | Retry logic with exponential backoff |
| `groq` | ≥1.2.0 | Groq API client |
| `fastapi` | ≥0.111.0 | Web framework |
| `uvicorn` | ≥0.30.0 | ASGI server |
| `transformers` | ==4.46.3 | Pinned — sentence-transformers compatibility |
| `pandas` | ≥3.0.3 | DDXPlus dataset manipulation |
| `pytest` | ≥8.0.0 | Test suite |

### 3.3 Environment Variables

```
GROQ_API_KEY=<required — from console.groq.com>
OPENROUTER_API_KEY=<optional — fallback LLM provider>
CEREBRAS_API_KEY=<optional — fallback>
GEMINI_API_KEY=<optional — fallback>
CRITIC_MODEL=<optional — override critic model>
```

Default critic model: `openrouter/google/gemma-4-31b-it:free`

---

## 4. Locked Schemas

These schemas are **immutable** after Phase 2. Fields can be added; they cannot be renamed or removed. Loop 2 and all log readers depend on this stability.

### 4.1 PatientProfile

Mutable throughout the session. Updated after each patient answer via `profile_updater.py`.

```python
class PatientProfile(BaseModel):
    session_id: str
    demographics: Demographics           # age: int, sex: str, other: dict
    chief_complaint: str
    symptoms: list[Symptom]
    history: History
    ruled_out: list[str]                 # diagnoses ruled out (not denied symptoms)
    ruled_in: list[str]
    free_notes: str
    running_summary: str                 # compressed summary from compressor.py
```

**Symptom**:
```python
class Symptom(BaseModel):
    name: str        # e.g., "chest pain"
    onset: str       # e.g., "3 days ago"
    severity: str    # "mild" | "moderate" | "severe"
    notes: str       # additional detail
```

**History**:
```python
class History(BaseModel):
    medical: list[str]      # chronic conditions, prior diagnoses, past events
    medications: list[str]  # current/past medications
    allergies: list[str]
    family: list[str]       # blood relatives' health
    social: list[str]       # lifestyle ONLY: smoking, alcohol, occupation, diet, exercise
```

**Critical distinction enforced in all prompts**:
- `symptoms` = current presenting complaint
- `history.medical` = past/chronic (NOT current symptoms — this is the most common error)

### 4.2 DoctorTurnOutput

Strict JSON output from the doctor LLM on every turn.

```python
class DoctorTurnOutput(BaseModel):
    turn_index: int
    current_differential: list[DiagnosisEntry]  # [{dx: str, prob: float}]
    biggest_uncertainty: str                    # highest-value missing info
    candidate_questions: list[str]              # min 1 candidate
    chosen_question: str                        # the single best question
    rationale: str                              # why chosen_question beats alternatives
    confidence_to_stop: float                   # 0.0–1.0
    should_stop: bool                           # true if confidence >= threshold
    safety_flags: list[str]                     # emergency interventions needed
```

**DiagnosisEntry**:
```python
class DiagnosisEntry(BaseModel):
    dx: str       # diagnosis name (Title Case, normalized)
    prob: float   # 0.0–1.0
```

### 4.3 TurnRecord

Logged output of one complete Q&A cycle. Written to JSONL before asking for next answer.

```python
class TurnRecord(BaseModel):
    schema_version: str                    # "0.5.0"
    turn_index: int
    doctor_output: DoctorTurnOutput
    patient_answer: str
    retrieved_exemplar_ids: list[str]      # which exemplars were injected
    timestamp: str                         # ISO8601
```

### 4.4 FinalRecord

Emitted once when session terminates. Written to `logs/final_records/final_{uuid}.json`.

```python
class FinalRecord(BaseModel):
    schema_version: str = "0.5.0"
    session_id: str
    started_at: str                        # ISO8601
    ended_at: str                          # ISO8601
    termination_reason: Literal[
        "confidence_threshold",            # normal stop
        "max_turns",                       # hit 15-turn cap
        "safety_stop",                     # emergency phrase detected
        "user_quit"                        # user typed quit
    ]
    final_profile: PatientProfile
    final_differential: list[DiagnosisEntry]
    primary_diagnosis: str                 # final_differential[0].dx
    turn_history: list[TurnRecord]
    model_metadata: ModelMetadata          # model names, prompt template versions
    closing_turn: Optional[ClosingTurn]    # plain-language summary (may be None)
```

### 4.5 Exemplar

Hand-curated clinical example, stored as individual JSON files in `exemplars/`.

```python
class Exemplar(BaseModel):
    exemplar_id: str                       # "ex_appendicitis_001"
    tags: list[str]                        # ["abdominal_pain", "emergency", ...]
    context_summary: str                   # scenario description (what gets embedded)
    good_next_question: str                # the strong question, natural spoken English
    rationale: str                         # why this question at this moment
    frozen: bool                           # true = never evicted from pool
    embedding: Optional[list[float]]       # 384-dim vector, populated by embed_exemplars.py
```

**Naming convention**: `ex_[condition_snake_case]_[3-digit-number].json`  
Examples: `ex_dvt_from_trauma_001`, `ex_ssnhl_001`, `ex_appendicitis_001`

---

## 5. Loop 1 — Core Diagnostic System

### 5.1 LLM Interface (`src/loop1/llm.py`)

All LLM calls go through here. Handles multi-provider routing, fallbacks, and JSON parsing.

**Key functions**:

```python
def call_llm(model: str, messages: list[dict], **kwargs) -> str
    # Returns content string only

def call_llm_with_usage(model: str, messages: list[dict], **kwargs) -> tuple[str, int]
    # Returns (content, prompt_tokens) — used by doctor.py for token logging

def call_llm_json(model: str, messages: list[dict], max_parse_retries: int = 3, **kwargs) -> dict
    # Parses JSON, feeds parse errors back to LLM on failure, retries up to 3x
```

**Fallback chain** (triggered automatically on 429 rate limit):
```
groq/llama-3.3-70b-versatile
    → groq/meta-llama/llama-4-scout-17b-16e-instruct
    → openrouter/[free models]
    → cerebras/[models]
    → gemini/[models]
```

No blocking sleeps. On 429, immediately tries next provider. On 5xx, exponential backoff via `tenacity`.

### 5.2 Doctor Turn Generation (`src/loop1/doctor.py`)

Generates one complete doctor turn: retrieves exemplars, builds prompt, calls LLM, validates output.

**Primary function**:
```python
def generate_turn_with_usage(
    profile: PatientProfile,
    history: list[TurnRecord] | None = None,
    turn_index: int = 0,
    max_retries: int = 3,
    forced_exemplars: list[Exemplar] | None = None,
) -> tuple[DoctorTurnOutput, int, list[str]]:
    # Returns (turn_output, prompt_tokens_used, exemplar_ids_injected)
```

**Per-turn flow**:
1. Build query from patient profile → embed → retrieve 3 exemplars via MMR
2. Load prompt template from `prompts/doctor_v{version}.txt`
3. Assemble messages: system (prompt + exemplars) + user (profile + history)
4. Call LLM → parse JSON → validate against `DoctorTurnOutput`
5. On failure: feed error back, retry up to `max_retries` times

**Doctor prompt v0.6 key rules** (from `prompts/doctor_v0_6.txt`):

| Rule | Detail |
|------|--------|
| Base-rate calibration (turn 0) | Common things ≥ 0.55 total prob; cancers/sepsis ≤ 0.15 without red flags |
| Mandatory differentials | Chest pain → include ACS + PE + pericarditis. SOB → PE + pneumonia + HF + asthma. Joint pain → SLE + RA. Severe headache → SAH + meningitis |
| Confidence anchors | 0.0–0.2 uncertain; 0.3–0.5 working hypothesis; 0.6–0.75 strong lead; 0.8+ stop |
| Differential updating | Prior differential is starting point. Confirmed → raise ≥ 0.10. Pathognomonic denial → remove. Generic denial → lower ≥ 0.10 |
| Systemic screening | For joint/muscle pain: ask multi-joint question at turn 1–2 before narrowing |
| Exemplar usage | Apply reasoning *pattern*, never copy question text verbatim |
| Output format | Strict JSON only. No markdown fences, no preamble |
| Language | Plain English: "blood clot" not "DVT"; "ECG" not "electrocardiogram" |

### 5.3 Exemplar Retrieval (`src/loop1/retrieval.py`)

Manages the FAISS index and exemplar selection.

**Embedding model**: `all-MiniLM-L6-v2` (384-dim, lazy-loaded via fastembed)

**Patient state encoding**:
```python
def embed_patient_state(profile: PatientProfile) -> np.ndarray:
    # Concatenates: chief_complaint + running_summary + top 5 symptom names
    # Returns unit-normalized 384-d vector
```

**Selection strategies**:

```python
def select_random(pool, n=3, rng=None) -> list[Exemplar]
    # Phase 4 baseline; rng can be seeded for replay determinism

def select_mmr(pool, query_vec, n=3, lambda_=0.7) -> list[Exemplar]
    # Phase 5+ default
    # lambda_: weight on similarity (0=pure diversity, 1=pure similarity)
    # Algorithm: iteratively pick exemplar maximizing λ·sim(q,e) - (1-λ)·max_sim(e, selected)
```

**FAISS index**: Built at startup from pre-computed embeddings in exemplar JSON files. Flat inner-product index — works as cosine similarity because all vectors are unit-normalized. Requires running `scripts/embed_exemplars.py` before the system can retrieve exemplars.

**Public entry point**:
```python
def get_exemplars_for_profile(profile, n=3, lambda_=0.7) -> list[Exemplar]
```

### 5.4 Context Compression (`src/loop1/compressor.py`)

Prevents prompt bloat by summarizing older turns. Called automatically at turn 4+ when `len(history) > keep_recent`.

**Two-stage compression**:

**Stage 1 — Code-level deduplication** (no LLM, fast):

*Symptom merging (4 passes)*:
1. Exact name match (case-insensitive) → merge notes, keep first
2. Substring containment → "left-sided chest pain" absorbed by "chest pain"
3. Content-word subset → "leg swelling" and "swollen legs" merged
4. Jaccard similarity > 80% on content words → merged

*History lists*: Case-insensitive dedup, preserve first occurrence.

*Free notes*: Split on `.` / `\n`, dedup by lowercase sentence key.

**Stage 2 — LLM recategorization**:
```python
def _llm_recategorize_and_summarize(
    profile: PatientProfile,
    turns_to_summarize: list[TurnRecord],
) -> tuple[History, str]:
    # Fixes mis-categorized items (symptoms incorrectly in history.medical)
    # Generates running_summary: 2-4 sentences of confirmed symptoms, timeline, ruled-in/out
    # Anti-hallucination: "extract only, never infer"
```

**Result**: `profile.running_summary` is updated; old turns are no longer passed raw. Last 3 turns always passed verbatim.

### 5.5 Profile Updater (`src/loop1/profile_updater.py`)

After each patient answer, extracts new structured information into a `ProfileDelta`.

```python
def extract_profile_delta(
    profile: PatientProfile,
    question: str,
    answer: str,
) -> ProfileDelta
```

**ProfileDelta**:
```python
class ProfileDelta(BaseModel):
    new_symptoms: list[Symptom]
    symptom_updates: list[SymptomUpdate]         # updates to existing symptoms
    history_additions: HistoryAdditions           # medical/meds/allergies/family/social
    append_ruled_out: list[str]                   # diagnoses only, not denied symptoms
    append_ruled_in: list[str]
    free_notes_append: str
```

**Critical extraction rules** (enforced in `prompts/profile_updater_v0_3.txt`):
- Never infer — only extract what patient explicitly stated
- Check existing symptoms before adding new ones; update rather than duplicate
- Denied symptoms do NOT create entries (neither in symptoms nor ruled_out)
- `ruled_out` is for diagnoses only, never for denied symptoms
- `social` history = lifestyle only (smoking, alcohol, occupation, diet, exercise)

Applied via `apply_delta(profile, delta)` → returns updated `PatientProfile` without mutating original.

### 5.6 Session Loop (`src/loop1/session.py`)

Main orchestrator. Drives the multi-turn dialogue from start to `FinalRecord`.

**Class**: `Session`

```python
session = Session(
    profile=initial_profile,
    max_turns=15,
    confidence_threshold=0.55,
    patient_source=None  # None = interactive input; or SimulatedPatient
)
final_record = session.run()
```

**Turn loop (per turn)**:
1. Call `doctor.generate_turn_with_usage()` → get `DoctorTurnOutput`
2. Check `safety_flags` → terminate if triggered
3. Check `should_stop` / `confidence_to_stop` → terminate if met
4. Display question, collect patient answer (from `patient_source` or `input()`)
5. Check patient answer for emergency phrases → terminate if matched
6. Log `TurnRecord` to JSONL
7. Call `profile_updater.extract_profile_delta()` → apply delta to profile
8. If `len(history) > keep_recent`: call `compressor.compress_context()`
9. Continue to next turn

**Post-loop**:
- Call `closing_turn.generate_closing_turn()` → patient-facing summary
- Assemble `FinalRecord`
- Write to `logs/final_records/final_{uuid}.json`

### 5.7 Safety Layer (`src/loop1/safety.py`)

Detects emergency phrases in patient input and terminates the session.

**Trigger phrases**:
```
can't breathe, cannot breathe, face drooping, unresponsive,
severe bleeding, suicidal, kill myself, throat swelling, seizure, overdose
```

```python
def check_safety(patient_input: str) -> str | None
    # Returns matched phrase, or None if safe
```

On match: logs safety event, displays "Please call emergency services" message, sets `termination_reason = "safety_stop"`.

### 5.8 Closing Turn (`src/loop1/closing_turn.py`)

Generates a plain-language summary for the patient at session end.

```python
def generate_closing_turn(
    profile: PatientProfile,
    final_differential: list[DiagnosisEntry],
) -> ClosingTurn | None:    # None if LLM call fails — session continues gracefully
```

**ClosingTurn fields**:
- `leading_diagnosis`: one sentence, jargon-free (no Latin, no acronyms)
- `differential_summary`: 2–3 candidate conditions (no percentages)
- `recommended_next_steps`: concrete action list
- `unresolved_patient_questions`: questions patient asked but weren't fully addressed

### 5.9 Logging (`src/loop1/logging_utils.py`)

**Session JSONL** (`logs/sessions/session_{uuid}.jsonl`):
- One JSON object per line
- Event types: `session_start`, `turn_complete`, `compression_complete`, `safety_event`, `session_end`, `closing_turn_complete`
- Written incrementally — crash-safe

**Final record** (`logs/final_records/final_{uuid}.json`):
- Complete `FinalRecord` as pretty-printed JSON
- Written once at session end

```python
def log_event(session_id: str, event_type: str, data: dict) -> None
def write_final_record(session_id: str, record: dict) -> Path
def read_session_events(session_id: str) -> list[dict]    # for replay, critic, eval
```

---

## 6. Loop 2 — Evaluation & Critic

### 6.1 Patient Simulator (`src/loop2/ddxplus/patient_simulator.py`)

Simulates a patient grounded in a DDXPlus structured record. Used in Phase 7 eval to run sessions without real patients.

**Class**: `SimulatedPatient`

```python
class SimulatedPatient:
    def initial_complaint(self) -> str
        # Returns opening statement from patient record

    def answer(self, doctor_question: str) -> str
        # LLM call grounded in DDXPlus record
        # Temperature 0 for reproducibility
        # Tracks disclosed_symptoms to stay consistent
```

**Guardrails**:
1. **Truthfulness**: Never invents symptoms not in the DDXPlus record
2. **Leakage detection**: Post-hoc scan of response for tokens from `ground_truth_pathology`. On match → regenerate (up to 3 retries) → fallback to generic safe response
3. **Consistency**: Tracks `disclosed_symptoms` to avoid contradictions across turns

### 6.2 Critic LLM (`src/loop2/critic/critic.py`, `critique_schema.py`)

Scores doctor question quality after each turn. Called post-session, not in real time.

```python
def critique_turn(
    turn_event: dict,           # one event from session JSONL
    session_events: list[dict], # all session events (for context)
    session_id: str,
) -> TurnCritique
```

**TurnCritique schema**:
```python
class TurnCritique(BaseModel):
    schema_version: str = "0.7.0"
    session_id: str
    turn: int
    question_quality_score: float           # 0.0–1.0
    differential_quality_score: float       # 0.0–1.0
    reasoning_quality_score: float          # 0.0–1.0
    confidence_calibration: Literal[
        "well-calibrated", "overconfident", "underconfident"
    ] | None
    weakness: str | None                    # description of the issue
    weakness_category: Literal[
        "redundant_question", "missed_red_flag", "poor_differential",
        "premature_confidence", "other"
    ] | None
    would_have_asked: str | None            # suggested better question
    rationale: str                          # scoring explanation
    critic_model: str
```

**What the critic does NOT receive**: ground truth diagnosis or the patient's answer. It evaluates the doctor's question in isolation, before seeing the response.

**Critic model**: Configurable via `CRITIC_MODEL` env var. Default: `openrouter/google/gemma-4-31b-it:free`

### 6.3 Ground-Truth Evaluator (`src/loop2/ddxplus/evaluator.py`)

Compares doctor's final diagnosis against DDXPlus ground truth. No LLM calls.

```python
def evaluate_session(
    session_jsonl_path: str,
    ddxplus_record: DDXPlusPatient,
) -> DDXPlusEvalResult
```

**DDXPlusEvalResult**:
```python
class DDXPlusEvalResult(BaseModel):
    schema_version: str = "0.7.0"
    patient_id: str
    session_id: str
    leading_diagnosis_correct: bool        # top-1 matches ground truth
    top3_contains_truth: bool
    differential_overlap: float            # Jaccard overlap
    turns_to_correct_diagnosis: int | None # first turn where truth was top-1
    final_confidence: float
    terminated_on_confidence: bool
    total_turns: int
    in_exemplar_pool: bool                 # was this disease in the exemplar bank?
```

### 6.4 Aggregator (`src/loop2/critic/aggregator.py`)

Rolls up critique results across sessions:
- Mean/median scores per metric
- Distribution of `confidence_calibration` labels
- Most common `weakness_category` patterns
- Correlation between low `question_quality_score` and diagnostic failure

### 6.5 Phase 7 Eval Runner (`src/loop2/runners/phase7_eval.py`)

End-to-end eval on 20 DDXPlus patients:

1. For each patient: `SimulatedPatient` → `Session.run()` → `DDXPlusEvalResult` → per-turn `TurnCritique`
2. After all 20: `aggregator` → summary stats
3. Output: `phase7_findings.md`

**Resumable**: Skips patients that already have a completed JSONL log.

---

## 7. Loop 3 — Routing & Specialist Assignment

### 7.1 Routing Schemas (`src/loop3/routing/routing_schema.py`)

```python
class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"    # life-threatening: ED or 999
    URGENT = "urgent"          # within 24–48h
    ROUTINE = "routine"        # scheduled appointment

class AppointmentType(str, Enum):
    EMERGENCY = "emergency_visit"
    SAME_DAY = "same_day_appointment"
    URGENT = "urgent_appointment_within_48h"
    ROUTINE = "routine_appointment"
    IN_PERSON = "in_person"
    TELEHEALTH = "telehealth_consultation"

class RoutingOption(BaseModel):
    specialty: str                  # "Cardiology", "Neurology", etc.
    urgency: UrgencyLevel
    reasoning: str                  # one sentence
    appointment_type: AppointmentType
    confidence_weight: float        # probability mass from differential

class RoutingDecision(BaseModel):
    schema_version: str = "0.9.0"
    session_id: str
    patient_id: Optional[str] = None
    is_ambiguous: bool
    primary_routing: RoutingOption  # always present; highest urgency first
    alternative_routings: list[RoutingOption]  # empty if unambiguous
    final_differential: list[tuple[str, float]]
    final_confidence: float
    routing_model_version: str      # e.g., "disease_specialty_map_v1"
```

### 7.2 Disease → Specialty Map (`data/disease_specialty_map.json`)

70+ DDXPlus conditions mapped to clinical specialties.

**Entry format**:
```json
"Pulmonary embolism": {
  "specialty": "Pulmonology",
  "urgency": "emergency",
  "appointment_type": "emergency_visit",
  "reasoning": "PE is life-threatening, requires immediate anticoagulation"
}
```

**Covered specialties**: Cardiology, Pulmonology, Neurology, Gastroenterology, Infectious Disease, Rheumatology, Hematology, Orthopedics, Dermatology, Urology, Nephrology, Endocrinology, Emergency Medicine, General Surgery, Internal Medicine.

**Urgency rules**:
- Emergency diseases (PE, MI, appendicitis, SAH): always `emergency`
- Urgent diseases (pneumonia, DVT, febrile UTI): always `urgent`
- Routine diseases (viral pharyngitis, GERD): always `routine`
- **Confidence adjustment**: if `final_confidence < 0.4` and base urgency is `routine` → bumped to `urgent`

### 7.3 Ambiguity Handler (`src/loop3/routing/ambiguity_handler.py`)

```python
def is_ambiguous(differential, specialty_map) -> bool
    # True if top specialty's prob-weighted score < 1.5x second specialty's score

def get_routing_options(differential, specialty_map, final_confidence, max_options=3)
    -> list[RoutingOption]
    # 1. Map each diagnosis → specialty
    # 2. Accumulate probability weight per specialty
    # 3. Sort by total weight descending
    # 4. Return top N (2–3 if ambiguous, 1 if not)
```

**Safety override**: If *any* disease in the differential is emergency-level, it becomes `primary_routing` regardless of probability weight. A 20% PE always surfaces above an 80% musculoskeletal strain.

### 7.4 Main Router (`src/loop3/routing/router.py`)

```python
def route(
    session_id: str,
    final_differential: list[tuple[str, float]],
    final_confidence: float,
    patient_id: str | None = None,
) -> RoutingDecision
```

**Orchestration**:
1. Load specialty map
2. Get routing options (ambiguity detection + aggregation)
3. Classify urgency per option
4. Rank: highest urgency first → then by probability weight
5. Assemble and return `RoutingDecision`

---

## 8. Web API & Frontend

### 8.1 FastAPI Backend (`web/api.py`)

**Tech**: FastAPI + uvicorn, CORS middleware, no-cache on static files.  
**Auth**: JWT tokens in `localStorage`, sent as `Authorization: Bearer <token>`.  
**Session state**: In-memory `dict[str, APISession]` — lost on server restart.

**All endpoints**:

```
# Auth
POST /api/register                    → patient sign-up
POST /api/login                       → login (returns JWT + role)
GET  /api/me                          → current user info

# Cases
GET  /api/cases                       → list 6 predefined cases
GET  /api/cases/{case_id}             → single case JSON

# Sessions
POST /api/session/start               → start from predefined case
POST /api/session/start-custom        → start from custom intake form
POST /api/session/{id}/turn           → submit patient answer → get next doctor turn
GET  /api/session/{id}/report         → FinalRecord JSON when session complete
GET  /api/sessions                    → patient's session history

# Routing & Booking (Phase 10)
GET  /api/session/{id}/routing        → RoutingDecision + matching doctors list
POST /api/session/{id}/book           → book appointment (writes to appointments.json)
GET  /api/appointments                → patient's booked appointments

# Doctor Portal (Phase 10)
GET  /api/doctor/appointments         → doctor's appointment queue (doctor role only)
GET  /api/doctor/appointments/{id}    → full session detail for one appointment
```

### 8.2 APISession (`web/api_session.py`)

Wraps `Session` for stateful web use (one request at a time, across HTTP boundaries).

```python
class APISession:
    def next_turn(self, patient_answer: str) -> DoctorTurnOutput
    def get_final_record(self) -> FinalRecord | None
```

### 8.3 Frontend Pages

All pages use vanilla HTML/CSS/JS (no frameworks).  
**Dark theme**: Background `#0f1117`, surfaces `#1a1f2e`, accent teal `#00b4d8`.

| Page | File | Purpose |
|------|------|---------|
| Login | `login.html` | Sign in / register. Checks `role` → redirects patient to `index.html`, doctor to `doctor-portal.html` |
| Case grid | `index.html` | 6 predefined cases + "start custom" button |
| Custom intake | `patient-info.html` | Collects chief complaint, demographics, symptoms |
| Live session | `session.html` | Chat UI: doctor question + patient answer input. Shows differential updating per turn |
| Report | `report.html` | Final diagnosis, differential chart, critic scores, specialist routing card, "Book Appointment" |
| Doctor portal | `doctor-portal.html` | Appointment sidebar + session detail (transcript, differential timeline, critic scores table) |
| History | `history.html` | Patient's past sessions list |
| Auth helpers | `auth.js` | JWT store/retrieve, inject into request headers, redirect if expired |

### 8.4 Mock Data Files (`web/data/`)

**`doctors.json`**:
```json
{
  "id": "dr_001",
  "name": "Dr. Sarah Chen",
  "specialty": "Cardiology",
  "hospital": "City Heart Centre",
  "rating": 4.8,
  "avatar_initials": "SC",
  "available_slots": ["2026-06-17T09:00", "2026-06-17T14:00"]
}
```

**`appointments.json`**:
```json
{
  "appt_uuid": {
    "appointment_id": "appt_uuid",
    "session_id": "...",
    "patient_user_id": "...",
    "doctor_id": "dr_001",
    "doctor_name": "Dr. Sarah Chen",
    "specialty": "Cardiology",
    "slot": "2026-06-17T09:00",
    "primary_diagnosis": "Possible NSTEMI",
    "urgency": "urgent"
  }
}
```

---

## 9. Prompts (Versioned)

All prompts are versioned and treated as immutable once deployed. Old logs reference the version they were generated with — this is essential for replay and comparison.

| File | Version | Role |
|------|---------|------|
| `doctor_v0_6.txt` | v0.6 | Current doctor prompt (active) |
| `doctor_v0_{1-5}.txt` | v0.1–v0.5 | Archived (for replay of old sessions) |
| `compressor_v0_2.txt` | v0.2 | Context compression + running_summary generation |
| `profile_updater_v0_3.txt` | v0.3 | Profile delta extraction from Q&A |
| `critic_turn_v0_1.txt` | v0.1 | Turn-level critic scoring |
| `simulated_patient_v0_1.txt` | v0.1 | DDXPlus-grounded patient simulator |

**Doctor prompt key changes across versions**:
- v0.1: Basic role + output schema
- v0.2: Added CoT rationale field
- v0.3: Added confidence calibration anchors
- v0.4: Added mandatory differential rules (ACS+PE for chest pain, etc.)
- v0.5: Added systemic screening rule for joint pain
- v0.6: Added base-rate calibration + evidence-based differential updating rules

---

## 10. Exemplars & Test Cases

### 10.1 Exemplar Pool (`exemplars/`)

~50 JSON files covering diverse clinical scenarios. All have `frozen: true`.

**Current exemplars include**:
- `ex_appendicitis_001` — localization question for migratory RLQ pain
- `ex_biliary_colic_001` — colicky pain characterization
- `ex_cauda_equina_screen_001` — red-flag screening for back pain
- `ex_cvst_001` — cerebral venous sinus thrombosis workup
- `ex_dvt_from_trauma_001` — DVT risk factor probe after injury
- `ex_inflammatory_back_001` — distinguishing inflammatory vs mechanical back pain
- `ex_measles_001` — pediatric travel + contact history
- `ex_patient_question_handling_001` — handling when patient asks a question back
- `ex_peds_travel_contact_001` — pediatric infectious workup
- `ex_pericarditis_001` — pleuritic chest pain characterization
- `ex_pregnancy_001` — pregnancy possibility screen
- `ex_pulmonary_embolism_001` — PE risk factor probe
- `ex_sle_screen_001` — multi-system autoimmune screening
- `ex_stimulant_acs_001` — ACS in young adult with stimulant use
- `ex_cellulitis_001` — skin infection vs DVT differentiation
- ENT exemplars (in progress)

**Embedding requirement**: All exemplars must have their `embedding` field populated before retrieval works. Run `scripts/embed_exemplars.py` after adding new exemplars.

### 10.2 Adding New Exemplars

1. Create `exemplars/ex_[condition]_001.json` with `"embedding": null`
2. Fill in all fields (see schema above)
3. Run: `.venv/bin/python scripts/embed_exemplars.py`

The script is idempotent — skips files that already have embeddings.

### 10.3 Test Cases (`test_cases/`)

6 hand-written patient profiles used for interactive testing:
- `case_01.json` through `case_06.json`
- Cover headache, chest pain, abdominal pain, rash, joint pain, shortness of breath

---

## 11. Data Files & Datasets

### 11.1 DDXPlus Dataset (`data/DDXPlus_Raw/`)

GITIGNORED — must be downloaded separately.  
Contains 70+ conditions, symptom definitions, and thousands of synthetic patient records.

Key files:
- `release_conditions.json` — disease metadata
- `release_evidences.json` — symptom/evidence definitions and value types
- `release_train_patients`, `release_validate_patients`, `release_test_patients`

**Patient record format**:
- `age`, `sex`, `initial_evidence` (chief complaint)
- `symptoms: dict[str, bool | str]` — all symptom values
- `antecedents: dict[str, bool]` — history
- `ground_truth_pathology: str` — correct diagnosis

### 11.2 DDXPlus Eval Set (`data/ddxplus_eval_set.json`)

Committed. 20 hand-curated patient IDs, stratified:
- 5 patients with diseases covered by exemplar pool (baseline)
- 10 patients with diseases NOT in exemplar pool (coverage stress test)
- 5 patients with ambiguous differentials (discrimination test)

### 11.3 Disease Specialty Map (`data/disease_specialty_map.json`)

70+ entries. Also duplicated at `web/data/disease_specialty_map.json` for the API.

---

## 12. Scripts & Utilities

| Script | Purpose |
|--------|---------|
| `hello_world.py` | Sanity check: config + one LLM call + one log event |
| `run_session.py` | Interactive CLI session: `--profile test_cases/case_02.json` |
| `run_single_turn.py` | Single turn test for prompt iteration |
| `replay_session.py` | Deterministic replay of logged session (temperature=0, seeded RNG) |
| `embed_exemplars.py` | Populate `embedding` field in all exemplar JSON files |
| `eval_retrieval.py` | Verify MMR retrieval quality on sample queries |
| `curate_eval_set.py` | Interactive helper to select 20 DDXPlus eval patients |
| `diagnose_confidence.py` | Analyze confidence plateau patterns across sessions |
| `validate_confidence.py` | Check confidence calibration on test set |
| `bake_off.py` | Compare two prompt versions on same test cases |

**Running scripts** (use the project venv, not system Python):
```bash
# Option 1: direct venv
.venv/bin/python scripts/embed_exemplars.py

# Option 2: uv (requires Python 3.11.0 in managed installs)
uv run python scripts/embed_exemplars.py
```

---

## 13. Phase Progression

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Setup, schemas, hello world | ✅ Complete |
| 1 | Single-turn sanity check | ✅ Complete |
| 2 | Multi-turn loop + logging | ✅ Complete |
| 3 | Context compression | ✅ Complete |
| 4 | Static ICL exemplars (random selection) | ✅ Complete |
| 5 | Dynamic FAISS retrieval + MMR | ✅ Complete |
| 6 | Hardening: replay mode, safety layer, schema versioning | ✅ Complete |
| 7 | Turn-level critic + DDXPlus evaluation (N=20) | ✅ Complete |
| 8 | *(skipped — reserved for exemplar mining)* | — |
| 9 | Disease → specialty routing (Loop 3) | ✅ Complete |
| 10 | Specialist routing UI + doctor portal (web) | 🔄 In Progress |

**Deferred to future loops**:
- DPO/LoRA fine-tuning (explicitly out of Loop 1 scope)
- Critic-in-the-loop at inference time
- Exemplar mining from high-scoring critic sessions
- Real provider/clinic booking integration
- Multimodal input (images, audio, EHR)

---

## 14. Deployment & Environment

### 14.1 Local Setup

```bash
# Install dependencies
uv sync

# Set environment variables
cp .env.example .env   # then add GROQ_API_KEY

# Embed exemplars (required once, or after adding new ones)
.venv/bin/python scripts/embed_exemplars.py

# Start web server
uv run python -m uvicorn web.api:app --reload --host 0.0.0.0 --port 8000
# Access: http://localhost:8000
```

### 14.2 Heroku Deployment

```
# Procfile
web: gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker web.api:app
```

Note: In-memory session state is lost on dyno restart/sleep. Users mid-session will need to start over.

### 14.3 Run Tests

```bash
uv run pytest
```

---

## 15. Key Architectural Decisions

### Schema Versioning Discipline

Every log entry carries `schema_version`. Schemas are additive-only — no renames, no removals. This guarantees that Phase 7 eval, critic, and replay can always read logs written in any earlier phase.

### Exemplar Pool as a Living Document

The `frozen: true` seed set is permanent. Future phases (Loop 2+) may add machine-generated exemplars with `frozen: false`, which can be evicted based on performance metrics.

### Prompt Immutability After Deployment

Every prompt version is kept on disk permanently. Old sessions reference the prompt version used to generate them, enabling exact replay and A/B comparison across versions.

### LLM Resilience Without Blocking

The fallback chain has no `sleep()` calls. Rate limit → immediate next provider. This keeps sessions responsive at the cost of occasionally using a weaker fallback model. The fallback model is logged in `model_metadata` so it's auditable.

### Context Compression Tradeoff

Passing full turn history beyond turn ~8 causes context confusion and prompt bloat. The 3-turn verbatim window + `running_summary` approach trades some detail for coherence. The compression is logged separately so it can be audited.

### Safety-First Routing

Urgency wins over probability. A differential where PE has only 20% probability still routes to Pulmonology emergency as the primary option. This is intentional: false negatives for emergencies are far more costly than false positives.

---

## 16. Common Workflows

### Run an Interactive Session (Terminal)

```bash
.venv/bin/python scripts/run_session.py --profile test_cases/case_02.json
# Type patient answers when prompted
# Session ends at confidence threshold or 15 turns
# Final record written to logs/final_records/
```

### Replay a Logged Session

```bash
.venv/bin/python scripts/replay_session.py --session-id <uuid>
# Deterministic: temperature=0, seeded RNG
# Useful for prompt iteration without real patient input
```

### Add New Exemplars

```bash
# 1. Create the JSON file
# exemplars/ex_[condition]_001.json (with "embedding": null)

# 2. Populate embeddings
.venv/bin/python scripts/embed_exemplars.py
# Output: "EMBED ex_[condition]_001 (384d)"
```

### Run Phase 7 Evaluation

```bash
.venv/bin/python -m loop2.runners.phase7_eval
# Runs 20 simulated sessions + critic scoring
# Outputs phase7_findings.md
```

### Extract Routing from a Completed Session

```python
import json
from loop3.routing.router import route

record = json.load(open("logs/final_records/final_<uuid>.json"))
decision = route(
    session_id=record["session_id"],
    final_differential=[(d["dx"], d["prob"]) for d in record["final_differential"]],
    final_confidence=record["final_differential"][0]["prob"],
)
print(decision.primary_routing.specialty, decision.primary_routing.urgency)
```

### Web Interaction (Patient Flow)

```
1. http://localhost:8000 → login.html
2. Register or log in as patient
3. Pick predefined case OR fill custom intake form
4. Answer doctor's questions in session.html
5. Session completes → redirected to report.html
6. View diagnosis, critic scores, recommended specialist
7. Book appointment → slot saved to appointments.json
```

### Web Interaction (Doctor Flow)

```
1. Log in with doctor account
2. Redirected to doctor-portal.html
3. See appointment queue in sidebar
4. Click appointment → full session detail panel:
   - Patient demographics
   - Differential evolution per turn
   - Conversation transcript
   - Critic scores table
```

---

## 17. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `GROQ_API_KEY not set` | Missing env var | Add to `.env` in loop1/ |
| FAISS build error at startup | Exemplars missing embeddings | Run `embed_exemplars.py` |
| `ImportError: PreTrainedModel from transformers` | Wrong Python env (base conda) | Use `.venv/bin/python`, not `python` |
| `No interpreter found for Python 3.11.0` | uv managed installs | Use `.venv/bin/python` directly |
| Confidence never crosses threshold | Model sandbagging or prompt issue | Lower `confidence_to_stop` to 0.4 temporarily; check differential is updating |
| Doctor asks same question twice | Exemplar retrieval not working, or history not in prompt | Run `eval_retrieval.py`; verify `turn_history` is included in prompt |
| Profile not updating | Profile delta extraction failing | Test `extract_profile_delta()` in isolation; check prompt version |
| Symptoms appearing in `history.medical` | Common LLM categorization error | Check compressor prompt v0.2; reinforce active vs. past distinction |
| Session hits max_turns every time | Expected if confidence doesn't converge | Check `termination_reason` in logs; `max_turns` is the safety net |
| Replay is non-deterministic | Temperature not 0, or MMR tie-breaking unseeded | Ensure `temperature=0` and `rng=np.random.default_rng(seed)` in retrieval |
| Web session lost on restart | In-memory state only | Expected behavior; users must restart session |
| Critic scores unavailable in portal | Critic runs post-session; may not have run yet | Run `phase7_eval.py` or trigger critique manually |

---

## 18. Glossary

| Term | Definition |
|------|-----------|
| **Chief complaint** | Patient's primary reason for the visit ("chest pain", "fever") |
| **Differential** | Ranked list of candidate diagnoses with probabilities |
| **Exemplar** | Hand-curated Q&A clinical example showing strong diagnostic reasoning at a specific moment |
| **MMR** | Maximal Marginal Relevance — retrieval algorithm balancing similarity to query vs. diversity among results |
| **Profile delta** | Structured extraction of new information from a single patient answer |
| **Running summary** | LLM-compressed narrative of dialogue turns 0 to N-3, stored in `profile.running_summary` |
| **Turn** | One Q&A cycle: doctor asks, patient answers |
| **TurnRecord** | Logged output of one turn: question, answer, differential, exemplar IDs, timestamp |
| **Context compression** | Replacing raw old turns with a running summary to reduce prompt token count |
| **Safety flag** | Emergency phrase detected in patient input → immediate session termination |
| **Confidence threshold** | Doctor's `confidence_to_stop` value at which to stop questioning (default 0.55) |
| **Frozen exemplar** | Seed-set exemplar preserved permanently; never evicted even when Loop 2 adds new ones |
| **Routing** | Mapping final diagnosis to a medical specialty + urgency level + appointment type |
| **Ambiguous routing** | Differential splits across multiple specialties with similar weight; returns 2–3 options |
| **Leakage** | Patient simulator accidentally revealing ground-truth diagnosis (detected and regenerated) |
| **DDXPlus** | Synthetic medical dataset of 70+ conditions used for evaluation (Phase 7) |
| **Schema version** | Version field on every log entry ensuring forward-compatible parsing |
| **Prompt version** | Filename suffix (v0.6) identifying which prompt template generated a session |
| **Base-rate calibration** | Turn 0 rule: common diseases must have high prior probability before being suppressed by questioning |
| **Evidence-based update** | Rule: differential probabilities must shift by ≥ 0.10 per confirmed/denied finding |
