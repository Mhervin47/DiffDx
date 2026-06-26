# Phase 7 Notes — Turn-Level Critic + Minimal DDXPlus Eval

## Context

Phase 6 complete. Loop 1 is hardened: closing turn, schema_version on all JSONL records, replay determinism, safety substring matching, confidence plateau diagnosed, symptom categorization bug fixed, long-session dedup. All Phase 5 + Phase 6 tests passing. Phase 6 tagged `phase6-complete`.

Phase 7 starts the Loop 2 work, but **scoped small**. Two parallel deliverables:

1. **Turn-level critic LLM** — reads logged Loop 1 sessions and rates the doctor's questioning quality per turn
2. **Minimal DDXPlus eval** — runs the existing Loop 1 system against 20 hand-curated DDXPlus patients with ground truth

Both produce structured JSONL output. Both feed into Phase 8 decisions (expand exemplars, iterate prompt, build bigger critic, or stop).

This is **measurement infrastructure plus a small first critic**, not a fully built-out actor-critic training loop. DPO is out of scope. Model fine-tuning is out of scope. This phase produces signal, not weight updates.

---

## Project end-state context (for Phase 8+ awareness)

The project's eventual goal is a triage system: patient describes symptoms → diagnostic dialogue → disease identification → routing to appropriate medical specialty → booking. Phase 7's diagnostic-reliability measurements are what enables the routing layer (Phase 9+) to be trustworthy. Calibration matters because overconfident wrong diagnoses produce wrong routing — which is why a `confidence_calibration` field is included in the critic schema below.

Do not build routing or booking logic in Phase 7. That's later.

---

## What's In Scope

1. DDXPlus loader
2. Hand-curated 20-patient eval set
3. LLM-based patient simulator with guardrails
4. Ground-truth evaluator
5. Turn-level critic LLM + prompt
6. Critique aggregator
7. Phase 7 runner (integrated pipeline)
8. Findings report

---

## Settled Design Decisions

### 1. Critic model — Gemini 2.0 Flash via Google AI Studio (free tier)

Critic-as-judge depends on the critic being at least as smart as the actor (the Llama-3.3-70b doctor). Gemini 2.0 Flash is capable and available free via Google AI Studio.

Default: **`gemini/gemini-2.0-flash`** via Google AI Studio API (litellm wrapper, same retry/logging pattern as the doctor). API key stored as `GEMINI_API_KEY` in `.env`. Critic model configurable via `CRITIC_MODEL` env var for future upgrades (e.g. `gemini/gemini-2.5-flash` or `gemini/gemini-1.5-flash`).

### 2. Critic scope — per-turn only

Per-session and comparative critique are out of scope. Per-turn is enough to drive Phase 7 conclusions.

### 3. Patient simulator — LLM-based with hard guardrails

- Cheap Groq model (Llama-3.1-8B or similar) as the translator
- LLM receives the DDXPlus patient record as structured JSON in system prompt
- Instructed: "Only answer using fields present in the record. If asked about a symptom not in the record, say 'no' or 'I don't think so.' Never mention the diagnosis or the differential."
- Temperature 0
- Post-hoc check: scan generated response for tokens appearing in `ground_truth_pathology` string; if any match, regenerate
- Disclosed-only behavior: track which symptoms have been revealed to the doctor; don't proactively volunteer unrelated info

### 4. DDXPlus eval set — N=20, hand-curated, stratified

- **5 patients** on diseases already in the exemplar pool (CVST, measles, PE, appendicitis, etc.) — happy-path baseline
- **10 patients** on diseases NOT in the exemplar pool but in DDXPlus — coverage stress test
- **5 patients** with ambiguous/overlapping symptoms — discrimination test

Curated once, saved to `data/ddxplus_eval_set.json`, committed to git. Fixed eval set for Phase 7 and future comparisons.

### 5. Integrated pipeline (critic runs on DDXPlus sessions)

Single runner: each DDXPlus session gets both ground-truth eval AND critic eval. Cross-reference: where does the critic flag bad questions AND the system gets the diagnosis wrong? This is the most interesting data Phase 7 can produce.

### 6. Schema version bump — 0.7.0

New records (`TurnCritique`, `DDXPlusEvalResult`, `SimulatedPatientResponse` if logged) all carry `schema_version: "0.7.0"`.

---

## Schemas

### TurnCritique

```python
class TurnCritique(BaseModel):
    schema_version: str = "0.7.0"
    session_id: str
    turn: int
    question_quality_score: float           # 0-1
    differential_quality_score: float       # 0-1, given context at this turn
    reasoning_quality_score: float          # 0-1
    confidence_calibration: str | None      # "well-calibrated" | "overconfident" | "underconfident"
    weakness: str | None                    # short — what was off
    would_have_asked: str | None            # critic's suggested alternative question
    rationale: str                          # one sentence on why scores are what they are
    critic_model: str                       # for traceability
```

### DDXPlusPatient

```python
class DDXPlusPatient(BaseModel):
    patient_id: str
    age: int
    sex: str
    initial_evidence: str                   # the "chief complaint" analog
    symptoms: dict[str, Any]                # binary/categorical/multi-choice
    antecedents: dict[str, Any]
    ground_truth_pathology: str
    ground_truth_differential: list[tuple[str, float]]
```

### DDXPlusEvalResult

```python
class DDXPlusEvalResult(BaseModel):
    schema_version: str = "0.7.0"
    patient_id: str
    session_id: str
    leading_diagnosis_correct: bool         # top-1 matches ground truth
    top3_contains_truth: bool               # ground truth in doctor's top 3
    differential_overlap: float             # Jaccard overlap with GT differential
    turns_to_correct_diagnosis: int | None  # None if never reached
    final_confidence: float
    terminated_on_confidence: bool
    total_turns: int
    in_exemplar_pool: bool                  # was this disease covered by an exemplar?
```

---

## File Structure

```
src/loop2/                              # new package
├── __init__.py
├── critic/
│   ├── __init__.py
│   ├── critic.py                       # LLM call, retry, schema parsing
│   ├── critique_schema.py              # TurnCritique Pydantic model
│   └── aggregator.py                   # rolls critique JSONL into summary stats
├── ddxplus/
│   ├── __init__.py
│   ├── schemas.py                      # DDXPlusPatient, DDXPlusEvalResult
│   ├── loader.py                       # parses DDXPlus JSON files
│   ├── patient_simulator.py            # LLM-based simulator with guardrails
│   └── evaluator.py                    # compares session to ground truth
└── runners/
    ├── __init__.py
    └── phase7_eval.py                  # runs all 20 patients + critic, produces report

prompts/
├── critic_turn_v0_1.txt                # critic system prompt
└── simulated_patient_v0_1.txt          # patient simulator prompt

data/
├── ddxplus_raw/                        # downloaded dataset, GITIGNORED
└── ddxplus_eval_set.json               # the 20 curated patients, COMMITTED

scripts/
└── curate_eval_set.py                  # interactive helper to build the 20-patient set

tests/
├── test_critic.py
├── test_ddxplus_loader.py
├── test_patient_simulator.py
├── test_evaluator.py
└── test_aggregator.py
```

**Existing Loop 1 code (`src/loop1/`) is NOT modified.** Phase 7 is purely additive.

---

## Build Order

### Step 1 — DDXPlus loader

`src/loop2/ddxplus/loader.py` + `schemas.py`

- Parse DDXPlus JSON into `DDXPlusPatient` Pydantic objects
- Provide `load_ddxplus(path, split="test") -> list[DDXPlusPatient]`
- Handle the symptom hierarchy and antecedent structure

**Pre-step**: confirm `data/ddxplus_raw/` exists and has the expected files. Add `data/ddxplus_raw/` to `.gitignore`.

**Tests**: load 100 patients, verify schema parses cleanly. No malformed records crash the loader.

### Step 2 — Curate the 20-patient eval set

`scripts/curate_eval_set.py`

Interactive (or scripted) helper to pick:
- 5 patients matching exemplar pool diseases
- 10 patients on non-pool diseases
- 5 ambiguous-symptom patients

Output: `data/ddxplus_eval_set.json` with the 20 patient IDs + the curation rationale per patient. COMMIT this file. The dataset itself stays gitignored; the eval set selection is committed.

Exemplar pool diseases (for matching the first 5):
- CVST, measles, PE, appendicitis, cellulitis, pericarditis, biliary colic, cauda equina, DVT, inflammatory back, pregnancy, SLE, stimulant ACS, pediatric travel contact, patient question handling

### Step 3 — Patient simulator

`src/loop2/ddxplus/patient_simulator.py`

```python
class SimulatedPatient:
    def __init__(self, record: DDXPlusPatient, llm_client):
        self.record = record
        self.disclosed_symptoms: set[str] = set()
        self.llm = llm_client

    def initial_complaint(self) -> str:
        """Return a natural-language opening statement.
        Use record.initial_evidence as the primary signal."""

    def answer(self, doctor_question: str) -> str:
        """Given the doctor's question, return a natural-language answer
        consistent with the patient record. Never leak ground truth."""
```

Prompt template: `prompts/simulated_patient_v0_1.txt`

Guardrails:
- Truthful: never invent symptoms not in record
- Disclosed-only: don't volunteer unrelated info
- No leakage: post-hoc scan for ground truth tokens, regenerate if matched
- Temperature 0
- Tracks disclosed_symptoms across the session

**Tests**: 5 hand-checked sessions where the simulator answers consistently and doesn't leak.

### Step 4 — Ground-truth evaluator

`src/loop2/ddxplus/evaluator.py`

```python
def evaluate_session(
    session_jsonl_path: str,
    ddxplus_record: DDXPlusPatient,
) -> DDXPlusEvalResult:
    """Read JSONL output, compare against ground truth, produce metrics."""
```

No LLM calls. Pure comparison logic.

- `leading_diagnosis_correct`: does doctor's final top-1 match `ground_truth_pathology` (case-insensitive, normalized)?
- `top3_contains_truth`: is ground truth pathology in doctor's final top-3?
- `differential_overlap`: Jaccard overlap between doctor's final differential and `ground_truth_differential`
- `turns_to_correct_diagnosis`: first turn at which ground truth pathology appeared as top-1
- `in_exemplar_pool`: lookup against the exemplar pool's covered-disease list

**Tests**: hand-construct 5 fake JSONL sessions with known properties; verify metrics come out correct.

### Step 5 — Critic module

`src/loop2/critic/critic.py` + `critique_schema.py`

```python
def critique_turn(
    turn_record: TurnRecord,
    session_context: SessionContext,  # profile, history up to this turn
) -> TurnCritique:
    """Critique a single doctor turn. One LLM call per turn."""
```

Prompt template: `prompts/critic_turn_v0_1.txt`

The critic prompt receives:
- The doctor's question, reasoning, differential, confidence for this turn
- The patient state at this turn (profile snapshot)
- The conversation history up to (but not including) the current turn
- DOES NOT receive the ground truth or the patient's answer to the doctor's question (the critic is judging the question, not whether the patient revealed something)

Critic model: paid Claude Sonnet by default. Configurable via env var `CRITIC_MODEL` to allow Groq fallback.

**Tests**:
- Critique a hand-constructed "good" turn → all scores > 0.7
- Critique a hand-constructed "bad" turn (generic question when red flag should be screened) → at least one score < 0.5
- Schema parses cleanly on real critic output

### Step 6 — Aggregator

`src/loop2/critic/aggregator.py`

Reads critique JSONL from multiple sessions, produces summary statistics:

- Mean / median question quality across all turns
- Distribution of `confidence_calibration` labels
- Most common weakness patterns (count `weakness` strings, find clusters)
- Correlation between low question_quality_score and downstream diagnostic failure

Pure data crunching. No LLM calls.

### Step 7 — Phase 7 runner

`src/loop2/runners/phase7_eval.py`

Single integrated pipeline:

```
for each of the 20 patients in eval set:
    1. Instantiate SimulatedPatient
    2. Run a Loop 1 session driven by SimulatedPatient (not human input)
       - This means session.py needs to accept a patient source other than terminal
       - Cleanest: add a `patient_source` parameter (terminal vs SimulatedPatient)
       - Or wrap: build a thin adapter that calls into existing session loop
    3. Save JSONL log as usual
    4. Run evaluator → DDXPlusEvalResult
    5. For each turn in the JSONL: run critic → TurnCritique
    6. Save all critique records to a parallel JSONL
After all 20:
    7. Run aggregator on the critique JSONL
    8. Produce a markdown report: phase7_findings.md
```

Resumability: if the runner crashes mid-batch, skip patients that already have completed JSONL output.

### Step 8 — Findings report

Auto-generated `phase7_findings.md` includes:

- Per-disease diagnostic accuracy (broken down: in-pool vs out-of-pool)
- Mean question quality across all critiqued turns
- Confidence calibration distribution
- Cross-reference: which sessions had low question quality AND wrong diagnosis?
- Categorized failure modes (no matching exemplar / right exemplar retrieved but ignored / max-turns plateau / categorization bug recurrence / other)

This report is the Phase 7 deliverable. Hand-read it. Phase 8 decisions follow from it.

---

## Eval Criteria

| Item | Pass condition |
|---|---|
| DDXPlus loader | Loads 100 patients without error; schema parses cleanly |
| Eval set curation | 20 patients selected with documented rationale; committed |
| Patient simulator | 5 hand-checked sessions show consistent, non-leaking responses |
| Evaluator | Correct metrics on 5 hand-constructed fake sessions |
| Critic | Differentiates good vs bad hand-constructed turns; schema-valid output |
| Aggregator | Produces statistics on synthetic critique JSONL test data |
| Phase 7 runner | 20 patients complete end-to-end; both JSONLs saved per session |
| Findings report | Generated automatically, hand-readable, identifies failure mode categories |

---

## Constraints Carried Forward

- **Loop 1 code untouched.** No changes to `src/loop1/` except possibly adding a `patient_source` parameter to `session.py` for the simulator (see Step 7).
- **Exemplar pool frozen.** No new exemplars in Phase 7. Pool expansion is Phase 8+ depending on findings.
- **Doctor prompt unchanged.** No `doctor_v0_5.txt`. Iteration is Phase 8+.
- **No fine-tuning, no DPO, no training.** Critic is API-only.
- **Local compute only for embeddings + FAISS.** Critic is remote API. Same constraint as Phase 5.
- **All Phase 5 + Phase 6 tests must still pass.**
- **Schema versioning discipline.** All new records carry `schema_version: "0.7.0"`.

---

## Phase 7 Done When

- 20 DDXPlus sessions complete end-to-end with both ground-truth eval and per-turn critique
- `phase7_findings.md` auto-generated with the cross-reference analysis
- New tests passing; existing tests still passing
- Failure modes categorized so Phase 8 has clear direction

---

## Known Out-of-Scope (Phase 8+)

- **Per-session and comparative critique** — Phase 8 if Phase 7 shows per-turn isn't enough signal
- **Exemplar mining from critic suggestions** — Phase 8 (use `would_have_asked` field to draft new exemplars)
- **Larger DDXPlus runs (N=200, N=500)** — Phase 8 if Phase 7 baseline is informative enough to justify scaling
- **Disease → specialty mapping (routing layer)** — Phase 9+
- **Booking integration** — Phase 10+
- **Critic-in-the-loop at inference time** — explicitly not planned
- **DPO or any form of weight updates** — out of project scope

---

## Roadmap Snapshot

```
Phase 7 (current)  →  Turn-level critic + minimal DDXPlus eval (N=20)
                      Deliverable: phase7_findings.md driving Phase 8 decisions

Phase 8            →  Data-driven improvement: exemplar expansion OR prompt
                      iteration OR critic expansion, depending on findings

Phase 9            →  Disease → specialty mapping (routing layer)

Phase 10           →  Booking integration

Phase 11+          →  Deploy
```

---

## First-Session Build Plan for Claude Code

If starting in a fresh Claude Code session, the recommended order is:

1. **Read this file fully**, then read `phase5_notes.md` and `phase6_notes.md` for context on what already exists
2. **Verify `data/ddxplus_raw/` is present** before doing anything else. If not, stop and ask the user to download DDXPlus first.
3. **Build Steps 1-4 in order** (loader → eval set curation → simulator → evaluator). These are independent of the critic and can be tested standalone.
4. **Then Steps 5-6** (critic + aggregator). Independent of DDXPlus.
5. **Then Step 7** (the integrated runner). Tests everything together.
6. **Step 8** (the report) runs automatically as part of Step 7.

Do not skip ahead to the runner before the individual modules are tested. Each module has its own test suite — get those green before integration.