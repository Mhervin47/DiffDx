# Phase 7 — Claude Code Handoff Brief

## How to use this file

Read `phase7_notes.md` first. This brief adds implementation details and resolves
open decisions from that doc. **Where they conflict, this brief wins.**

Starting instruction:
> Verify `data/DDXPlus_Raw/` exists and contains the 5 expected files. Then begin Step 1.

---

## 1. Project context

Loop 1 is a multi-turn LLM diagnostic dialogue system (doctor model: Groq
Llama-3.3-70b-versatile). Phase 6 hardened the system end-to-end. Phase 7 adds
measurement infrastructure: a turn-level critic LLM and a 20-patient DDXPlus eval.
No Loop 1 code is modified except one `patient_source` parameter addition to
`session.py` (see Section 4).

---

## 2. Hardware / API constraints

- Machine: 8GB M2 MacBook Air
- LLM inference: Groq API (free tier) for doctor + simulator
- Critic model: gemini/gemini-2.0-flash via Google AI Studio API (litellm wrapper, free tier). API key loaded from GEMINI_API_KEY in .env. CRITIC_MODEL env var wired for future upgrade.
- Local compute only for embeddings + FAISS (already in place from Phase 5)
- Critic model configurable via env var `CRITIC_MODEL` for future upgrades

---

## 3. DDXPlus dataset — confirmed format and exact paths

**Location:** `data/DDXPlus_Raw/` (capital D, capital P, capital R — exact casing matters on Mac)

**Files:**
```
data/DDXPlus_Raw/
├── release_conditions.json
├── release_evidences.json
├── release_test_patients        ← no file extension
├── release_train_patients       ← no file extension
└── release_validate_patients    ← no file extension
```

**Use `release_test_patients` only** for the eval set. Keep train/validate untouched.

### Parsing rules (confirmed by inspection)

**Patient files:**
- No file extension but CSV-formatted. Load with:
  ```python
  pd.read_csv('data/DDXPlus_Raw/release_test_patients')
  ```
- `EVIDENCES` column is a **Python-list-formatted string with single quotes** — use
  `ast.literal_eval()`, NOT `json.loads()`. Confirmed working.
- `DIFFERENTIAL_DIAGNOSIS` column is also `ast.literal_eval()` — it's a list of
  `[condition_name_english, probability]` pairs, already in English.
- `PATHOLOGY` column is a plain English string (e.g. `"GERD"`, `"Bronchitis"`).
- `INITIAL_EVIDENCE` is a single evidence code string (e.g. `"E_201"`).

**Evidence string decoding:**
```python
# Three formats exist in the EVIDENCES list:
# Binary:       "E_91"           → symptom present, value = 1
# Categorical:  "E_10_@_2"       → symptom E_10, value = "2"
# Multi-choice: "E_54_@_V_112"   → symptom E_54, value = "V_112"

def decode_evidence(ev_string: str, evidences_ontology: dict) -> tuple[str, Any]:
    parts = ev_string.split("_@_")
    code = parts[0]
    value = parts[1] if len(parts) > 1 else 1  # binary = present if in list
    name = evidences_ontology[code]["question_en"]
    return name, value
```

Split decoded evidences into `symptoms` vs `antecedents` using the `is_antecedent`
flag in `release_evidences.json`. Use `question_en` as the human-readable key in
output dicts — NOT the `E_` code. The simulator and critic both need readable names.

**`release_evidences.json`:**
- Keys are `E_` prefixed codes (e.g. `"E_91"`). Standard format confirmed.
- `data_type` field: `"B"` (binary), `"C"` (categorical), `"M"` (multi-choice).
- `is_antecedent` flag: `true` = antecedent, `false` = symptom.
- Use `question_en` field as the human-readable symptom name.

**`release_conditions.json`:**
- Keys are **French condition names** (e.g. `"Pneumothorax spontané"`).
- Use `cond-name-eng` field to map to English.
- `PATHOLOGY` and `DIFFERENTIAL_DIAGNOSIS` in patient files already use English names
  directly — so `release_conditions.json` is only needed if severity scores or
  per-disease symptom lists are required. For Phase 7 it is **optional**; load it
  but don't block on it.

---

## 4. Resolved design decisions

### patient_source — parameter, not adapter

Add a `patient_source` parameter to `session.py`. Do NOT build a separate adapter
class. The branch point is inside the session loop (how to get the patient's next
reply). All other loop logic is untouched.

```python
class PatientSource(BaseModel):
    mode: Literal["interactive", "simulator", "ddxplus"]
    ddxplus_patient: DDXPlusPatient | None = None  # required if mode == "ddxplus"
```

Simulator guardrail logic (no leakage check) lives **inside the simulator branch**,
co-located with the mode logic. Not in a wrapper.

### TurnCritique schema — add weakness_category field

The `phase7_notes.md` schema has `weakness: str | None` as free text. Add a
structured field alongside it:

```python
weakness_category: Literal[
    "redundant_question",
    "missed_red_flag",
    "poor_differential",
    "premature_confidence",
    "other"
] | None = None
```

This makes the aggregator's clustering in Step 6 reliable without regex on free text.
Both fields are kept — `weakness` for human-readable detail, `weakness_category` for
aggregation.

### Schema version

All new records carry `schema_version: "0.7.0"` as specified in `phase7_notes.md`.

---

## 5. Step 0 — Critic calibration fixture (do this before Step 5)

Before running the critic on DDXPlus patients, validate it against a saved session
with known failure modes.

**The saved session:** `case_01.jsonl` (the Phase 6 live demo — 58yo M, chest pain +
leg swelling + Mexico trip, terminated at turn 10/15 on confidence plateau).

**Three known failure modes the critic MUST catch:**

| # | Failure | Expected critic signal |
|---|---------|----------------------|
| 1 | Differential ordering wrong: ranked ACS above PE despite confirmed DVT + travel history | `differential_quality_score < 0.5`, `weakness_category: "poor_differential"` |
| 2 | Suboptimal screening question: asked hemoptysis (late PE finding) instead of dyspnea/pleurisy | `question_quality_score < 0.5`, `weakness_category: "missed_red_flag"`, `would_have_asked` mentions dyspnea or pleurisy |
| 3 | Redundant question at turn 9: re-asked chest pain character already answered at turns 1 and 5 | `weakness_category: "redundant_question"` |

**Pass threshold:** critic must catch all 3/3 before running on 20 DDXPlus patients.
If 2/3, identify which critique dimension is weak and iterate the critic prompt.
If 0-1/3, the critic prompt needs significant revision before proceeding.

**Fixture location:**
```
tests/fixtures/critic_calibration/
  case_01_session.jsonl      # copy of the Phase 6 demo session
  case_01_expected.json      # documents the 3 expected flags per turn
```

Build a test in `tests/test_critic.py` that runs the critic on `case_01_session.jsonl`
and asserts the three failure modes are caught. This test must pass before Step 7
(the integrated runner) is started.

---

## 6. Critic prompt requirements

`prompts/critic_turn_v0_1.txt` must explicitly instruct the critic to detect:

1. **Redundant questions** — has this symptom/topic already been asked in a prior turn?
2. **Missed red flags** — given the symptoms disclosed so far, is there a high-yield
   screening question that was skipped in favor of a lower-yield one?
3. **Differential ordering errors** — does the ranked differential reflect the
   disclosed evidence, or is a less-supported diagnosis ranked above a better-supported one?
4. **Premature confidence** — is the confidence score consistent with the evidence
   gathered, or is the doctor claiming high confidence with thin evidence?

The critic receives:
- Doctor's question, reasoning, differential, confidence for the current turn
- Patient profile snapshot at this turn
- Conversation history up to (not including) the current turn

The critic does NOT receive:
- Ground truth pathology
- The patient's answer to the current question

---

## 7. Build order (with amendments)

Follow the order in `phase7_notes.md` Steps 1–8, with these additions:

- **Before Step 1:** confirm `data/DDXPlus_Raw/` exists with all 5 files. Halt and
  ask if missing.
- **After Step 4, before Step 5:** build the critic calibration fixture (Section 5
  above). Get the 3/3 calibration pass before proceeding.
- **Step 1 parsing:** use `ast.literal_eval()` not `json.loads()` for `EVIDENCES`
  and `DIFFERENTIAL_DIAGNOSIS`. Use exact path `data/DDXPlus_Raw/release_test_patients`
  (no extension).

Get each step's tests green before moving to the next. Do not skip ahead to the
runner before individual modules are tested.

---

## 8. Done criteria

Identical to `phase7_notes.md` eval table, plus:

| Item | Pass condition |
|---|---|
| Critic calibration | 3/3 known failure modes caught on case_01 before DDXPlus run |
| weakness_category | Populated on all TurnCritique records; aggregator counts by category |
| Findings report | Cross-reference: sessions with low question quality AND wrong diagnosis identified |