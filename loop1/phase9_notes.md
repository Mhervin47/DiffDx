# Phase 9 Notes — Disease → Specialty Routing Layer

## Context

Phase 7 produces a final differential diagnosis with confidence scores.
Phase 8 improves diagnostic reliability. Phase 9 takes the final differential
from Loop 1 and routes the patient to the appropriate medical specialty.

This phase is **buildable before Phase 7/8 complete** because:
- The DDXPlus disease list is fixed and known
- Specialty mappings are medical knowledge, not dependent on eval findings
- Routing logic is pure computation — no LLM calls required for the core layer

Phase 9 is purely additive. No Loop 1 or Loop 2 code is modified.

---

## Project end-state context

The full triage pipeline:
```
Patient describes symptoms
    → Loop 1: diagnostic dialogue (Phases 0-6)
    → Loop 2: critic evaluation (Phase 7)
    → Phase 9: specialty routing          ← THIS PHASE
    → Phase 10: appointment booking
```

Routing must be trustworthy before booking is added. Overconfident wrong
diagnoses produce wrong routing — which is why Phase 7's confidence
calibration work feeds directly into Phase 9's urgency logic.

---

## What's In Scope

1. Disease → specialty mapping JSON (all DDXPlus conditions covered)
2. Routing schema (`RoutingDecision` Pydantic model)
3. Router module (takes final differential → returns routing options)
4. Urgency classifier (routine / urgent / emergency per disease)
5. Ambiguity handler (returns top 2-3 specialties when differential is unclear)
6. Appointment type suggester (per specialty + urgency combination)
7. Routing runner (reads Loop 1 session JSONL → produces routing output)
8. Tests

---

## What's Out of Scope

- Actual booking integration (Phase 10)
- Real provider/clinic lookup (Phase 10)
- LLM-based routing (pure logic only in Phase 9)
- Modifying Loop 1 or Loop 2 code
- UI or patient-facing interface (Phase 11+)

---

## Settled Design Decisions

### 1. Specialty granularity — standard clinical

Routing targets standard clinical specialties:
Cardiology, Pulmonology, Neurology, Gastroenterology, Infectious Disease,
Rheumatology, Hematology, Orthopedics, Dermatology, Urology, Nephrology,
Endocrinology, Emergency Medicine, General Surgery, Internal Medicine (fallback).

No sub-specialty level (e.g. not "Interventional Cardiology").
No overly broad level (e.g. not just "Surgery").

### 2. Ambiguity handling — return top 2-3 specialties

When the differential is ambiguous (multiple diseases with overlapping
specialties, or final confidence below threshold), return the top 2-3
specialty options ranked by probability-weighted score and let the patient
choose. Do not force a single routing decision under uncertainty.

Ambiguity threshold: if the top specialty's weighted score is less than
**1.5x** the second specialty's score, treat as ambiguous and return multiple.

### 3. Routing output — full RoutingDecision

Every routing output includes:
- Specialty name(s)
- Urgency level (routine / urgent / emergency)
- Reasoning (one sentence explaining why)
- Suggested appointment type

### 4. No LLM calls in core routing

Routing is deterministic logic over the disease→specialty map and the
differential probability weights. No API calls. Fast, cheap, testable.

### 5. Schema version — 0.9.0

All new records carry `schema_version: "0.9.0"`.

---

## Schemas

### RoutingOption

```python
class AppointmentType(str, Enum):
    EMERGENCY = "emergency_visit"
    SAME_DAY = "same_day_appointment"
    URGENT = "urgent_appointment_within_48h"
    ROUTINE = "routine_appointment"
    TELEHEALTH = "telehealth_consultation"

class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"

class RoutingOption(BaseModel):
    specialty: str
    urgency: UrgencyLevel
    reasoning: str                      # one sentence
    appointment_type: AppointmentType
    confidence_weight: float            # probability mass supporting this routing
```

### RoutingDecision

```python
class RoutingDecision(BaseModel):
    schema_version: str = "0.9.0"
    session_id: str
    patient_id: str | None              # set if from DDXPlus eval
    is_ambiguous: bool                  # True if multiple options returned
    primary_routing: RoutingOption      # always present
    alternative_routings: list[RoutingOption]  # empty if unambiguous
    final_differential: list[tuple[str, float]]  # from Loop 1 session
    final_confidence: float             # from Loop 1 session
    routing_model_version: str          # e.g. "disease_specialty_map_v1"
```

---

## Disease → Specialty Map

All DDXPlus conditions mapped to standard clinical specialties with urgency
level and default appointment type. Stored in:
`data/disease_specialty_map.json`

### Urgency definitions

- **emergency**: life-threatening, needs immediate care (ED or call 999/911)
- **urgent**: needs attention within 24-48h, same-day or next-day appointment
- **routine**: can wait for a scheduled appointment (days to weeks)

### Full mapping (DDXPlus conditions)

```json
{
  "Spontaneous pneumothorax": {
    "specialty": "Pulmonology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Spontaneous pneumothorax requires immediate chest decompression assessment"
  },
  "Cluster headache": {
    "specialty": "Neurology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Cluster headache requires neurological evaluation and preventive therapy initiation"
  },
  "Boerhaave syndrome": {
    "specialty": "General Surgery",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Boerhaave syndrome is a surgical emergency requiring immediate intervention"
  },
  "SLE": {
    "specialty": "Rheumatology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "SLE requires rheumatological evaluation for immunosuppressive therapy"
  },
  "Pulmonary embolism": {
    "specialty": "Pulmonology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Pulmonary embolism is a life-threatening emergency requiring immediate anticoagulation"
  },
  "Anemia": {
    "specialty": "Hematology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Anemia requires hematological workup to determine underlying cause"
  },
  "Viral pharyngitis": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "telehealth_consultation",
    "reasoning": "Viral pharyngitis is self-limiting and manageable via telehealth"
  },
  "Tuberculosis": {
    "specialty": "Infectious Disease",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Tuberculosis requires infectious disease evaluation and contact tracing"
  },
  "Myocarditis": {
    "specialty": "Cardiology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Myocarditis can cause fatal arrhythmias and requires immediate cardiac monitoring"
  },
  "Pneumonia": {
    "specialty": "Pulmonology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Pneumonia requires prompt antibiotic therapy and severity assessment"
  },
  "Acute COPD exacerbation / infection": {
    "specialty": "Pulmonology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "COPD exacerbation requires bronchodilator therapy and infection management"
  },
  "GERD": {
    "specialty": "Gastroenterology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "GERD requires gastroenterological evaluation for acid suppression therapy"
  },
  "Bronchitis": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "telehealth_consultation",
    "reasoning": "Acute bronchitis is typically self-limiting and manageable via telehealth"
  },
  "Bronchospasm / acute asthma exacerbation": {
    "specialty": "Pulmonology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Asthma exacerbation requires pulmonological assessment and inhaler therapy adjustment"
  },
  "Allergic sinusitis": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "telehealth_consultation",
    "reasoning": "Allergic sinusitis is manageable with antihistamines via telehealth"
  },
  "Chagas disease": {
    "specialty": "Infectious Disease",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Chagas disease requires antiparasitic therapy and cardiac monitoring"
  },
  "Scombroid food poisoning": {
    "specialty": "Internal Medicine",
    "urgency": "urgent",
    "appointment_type": "same_day_appointment",
    "reasoning": "Scombroid poisoning requires antihistamine treatment and monitoring"
  },
  "Acute laryngitis": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "telehealth_consultation",
    "reasoning": "Acute laryngitis is self-limiting and manageable via telehealth"
  },
  "Possible NSTEMI / STEMI": {
    "specialty": "Cardiology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Possible MI is a cardiac emergency requiring immediate ECG and troponin assessment"
  },
  "Unstable angina": {
    "specialty": "Cardiology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Unstable angina requires immediate cardiac monitoring and intervention assessment"
  },
  "Stable angina": {
    "specialty": "Cardiology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Stable angina requires cardiological evaluation and anti-ischemic therapy"
  },
  "Pericarditis": {
    "specialty": "Cardiology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Pericarditis requires cardiac evaluation for anti-inflammatory therapy"
  },
  "Acute pulmonary edema": {
    "specialty": "Cardiology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Acute pulmonary edema is a cardiac emergency requiring immediate diuresis"
  },
  "Panic attack": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "telehealth_consultation",
    "reasoning": "Panic attack requires initial assessment and referral to mental health if recurrent"
  },
  "Guillain-Barré syndrome": {
    "specialty": "Neurology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Guillain-Barré can cause respiratory failure and requires immediate neurological care"
  },
  "Atrial fibrillation": {
    "specialty": "Cardiology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Atrial fibrillation requires rate control and stroke risk assessment"
  },
  "Larygospasm": {
    "specialty": "Internal Medicine",
    "urgency": "urgent",
    "appointment_type": "same_day_appointment",
    "reasoning": "Laryngospasm requires same-day evaluation to rule out airway compromise"
  },
  "Acute otitis media": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Acute otitis media requires antibiotic evaluation"
  },
  "Ruptured appendix": {
    "specialty": "General Surgery",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Ruptured appendix is a surgical emergency requiring immediate intervention"
  },
  "Appendicitis": {
    "specialty": "General Surgery",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Appendicitis requires urgent surgical evaluation to prevent rupture"
  },
  "Acute dystonic reactions": {
    "specialty": "Neurology",
    "urgency": "urgent",
    "appointment_type": "same_day_appointment",
    "reasoning": "Acute dystonic reactions require neurological assessment and anticholinergic therapy"
  },
  "Myasthenia gravis": {
    "specialty": "Neurology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Myasthenia gravis requires neurological evaluation and immunosuppressive therapy"
  },
  "Epiglottitis": {
    "specialty": "Emergency Medicine",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Epiglottitis can cause complete airway obstruction and is a medical emergency"
  },
  "Whooping cough": {
    "specialty": "Infectious Disease",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Whooping cough requires antibiotic therapy and contact tracing"
  },
  "Pulmonary neoplasm": {
    "specialty": "Pulmonology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Pulmonary neoplasm requires oncological staging workup"
  },
  "Sarcoidosis": {
    "specialty": "Pulmonology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Sarcoidosis requires pulmonological evaluation for systemic involvement"
  },
  "PSVT": {
    "specialty": "Cardiology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "PSVT requires electrophysiological evaluation and rhythm management"
  },
  "Anaphylaxis": {
    "specialty": "Emergency Medicine",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Anaphylaxis is a life-threatening emergency requiring immediate epinephrine"
  },
  "Acute rhinosinusitis": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "telehealth_consultation",
    "reasoning": "Acute rhinosinusitis is typically self-limiting and manageable via telehealth"
  },
  "Chronic rhinosinusitis": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Chronic rhinosinusitis requires evaluation for underlying allergy or anatomical cause"
  },
  "HIV (initial infection)": {
    "specialty": "Infectious Disease",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "HIV initial infection requires immediate infectious disease evaluation and ART initiation"
  },
  "Influenza": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "telehealth_consultation",
    "reasoning": "Influenza is manageable with antivirals via telehealth in non-high-risk patients"
  },
  "Pancreatic neoplasm": {
    "specialty": "Gastroenterology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Pancreatic neoplasm requires urgent oncological workup"
  },
  "Acute pancreatitis": {
    "specialty": "Gastroenterology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Acute pancreatitis requires immediate IV fluids and monitoring for complications"
  },
  "Chronic kidney disease": {
    "specialty": "Nephrology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "CKD requires nephrology evaluation for disease progression management"
  },
  "Urinary tract infection": {
    "specialty": "Internal Medicine",
    "urgency": "routine",
    "appointment_type": "telehealth_consultation",
    "reasoning": "Uncomplicated UTI is manageable with antibiotics via telehealth"
  },
  "Pyelonephritis": {
    "specialty": "Internal Medicine",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Pyelonephritis requires IV antibiotic evaluation to prevent sepsis"
  },
  "Nephrotic syndrome": {
    "specialty": "Nephrology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Nephrotic syndrome requires nephrology evaluation for biopsy and immunosuppression"
  },
  "Celiac disease": {
    "specialty": "Gastroenterology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Celiac disease requires gastroenterological confirmation and dietary counseling"
  },
  "Crohn's disease": {
    "specialty": "Gastroenterology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Crohn's disease requires gastroenterological evaluation for immunosuppressive therapy"
  },
  "Acute nephritis": {
    "specialty": "Nephrology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Acute nephritis requires nephrology evaluation to prevent progression to renal failure"
  },
  "Pulmonary fibrosis": {
    "specialty": "Pulmonology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Pulmonary fibrosis requires pulmonological evaluation for antifibrotic therapy"
  },
  "Hypothyroidism": {
    "specialty": "Endocrinology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Hypothyroidism requires endocrine evaluation and thyroid hormone replacement"
  },
  "Hyperthyroidism": {
    "specialty": "Endocrinology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Hyperthyroidism requires endocrine evaluation to prevent thyroid storm"
  },
  "Diabetes mellitus": {
    "specialty": "Endocrinology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Diabetes requires endocrinological evaluation for glycemic management"
  },
  "Systemic vasculitis": {
    "specialty": "Rheumatology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Systemic vasculitis requires rheumatological evaluation for immunosuppressive therapy"
  },
  "Phlebitis": {
    "specialty": "Internal Medicine",
    "urgency": "urgent",
    "appointment_type": "same_day_appointment",
    "reasoning": "Phlebitis requires evaluation to rule out DVT extension"
  },
  "DVT": {
    "specialty": "Hematology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "DVT requires anticoagulation initiation and PE risk assessment"
  },
  "Pulmonary arterial hypertension": {
    "specialty": "Pulmonology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "PAH requires pulmonological evaluation for vasodilator therapy"
  },
  "CVST": {
    "specialty": "Neurology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "CVST requires immediate anticoagulation to prevent venous infarction"
  },
  "Measles": {
    "specialty": "Infectious Disease",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Measles requires isolation, supportive care, and contact tracing"
  },
  "Cauda equina syndrome": {
    "specialty": "Neurology",
    "urgency": "emergency",
    "appointment_type": "emergency_visit",
    "reasoning": "Cauda equina syndrome is a surgical emergency to prevent permanent paralysis"
  },
  "Inflammatory back pain": {
    "specialty": "Rheumatology",
    "urgency": "routine",
    "appointment_type": "routine_appointment",
    "reasoning": "Inflammatory back pain requires rheumatological evaluation for spondyloarthropathy"
  },
  "Cellulitis": {
    "specialty": "Internal Medicine",
    "urgency": "urgent",
    "appointment_type": "same_day_appointment",
    "reasoning": "Cellulitis requires antibiotic therapy and monitoring for systemic spread"
  },
  "Biliary colic": {
    "specialty": "Gastroenterology",
    "urgency": "urgent",
    "appointment_type": "urgent_appointment_within_48h",
    "reasoning": "Biliary colic requires gastroenterological evaluation for cholecystectomy planning"
  }
}
```

**Note to Claude Code:** This mapping covers the primary DDXPlus conditions plus
the Loop 1 exemplar pool diseases. When the router encounters a disease not in
this map, fall back to `Internal Medicine` / `routine` / `routine_appointment`
and log a warning. Do not crash. The fallback case should be tracked so the map
can be expanded in Phase 10.

---

## File Structure

```
src/loop3/                              # new package
├── __init__.py
└── routing/
    ├── __init__.py
    ├── router.py                       # core routing logic
    ├── routing_schema.py               # RoutingOption, RoutingDecision Pydantic models
    ├── urgency_classifier.py           # maps disease + confidence → urgency
    └── ambiguity_handler.py            # detects and handles ambiguous differentials

data/
└── disease_specialty_map.json          # the full mapping above, COMMITTED

src/loop3/runners/
├── __init__.py
└── phase9_routing_runner.py            # reads Loop 1 JSONL → produces RoutingDecision

tests/
├── test_router.py
├── test_urgency_classifier.py
├── test_ambiguity_handler.py
└── test_routing_runner.py
```

**Loop 1 and Loop 2 code untouched.**

---

## Build Order

### Step 1 — Routing schema

`src/loop3/routing/routing_schema.py`

Define `UrgencyLevel`, `AppointmentType`, `RoutingOption`, `RoutingDecision`
Pydantic models exactly as specified in the Schemas section above.

**Tests**: schema instantiation, serialization, schema_version present.

### Step 2 — Disease → specialty map

Save the full JSON mapping from this file to `data/disease_specialty_map.json`.
Add a loader function:

```python
def load_disease_specialty_map(path: str) -> dict:
    """Load and validate the disease → specialty mapping."""
```

**Tests**: load map, verify all entries have required keys, verify no missing
urgency or appointment_type fields.

### Step 3 — Urgency classifier

`src/loop3/routing/urgency_classifier.py`

```python
def classify_urgency(
    disease: str,
    confidence: float,
    specialty_map: dict,
) -> UrgencyLevel:
    """
    Return urgency level for a disease.
    If confidence < 0.4 and urgency would be 'routine', bump to 'urgent'
    (low confidence on a routine diagnosis → still needs a human to confirm).
    If disease not in map, return 'routine' and log warning.
    """
```

Confidence adjustment rule:
- If `final_confidence < 0.4`: bump `routine` → `urgent` (never downgrade emergency)
- If `final_confidence >= 0.4`: use map value as-is

**Tests**: emergency disease at any confidence stays emergency. Routine disease
at low confidence bumps to urgent. Unknown disease returns routine + warning.

### Step 4 — Ambiguity handler

`src/loop3/routing/ambiguity_handler.py`

```python
def is_ambiguous(
    differential: list[tuple[str, float]],
    specialty_map: dict,
) -> bool:
    """
    Returns True if the top specialty's probability-weighted score is less
    than 1.5x the second specialty's score.
    """

def get_routing_options(
    differential: list[tuple[str, float]],
    specialty_map: dict,
    final_confidence: float,
    max_options: int = 3,
) -> list[RoutingOption]:
    """
    Aggregate differential probabilities by specialty.
    Return top N specialty options ranked by probability-weighted score.
    Always returns at least 1 option.
    """
```

Aggregation logic:
1. For each disease in the differential, look up its specialty and weight by probability
2. Sum probability weights per specialty
3. Sort specialties by total weight descending
4. If ambiguous (top score < 1.5x second), return top 2-3
5. If unambiguous, return top 1 only

**Tests**: unambiguous differential → 1 option. Ambiguous differential → 2-3
options. All-same-specialty differential → 1 option regardless of threshold.

### Step 5 — Core router

`src/loop3/routing/router.py`

```python
def route(
    session_id: str,
    final_differential: list[tuple[str, float]],
    final_confidence: float,
    patient_id: str | None = None,
) -> RoutingDecision:
    """
    Main entry point. Orchestrates urgency classification,
    ambiguity detection, and RoutingDecision assembly.
    """
```

Logic:
1. Load specialty map
2. Get routing options via `get_routing_options`
3. Classify urgency for each option via `classify_urgency`
4. Detect ambiguity via `is_ambiguous`
5. Assemble and return `RoutingDecision`

The highest-urgency option always becomes `primary_routing` regardless of
probability weight — safety first. If a low-probability disease is emergency
level, it surfaces as primary.

**Tests**: PE + DVT differential → Pulmonology emergency as primary. Mixed
routine/emergency differential → emergency surfaces as primary. Unknown disease
→ fallback to Internal Medicine routine.

### Step 6 — Phase 9 runner

`src/loop3/runners/phase9_routing_runner.py`

```python
def run_routing_from_session(session_jsonl_path: str) -> RoutingDecision:
    """
    Read a completed Loop 1 session JSONL.
    Extract final_differential and final_confidence from the closing turn.
    Return a RoutingDecision.
    """
```

Can run standalone on any saved Loop 1 session JSONL. Also callable from the
Phase 7 integrated runner so routing decisions can be produced alongside
ground-truth eval and critic output.

**Tests**: run on `case_01.jsonl` (the Phase 6 demo session). Verify PE routes
to Pulmonology emergency (the correct routing given the known failure mode where
the doctor ranked ACS above PE — the router should still catch PE if it appears
anywhere in the top differential).

---

## Eval Criteria

| Item | Pass condition |
|---|---|
| Routing schema | Instantiates cleanly, all fields present, schema_version correct |
| Disease map | All entries load without error, no missing required fields |
| Urgency classifier | Correct urgency on 10 hand-checked diseases; low-confidence bump works |
| Ambiguity handler | Correctly detects ambiguous vs unambiguous differentials |
| Core router | Emergency diseases surface as primary regardless of probability weight |
| Phase 9 runner | Runs end-to-end on case_01.jsonl; produces valid RoutingDecision |

---

## Constraints

- **Loop 1 and Loop 2 code untouched.**
- **No LLM calls in routing logic.** Pure deterministic computation.
- **No real provider/clinic lookup.** That is Phase 10.
- **All Phase 5, 6, and 7 tests must still pass.**
- **Schema versioning discipline.** All new records carry `schema_version: "0.9.0"`.
- **Fallback always returns something.** Router never raises on unknown disease.

---

## Known Out-of-Scope (Phase 10+)

- Real provider directory lookup
- Appointment slot availability
- Patient-facing UI for specialty selection
- Booking confirmation and calendar integration
- Specialty map expansion beyond DDXPlus diseases
- Multi-language routing output

---

## Roadmap Snapshot

```
Phase 7 (parallel)  →  Turn-level critic + DDXPlus eval (N=20)
Phase 8             →  Data-driven improvement (exemplar expansion / prompt iteration)
Phase 9 (current)   →  Disease → specialty routing layer
Phase 10            →  Booking integration
Phase 11+           →  Deploy
```

---

## First-Session Build Plan for Claude Code

1. Read this file fully.
2. Read `phase7_notes.md` and `phase7_claudecode_brief.md` for system context.
3. Confirm `data/` directory exists and is writable.
4. Build Steps 1-5 in order. Get each step's tests green before moving on.
5. Build Step 6 (runner) last. Test on `case_01.jsonl`.
6. Do not modify any files in `src/loop1/` or `src/loop2/`.