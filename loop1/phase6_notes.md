# Phase 6 Notes — Hardening and Closing Turn

## Context

Phase 5 complete. 47 + new tests passing. MMR retrieval hits 100% target exemplar appearance rate for measles and CVST (vs 15–20% random). Phase 5 tagged `phase5-complete`.

Phase 6 is hardening only. No new ML capabilities. No deploy. No frontend. No changes to the exemplar pool, embedding model, or MMR logic. The doctor prompt stays at v0.4 unless confidence plateau diagnosis (item 5) requires a targeted patch.

**Deploy moved out of Phase 6.** Railway deploy + GUI now happen after Loop 2 (Phase 8+). Rationale: deploying a Loop 1 snapshot before Loop 2 produces a system you'll throw away once the critic-trained doctor model replaces the current Llama-3.3 doctor. The current pipeline runs fine locally for Loop 2 training data generation; that's all it needs to do until then.

---

## What's In Scope

1. Closing turn
2. `schema_version` on every JSONL entry
3. Replay determinism for prompt iteration
4. Safety substring matching for emergency phrases
5. Confidence plateau diagnosis and fix
6. Symptom categorization bug fix
7. Long-session symptom over-listing fix

---

## 1. Closing Turn

### Problem

When termination fires (confidence threshold or max-turns), the session ends abruptly. The patient receives no summary, no next steps, no acknowledgment of unresolved questions. This is a UX gap and a clinical realism gap — no doctor ends a consultation without a closing statement.

Beyond UX, this matters for Loop 2: training data with clean session closures is more useful than training data that stops mid-thought.

### Design

New field on `FinalRecord`:

```python
class ClosingTurn(BaseModel):
    leading_diagnosis: str           # plain language, no jargon
    differential_summary: str        # brief — top 2-3 with rough likelihoods
    recommended_next_steps: list[str]
    unresolved_patient_questions: list[str]  # from session history if any
    generated_by: str                # model string, for traceability

class FinalRecord(BaseModel):
    ...
    closing_turn: ClosingTurn | None = None
```

### Behavior

- Triggered at termination, after the final `TurnRecord` is written
- Separate LLM call (not part of the diagnostic turn loop) — uses the same model, temperature 0
- Prompt: give the model the full profile + final differential, ask for a closing statement in plain language addressed to the patient
- Closing turn is logged to JSONL but not shown to the doctor LLM on subsequent turns (there are none)
- If the closing turn generation fails (API error, parse error), log the failure and continue — do not crash the session

### Schema note

`closing_turn: ClosingTurn | None = None` — optional so existing JSONL logs remain valid without migration.

---

## 2. `schema_version` on Every JSONL Entry

### Problem

`schema_version` is partially present. Inconsistent across record types. Makes log parsing fragile when the schema evolves — and Loop 2 will be reading these logs heavily, so fragility here propagates downstream.

### Design

Add `schema_version: str = "0.5.0"` to every Pydantic model that gets serialized to JSONL:

- `TurnRecord`
- `FinalRecord`
- `ClosingTurn`
- `SessionMeta` (if it exists as a separate record type)
- `SafetyEvent` (new in this phase, item 4)

Version format: `MAJOR.MINOR.PATCH`
- MAJOR: breaking schema change (field removed or renamed)
- MINOR: additive change (new optional field)
- PATCH: internal/logging change

Current version at Phase 6 start: `0.5.0` (Phase 5 was `0.4.x`).

### Migration

Existing Phase 4/5 logs are not migrated. Log readers should treat missing `schema_version` as `<0.5.0` and handle gracefully (ignore unknown fields, fill missing optional fields with None).

---

## 3. Replay Determinism for Prompt Iteration

### Problem

Prompt iteration (e.g., testing a v0.5 doctor prompt) currently requires re-running live sessions. This is slow, consumes Groq API quota, and introduces patient-input variance — you can't isolate whether a change in output is due to the prompt or the patient responses.

This becomes especially important if Phase 6 item 5 (confidence plateau) calls for a doctor prompt patch — replay is the only clean way to validate prompt changes against prior baselines.

### Design

**Replay mode**: given a logged JSONL session, re-run the doctor LLM on each turn using the saved patient inputs, and produce a new JSONL log with the new prompt.

```
scripts/replay_session.py --session <session_id> --prompt <prompt_file> --out <output_jsonl>
```

- Reads the original session JSONL
- For each turn: reconstructs the exact context (profile, history, exemplars) from the log
- Calls the doctor LLM with the new prompt, same temperature (0)
- Writes a new JSONL with `replay_of: <original_session_id>` and `prompt_version: <new_version>` fields on each record
- Does NOT re-run the patient LLM or profile updater — patient inputs are taken verbatim from the original log
- Output is diff-able against the original log for side-by-side comparison

### Constraints

- Replay requires that exemplar embeddings are deterministic (they are — committed to git)
- Replay requires that `retrieved_exemplar_ids` is logged per turn (it is — Phase 4)
- Replay injects the same exemplar IDs from the original log, not re-retrieved ones — this isolates prompt changes from retrieval changes
- Temperature 0 throughout — deterministic given same context

### Use case

Primary use: validating doctor prompt v0.5+ against Phase 4/5 baseline sessions without consuming new Groq quota on patient simulation. Also useful for Loop 2 prep: re-running sessions with different prompts to generate paired preference data.

---

## 4. Safety Substring Matching for Emergency Phrases

### Problem

The current session loop has no real-time safety layer. If a patient input contains emergency language ("chest pain and can't breathe", "I'm having a stroke", "suicidal"), the system continues the diagnostic dialogue loop rather than breaking out.

Even though Loop 1 is not user-facing yet, training data with emergency-language inputs being treated as normal diagnostic inputs is bad training data. Building the safety layer now means Loop 2 trains on cleaner signal.

### Design

**Substring match on patient input before each turn**, against a curated list of emergency phrases.

```python
EMERGENCY_PHRASES = [
    "can't breathe", "cannot breathe", "difficulty breathing",
    "chest pain",
    "stroke", "face drooping", "arm weakness", "speech difficulty",
    "unconscious", "unresponsive",
    "severe bleeding", "won't stop bleeding",
    "suicidal", "want to die", "kill myself",
    "anaphylaxis", "throat closing", "throat swelling",
    "seizure",
    "overdose",
]
```

Match is case-insensitive, whole-phrase (not character-level substring — "chest pain" matches "I have chest pain", not "orchestral painting").

**On match:**
- Immediately terminate the session loop
- Log a `SafetyEvent` record to JSONL with the matched phrase and the original input
- Print a hardcoded emergency message to the terminal (not LLM-generated):
  ```
  ⚠️  Emergency language detected. Please call emergency services (112/911) immediately.
  This system is not a substitute for emergency medical care.
  ```
- Do not generate a closing turn (inappropriate in an emergency context)

### Schema

```python
class SafetyEvent(BaseModel):
    schema_version: str = "0.5.0"
    session_id: str
    turn: int
    matched_phrase: str
    patient_input: str
    action: str = "session_terminated"
```

### Scope note

This is a substring filter, not an LLM safety classifier. It catches explicit emergency language only. It does not catch subtle distress, implicit suicidality, or clinical deterioration described in non-emergency terms. A full safety classifier is out of scope for Phase 6 and tracked for post-Loop-2.

---

## 5. Confidence Plateau Diagnosis and Fix

### Problem

Confidence consistently plateaus near 0.6 across cases. The termination threshold is 0.65, meaning sessions almost always terminate on max-turns rather than genuine confidence. This was noted in Phase 2 (threshold lowered from 0.7 to 0.65 as a "Llama-3.3 calibration ceiling") and carried forward through Phase 5 unresolved.

### Diagnosis protocol

Before fixing, determine the cause:

**Hypothesis A — Prompt anchoring**: the doctor prompt's confidence scale description is anchoring the model below 0.65. Test by removing or rephrasing the scale description and re-running on 3 cases.

**Hypothesis B — Model calibration ceiling**: Llama-3.3-70b genuinely doesn't output >0.65 confidence in JSON regardless of prompt. Test by asking the model directly in a one-shot call: "Given this clear-cut case of measles with all classic features confirmed, what confidence would you assign?" If it refuses to go above 0.65, it's a model ceiling.

**Hypothesis C — Threshold too high for the task**: diagnostic confidence of 0.65 is actually appropriate for a 5-turn dialogue — the model is correctly uncertain. Test by checking whether confidence correlates with case clarity (simple cases should hit 0.65 faster than complex ones).

### Fix (contingent on diagnosis)

- If A: patch doctor prompt to v0.5 with revised confidence scale. Validate with replay determinism (item 3).
- If B: lower termination threshold to 0.55, document as model limitation. Do not fight the model.
- If C: restructure termination to use turn count as primary signal and confidence as a tiebreaker. Document rationale.

Run diagnosis on cases 1, 2 (CVST), and measles before implementing any fix.

---

## 6. Symptom Categorization Bug Fix

### Problem

Occasional symptoms land in `history.medical` instead of `symptoms`. Carried forward from Phase 3. Example: "patient reports fever for 3 days" sometimes categorized as a medical history item rather than an active symptom.

This bug contaminates Loop 2 training data — the critic will see incorrectly structured profiles and learn from noise. Fixing now is cheap; fixing after Loop 2 has trained on bad data is expensive.

### Diagnosis

Add logging to the profile updater to capture every field-assignment decision for one full session on case 3 (infectious-GI — most likely to trigger the bug given symptom overlap with travel history). Identify which patient inputs trigger the miscategorization.

### Fix

The profile updater prompt (not the doctor prompt) is likely the source. Targeted patch to the updater prompt: add explicit examples of symptom vs history distinction. Do not touch the doctor prompt.

Validation: re-run case 3 five times, confirm zero miscategorizations.

---

## 7. Long-Session Symptom Over-Listing Fix

### Problem

Long sessions (turn 5+) produce symptom variant over-listing in the profile — e.g., "fever", "high fever", "fever since 3 days", "persistent fever" as four separate entries for what is one symptom. The Phase 3 dedup (case-insensitive + substring + content-word-subset) catches some of this but not all variant forms.

### Fix

Extend the profile dedup logic with a similarity threshold step:

- After the existing dedup passes, run a pairwise similarity check on the symptom list
- If two symptoms share >80% of content words (after stopword removal), keep the longer/more specific one and discard the other
- This is a string operation — no embeddings, no LLM call

Alternatively: add an explicit instruction to the compressor prompt v0.2 — "deduplicate symptom variants; prefer the most specific formulation" — and test whether the LLM handles it. If yes, no code change needed.

Try the prompt approach first (cheaper). Fall back to the string dedup if it doesn't hold across 5 test sessions.

---

## Build Order

| Step | Item | Dependency |
|---|---|---|
| 1 | `schema_version` on all JSONL records | None — do first, everything else builds on it |
| 2 | Safety substring matching | None — independent |
| 3 | Closing turn | schema_version done |
| 4 | Replay determinism | schema_version done |
| 5 | Confidence plateau diagnosis | None — diagnostic only |
| 6 | Confidence plateau fix | Diagnosis done; replay determinism (step 4) helpful |
| 7 | Symptom categorization bug fix | None — independent |
| 8 | Long-session over-listing fix | None — independent |

---

## Eval Criteria

| Item | Pass condition |
|---|---|
| Closing turn | Generated on every termination; graceful failure if LLM call fails; logged correctly |
| schema_version | Present on every JSONL record type; existing logs parse without error |
| Replay determinism | Replayed session with same prompt produces identical `question` field on every turn |
| Safety matching | Emergency phrase in patient input terminates session and logs `SafetyEvent`; normal input unaffected |
| Confidence plateau | Termination fires on confidence (not max-turns) on at least 3 of 5 base cases after fix |
| Symptom categorization | Zero miscategorizations on 5 runs of case 3 |
| Over-listing fix | No duplicate symptom variants in profile on 5 runs of a long session (turn 6+) |

---

## Constraints Carried Forward

- Local compute only for embeddings + FAISS. No API calls in retrieval path.
- Exemplar pool frozen. No additions or text edits.
- `prompts/doctor_v0_4.txt` unchanged unless confidence plateau diagnosis requires a targeted patch (item 5 only).
- `select_random` not deleted — kept for ablation.
- All Phase 5 tests must still pass after every Phase 6 step.

---

## Phase 6 Done When

- All 7 items implemented and passing their eval criteria
- 47 + Phase 5 + Phase 6 tests passing
- A full session produces a clean JSONL log with `schema_version` on every record, a closing turn, and (if applicable) safety event handling
- Replay script can reproduce a baseline session deterministically
- Ready to start generating Loop 2 training data at scale

---

## Known Out-of-Scope (Phase 7+)

- **Loop 2: offline critic + DPO training** on the JSONL logs produced by Loop 1 — this is the next major phase after Phase 6
- **Railway deploy + GUI** — deferred until after Loop 2 (Phase 8+). The current Loop 1 doctor will be replaced by the DPO-trained doctor; deploying before that produces a throwaway artifact.
- **Full LLM safety classifier** (replacing substring matching)
- **Clinical fine-tuned embedding model** (MedCPT or similar) replacing MiniLM
- **Exemplar pool expansion** beyond 15
- **Multi-patient concurrent sessions**

---

## Roadmap Snapshot

```
Phase 6 (current)  →  Hardening: closing turn, schema_version, replay,
                      safety, confidence fix, categorization fix, dedup fix

Phase 7            →  Loop 2: offline critic + DPO training on Loop 1 logs

Phase 8+           →  Railway deploy + GUI with the new DPO-trained doctor
                      (the system worth deploying is the post-Loop-2 system,
                      not the current Llama-3.3 baseline)
```