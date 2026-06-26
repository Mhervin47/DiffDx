# Loop 1 — Doctor Dialogue System: Spec-Driven Development

## Purpose

Build the runtime question-answering loop end-to-end with no learning. At the end of Loop 1, the system can take a patient profile, conduct a multi-turn diagnostic dialogue with ICL + CoT, manage its context window, and emit a structured final record. Every session is logged in a format that Loop 2 can later consume for training.

## Non-goals for Loop 1

- No critic LLM (built in Loop 2)
- No DPO, LoRA, or any fine-tuning
- No patient simulator (real user or scripted profiles only)
- No production deployment, real patients, or clinical use
- No multimodal input (images, audio) — text only

## Success criteria

Loop 1 is done when all of the following hold:

1. A patient can complete a dialogue from intake to final record without the system crashing or producing malformed JSON.
2. The doctor LLM asks relevant follow-up questions that reduce diagnostic uncertainty across turns (verified by manual review on 20 test cases).
3. Sessions terminate cleanly via confidence threshold or max-turn cap.
4. Every session is logged to disk in the locked schema.
5. Dynamic retrieval returns diverse, relevant exemplars (verified manually on 10 queries).
6. Context compression keeps the doctor coherent past turn 10 without prompt bloat.

---

## Locked schemas

These schemas must not change after Stage 2. Adding fields is allowed; renaming or removing is not. Loop 2 depends on stability here.

### Patient profile (mutable, updated each turn)

```json
{
  "session_id": "uuid",
  "demographics": {
    "age": 34,
    "sex": "F",
    "other": {}
  },
  "chief_complaint": "string",
  "symptoms": [
    {"name": "headache", "onset": "3 days ago", "severity": "moderate", "notes": ""}
  ],
  "history": {
    "medical": [],
    "medications": [],
    "allergies": [],
    "family": [],
    "social": []
  },
  "ruled_out": [],
  "ruled_in": [],
  "free_notes": "string"
}
```

### Doctor turn output (every turn, strict JSON)

```json
{
  "turn_index": 3,
  "current_differential": [
    {"dx": "migraine", "prob": 0.55},
    {"dx": "tension headache", "prob": 0.25},
    {"dx": "cluster headache", "prob": 0.10}
  ],
  "biggest_uncertainty": "string",
  "candidate_questions": ["q1", "q2", "q3"],
  "chosen_question": "string",
  "rationale": "string",
  "confidence_to_stop": 0.32,
  "should_stop": false,
  "safety_flags": []
}
```

### Final record (emitted on stop)

```json
{
  "session_id": "uuid",
  "started_at": "iso8601",
  "ended_at": "iso8601",
  "termination_reason": "confidence_threshold | max_turns | safety_stop | user_quit",
  "final_profile": { ... patient profile ... },
  "final_differential": [{"dx": "...", "prob": 0.X}],
  "primary_diagnosis": "string",
  "turn_history": [
    {
      "turn_index": 0,
      "doctor_output": { ... doctor turn output ... },
      "patient_answer": "string",
      "retrieved_exemplar_ids": ["ex_001", "ex_007", "ex_012"],
      "timestamp": "iso8601"
    }
  ],
  "model_metadata": {
    "model_name": "string",
    "model_version": "string",
    "prompt_template_version": "v1.0"
  }
}
```

### Exemplar (static, hand-curated initially)

```json
{
  "exemplar_id": "ex_001",
  "tags": ["cardiac", "chest_pain", "elderly"],
  "context_summary": "65yo M with chest pain, 2hr onset",
  "good_next_question": "Does the pain radiate to your arm or jaw?",
  "rationale": "Radiation pattern is critical for ACS vs MSK differentiation",
  "frozen": true
}
```

`frozen: true` exemplars are the human-curated seed set and never get evicted, even when Loop 2 starts adding machine-generated ones.

---

## Phases

Each phase has a goal, deliverable, exit criteria, and a debug checklist. Do not start a phase until the previous one's exit criteria are met.

### Phase 0 — Setup and decisions

**Goal:** Lock the tech choices and project skeleton before writing dialogue code.

**Deliverable:**
- Repo with directory structure (see below)
- `requirements.txt` / `pyproject.toml`
- Config file (`config.yaml`) for model choice, thresholds, paths
- Logging setup (JSONL files, one per session)
- Decision doc recording: which model, which embedding model, which vector store, why

**Decisions to lock:**
- Doctor LLM (recommend starting with Llama-3.x-Instruct or OpenBioLLM-8B for prototyping)
- Embedding model for retrieval (e.g. `BAAI/bge-small-en-v1.5` or similar)
- Vector store (FAISS for local, Chroma if you want persistence)
- JSON parsing strategy (Pydantic + retry on parse failure)
- Max turn cap (suggest 15)
- Confidence threshold to stop (suggest 0.75 — tune later)

**Suggested structure:**
```
/loop1
  /src
    /doctor       # prompt building, LLM call, JSON parsing
    /retrieval    # embedding, vector store, MMR
    /compression  # context summarization
    /session      # loop control, termination, logging
    /schemas      # Pydantic models for all locked schemas
  /exemplars      # hand-curated JSON files
  /test_cases     # patient profiles for manual testing
  /logs           # session JSONL output
  /prompts        # versioned prompt templates
  config.yaml
```

**Exit criteria:**
- `pip install -e .` works
- A "hello world" script can load the config, call the LLM, and log to a JSONL file
- All schemas are defined as Pydantic models with validation

---

### Phase 1 — Single-turn sanity

**Goal:** Produce one valid doctor turn from a hand-written patient profile.

**Deliverable:**
- Function: `generate_turn(profile, history=[]) -> DoctorTurnOutput`
- 5 hand-written test profiles in `/test_cases`
- Prompt template v0.1 saved in `/prompts`

**Implementation notes:**
- System prompt: define the doctor role, output schema, sentence-case for questions, no medical advice disclaimers (Loop 1 is sandboxed)
- For now, no retrieval and no exemplars — just the profile in, the schema out
- Use the LLM's structured output mode if available (function calling, JSON mode); fall back to parse-and-retry if not
- Validate every output against the Pydantic schema; if invalid, retry up to 3 times with the parse error fed back

**Exit criteria:**
- Run on all 5 test profiles, get valid JSON every time
- Manual review: the `chosen_question` is medically reasonable for the profile (eyeball test, no formal eval)
- Parse-failure rate under 5% across 50 runs

**Debug checklist if stuck:**
- JSON failing? Check if the model is wrapping output in markdown fences — strip them
- Bad questions? Look at your prompt — is the schema explained? Are you asking for reasoning before the question?
- Repetitive questions? Probably fine at this stage, you have no turn history yet

---

### Phase 2 — Multi-turn loop with manual patient input

**Goal:** Run a complete dialogue session with you typing the patient answers.

**Deliverable:**
- CLI script: `python -m loop1.session --profile test_cases/case_03.json`
- Loop terminates on `should_stop: true` or `max_turns` (set to 15)
- Full session logged to `/logs/session_{uuid}.jsonl` with one event per line
- Final record written to `/logs/final_{uuid}.json`

**Implementation notes:**
- Each turn appends to `turn_history` in working memory
- Pass full `turn_history` into the doctor prompt for now (compression comes in Phase 3)
- Update `patient_profile` after each patient answer — extract symptoms and history mentions into the structured fields. Can be a separate LLM call ("given this patient answer, what new info should be added to the profile?") that returns a profile delta
- Log every turn before asking for the next patient input (in case of crash, you don't lose data)

**Exit criteria:**
- Run 10 full sessions end-to-end. None crash. All emit a valid final record.
- Termination: at least 6 of 10 should stop via `confidence_to_stop`, not just `max_turns`. If everything is hitting the cap, the model isn't gaining confidence — investigate before moving on.
- Manual review: at turn 6+, questions are still relevant and non-repetitive

**Debug checklist if stuck:**
- Questions repeat across turns? Confirm `turn_history` is actually being included in the prompt
- Confidence never crosses threshold? Probably model is sandbagging — try lowering threshold to 0.6 temporarily and see if it converges. If it does, threshold tuning. If not, prompt issue.
- Profile not updating with new info? Test the delta-extraction call separately on a fixed patient answer
- Out-of-distribution diagnoses appearing late in dialogue? The differential is drifting — common sign of context confusion. Phase 3 will help.

---

### Phase 3 — Context compression

**Goal:** Keep the doctor coherent past turn 10 by summarizing older turns instead of passing all of them raw.

**Deliverable:**
- Function: `compress_context(turn_history, keep_recent=3) -> CompressedContext`
- Strategy: Keep last N turns verbatim, summarize earlier turns into a structured running summary
- Summary lives inside the patient_profile (`free_notes` field or a new `running_summary` field)
- Token-count monitoring per turn

**Implementation notes:**
- Compression is itself an LLM call. Prompt it to extract: key symptoms confirmed, key conditions ruled in/out, important timeline facts, anything unusual
- Decide compression cadence: compress every turn vs every N turns. Recommend compress at turn 4+, summarize turns 0 through (current - 3)
- Log both the raw history and the compressed version in the session log so you can compare

**Exit criteria:**
- Run a 15-turn dialogue. Prompt token count at turn 15 is no more than 1.5× the token count at turn 5.
- Manual review on 5 long dialogues: doctor still references info from early turns correctly (e.g. "you mentioned the headache started 3 days ago" at turn 12).
- A/B sanity check: pick one dialogue, run the last turn twice — once with full raw history, once with compressed. Outputs should be substantively similar.

**Debug checklist if stuck:**
- Doctor forgets old info? Compression summary is too lossy — add structured fields for the things being forgotten
- Doctor invents info that wasn't in history? Compression is hallucinating — temperature down, clearer "extract only, don't infer" instruction
- Token count not dropping? Compression isn't actually being inserted in place of raw history — trace the prompt construction

---

### Phase 4 — Static few-shot ICL

**Goal:** Improve question quality by injecting 3 hand-curated exemplars in every prompt.

**Deliverable:**
- 15-20 hand-written exemplars in `/exemplars`, covering diverse scenarios:
  - Vague chief complaint clarification
  - Red-flag screening (chest pain, severe headache, etc.)
  - Timeline pinning
  - Differential narrowing
  - Knowing when to stop
  - Pediatric / geriatric variations
  - Emergency escalation
- Updated prompt template (v0.2) that includes 3 randomly selected exemplars
- Comparison run: 10 sessions on v0.1 (no ICL) vs v0.2 (static ICL), same patient profiles

**Implementation notes:**
- Exemplars should show the full CoT structure, not just the question — include the rationale
- Mark all initial exemplars as `frozen: true`
- For Phase 4, random selection is fine (dynamic retrieval is Phase 5)

**Exit criteria:**
- Manual blind review of 10 paired outputs (v0.1 vs v0.2): v0.2 wins on at least 6 of 10 for question quality
- No regression in JSON validity rate
- No new failure modes (e.g. exemplar copying — doctor literally repeating an exemplar's question word-for-word). If you see copying, exemplars are too prescriptive — emphasize rationale, downplay exact wording.

**Debug checklist if stuck:**
- ICL hurts quality? Exemplars are bad. Symptoms: questions feel forced or off-topic. Fix exemplars, don't blame the model.
- Doctor copies exemplars verbatim? Add more variety, or rewrite exemplars to focus on reasoning patterns over specific phrasing
- No measurable improvement? Try 5 exemplars instead of 3, or check if your test cases are too easy to differentiate

---

### Phase 5 — Dynamic retrieval

**Goal:** Replace random exemplar selection with similarity-based retrieval, plus MMR for diversity.

**Deliverable:**
- Embed all exemplars at startup, store in FAISS/Chroma
- Function: `retrieve_exemplars(current_profile, k=3, mmr_lambda=0.5) -> List[Exemplar]`
- Query construction: embed a concatenation of `chief_complaint + top 3 symptoms + age/sex`
- MMR implementation: top-k by similarity, then re-rank for diversity
- Updated prompt template v0.3

**Implementation notes:**
- Query at every turn (the profile changes, so retrieval should adapt)
- MMR lambda: 0.5 is a reasonable starting point (balance similarity and diversity); tune later
- Log which exemplar IDs were retrieved each turn — this matters for Loop 2 and for debugging

**Exit criteria:**
- Manual review on 10 queries: retrieved exemplars are topically relevant (not just one famous exemplar dominating)
- Diversity check: across 20 different patient profiles, at least 10 different exemplars appear in top-3. If only 3-4 exemplars are ever retrieved, your bank is unbalanced or MMR is broken.
- Comparison run: v0.2 (random ICL) vs v0.3 (dynamic ICL) on 15 sessions. v0.3 should win on at least 9 of 15.

**Debug checklist if stuck:**
- Same exemplars always returned? Either the bank is dominated by one topic, or MMR is off — verify with cosine scores
- Retrieved exemplars off-topic? Bad query construction — try different fields, or use a separate query-rewrite LLM call
- Dynamic worse than static? Either the bank is too small (need 30+ exemplars for retrieval to shine), or your test patients are all similar to the same few exemplars

---

### Phase 6 — Hardening and logging completeness

**Goal:** Make Loop 1 robust enough to collect real session data that Loop 2 will consume.

**Deliverable:**
- Crash recovery: if a session crashes mid-dialogue, the JSONL log is replayable
- Schema versioning: every log entry has a `schema_version` field
- Safety net: detect emergency-flag phrases in patient input ("chest pain", "suicidal thoughts", etc.) and trigger an escalation message instead of normal flow
- Disclaimer: every session starts with "this is a research prototype, not medical advice"
- Session metadata: model name, model version, prompt template version, retrieval config — all stored with each record
- A `--replay` mode that reads a logged session and re-runs the doctor on it (useful for prompt iteration without burning real patient time)

**Exit criteria:**
- Run 30 sessions. Zero crashes. Zero malformed final records.
- Replay mode reproduces a previous session deterministically (with temperature=0)
- Safety triggers fire on a test suite of emergency phrases

**Debug checklist if stuck:**
- Replay non-deterministic? Either temperature isn't 0, or you have unseeded randomness in retrieval (MMR can have ties — sort deterministically)
- Logs missing fields? Schema_version mismatch — bump version and migrate
- Safety triggers too aggressive (false positives)? Move from string matching to a small classifier call, or tune the trigger list

---

## What you have at the end of Loop 1

- A working multi-turn diagnostic dialogue system using ICL + CoT
- A growing corpus of logged sessions in a stable, training-ready format
- Confidence that the system behaves predictably across diverse cases
- Everything Loop 2 needs to begin: scored records, exemplar bank ready to grow, prompt templates versioned

You can stop here and have a useful prototype. You can demo this. You can run small user studies. The architecture you built supports Loop 2 whenever you're ready.

---

## Decisions deferred to Loop 2

- Critic LLM design and prompt
- N-rollout sampling strategy per case
- Preference pair construction
- DPO training pipeline
- LoRA adapter management and versioning
- Exemplar bank growth from high-scoring records (with frozen seed set preserved)

---

## Engineering notes

- **Version your prompts.** Every prompt template gets a version string. Old logs reference the version they were generated with. This will save you when you change a prompt and old behavior disappears.
- **Don't optimize prematurely.** Phase 1-3 can use the model however is easiest. Save batching, caching, async for Phase 6 if at all.
- **Eyeballing beats metrics until Phase 5.** You don't have enough data for statistical claims; trust your medical-reasonableness judgment on small samples.
- **Keep one notebook of weird outputs.** When the doctor does something strange — invents a symptom, asks something wildly off-topic, hits a JSON edge case — log it. These become regression tests later.
- **Set a Phase 0 deadline before writing code.** It's easy to spend two weeks "deciding" on models. One week, then commit.