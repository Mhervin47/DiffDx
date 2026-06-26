# MedicalConvo — Phase Notes (Website)

## What has been built

Loop 1 is a working multi-turn diagnostic AI. A doctor LLM interviews a patient, builds a differential diagnosis turn-by-turn, and a critic LLM scores each question in real time.

### Demo flow (working end-to-end)

```
loop1/demo_with_critic.py
```

1. Loads `test_cases/case_01.json` — 58M, chest pain + SOB 2hrs (ACS/PE presentation)
2. Generates a fresh session UUID each run
3. Session runs for up to 6 turns (confidence threshold stops it earlier if confident)
4. After each patient answer → critic scores that turn immediately (real-time)
5. At session end → writes two JSON files:
   - `logs/final_records/final_{uuid}.json` — doctor's full session record
   - `logs/critic_reports/critic_{uuid}.json` — critic's full report

### Output files

#### `logs/final_records/final_{uuid}.json`
```json
{
  "session_id": "...",
  "started_at": "iso8601",
  "ended_at": "iso8601",
  "termination_reason": "confidence_threshold | max_turns",
  "primary_diagnosis": "Pulmonary Embolism",
  "final_differential": [
    {"dx": "Pulmonary Embolism", "prob": 0.60},
    {"dx": "ACS", "prob": 0.25},
    {"dx": "DVT", "prob": 0.10}
  ],
  "turn_history": [
    {
      "turn_index": 0,
      "doctor_output": {
        "chosen_question": "Do you have any swelling in your legs?",
        "current_differential": [...],
        "confidence_to_stop": 0.28,
        "rationale": "..."
      },
      "patient_answer": "yes, my left leg is swollen"
    }
  ],
  "closing_turn": {
    "leading_diagnosis": "plain language sentence",
    "differential_summary": "top 2-3 possibilities",
    "recommended_next_steps": ["Go to ER now", "..."]
  }
}
```

#### `logs/critic_reports/critic_{uuid}.json`
```json
{
  "schema_version": "0.7.0",
  "session_id": "...",
  "patient": {"age": 58, "sex": "M", "chief_complaint": "..."},
  "final_diagnosis": "Pulmonary Embolism",
  "termination_reason": "confidence_threshold",
  "total_turns": 4,
  "aggregate": {
    "mean_question_quality": 0.90,
    "mean_differential_quality": 0.58,
    "mean_reasoning_quality": 0.85,
    "weakness_category_counts": {"poor_differential": 3}
  },
  "turns": [
    {
      "turn": 0,
      "question_quality_score": 0.90,
      "differential_quality_score": 0.40,
      "reasoning_quality_score": 0.80,
      "confidence_calibration": "underconfident",
      "weakness_category": "poor_differential",
      "weakness": "ACS ranked above PE despite leg swelling and recent travel",
      "would_have_asked": null,
      "rationale": "..."
    }
  ]
}
```

---

## Tech stack (backend)

| Component | Model | Provider |
|---|---|---|
| Doctor | `llama-3.3-70b-versatile` | Groq |
| Profile updater | `llama-3.3-70b-versatile` | Groq |
| Compressor | `llama-3.3-70b-versatile` | Groq |
| Critic | `llama-3.3-70b-versatile` | Groq |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` | Local |

Fallback doctor model if Groq 70b TPD runs out: `cerebras/gpt-oss-120b` (needs `CEREBRAS_API_KEY` in `.env`)

Config file: `loop1/config.yaml`
Env file: `loop1/.env` (GROQ_API_KEY, CEREBRAS_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY)

---

## Key source files

```
loop1/
  demo_with_critic.py          ← main demo entry point
  config.yaml                  ← model names, thresholds, prompt versions
  .env                         ← API keys

  src/loop1/
    session.py                 ← main dialogue loop (Session class)
    doctor.py                  ← prompt building + LLM call for doctor turn
    llm.py                     ← unified LLM caller (Groq / Cerebras / OpenRouter routing)
    profile_updater.py         ← extract structured info from patient answer
    compressor.py              ← compress old turns into running summary
    closing_turn.py            ← generate plain-language closing statement
    schemas.py                 ← Pydantic models for all data structures
    logging_utils.py           ← write/read session JSONL logs

  src/loop2/
    critic/critic.py           ← critique_turn() — one LLM call per turn
    critic/critique_schema.py  ← TurnCritique Pydantic model
    critic/aggregator.py       ← aggregate_critiques() — roll-up stats

  prompts/
    doctor_v0_5.txt            ← current doctor prompt (v0.5)
    critic_turn_v0_1.txt       ← current critic prompt (v0.1)

  test_cases/
    case_01.json               ← 58M, chest pain + SOB (ACS/PE)
    case_02.json               ← 34F, headache + dizziness
    case_03.json               ← 46M, abdominal pain + diarrhea
    case_04.json               ← 7F, fever + rash
    case_05.json               ← 72M, "just don't feel right"
    case_06.json               ← fever + rash (3 days)

  logs/
    sessions/session_{uuid}.jsonl       ← turn-by-turn event log
    final_records/final_{uuid}.json     ← doctor's final record
    critic_reports/critic_{uuid}.json   ← critic's final report
```

---

## Website plan

### Goal
A web UI that lets a user run and watch a diagnostic session live — showing the doctor questions, patient answers, real-time critic scores, and a final report.

### Design tool
**Stitch** (stitch.withgoogle.com) — use it to generate the UI mockup/design. Export the HTML/CSS from Stitch and build the frontend on top of it.

### Pages / views

#### 1. Home / case select
- Brief explainer: "AI diagnostic assistant prototype"
- List of available test cases (case_01 through case_06) with chief complaint shown
- "Start session" button per case
- Disclaimer banner: "Research prototype — not medical advice"

#### 2. Session view (main page)
Left panel — conversation:
- Chief complaint at top
- Each turn: Doctor question (styled as doctor bubble) → Patient answer (text input or styled as patient bubble)
- After each patient answer: critic score bar appears inline below the turn
  - Q-quality | Diff-quality | Reasoning (color-coded green/yellow/red)
  - Weakness category badge if flagged
  - "Would have asked" shown if Q-quality < 0.7

Right panel — live differential:
- Ranked list of diagnoses with probability bars
- Updates after each turn (animate the bar changes)
- Color: top diagnosis highlighted

#### 3. Final report view
Shown after session ends (confidence threshold or max turns):
- Final diagnosis (large, prominent)
- Doctor's closing statement (plain language)
- Critic summary table (all turns, scores)
- Mean scores + weakness breakdown
- Export as PDF button (nice to have)

### API design (backend)

The website needs a simple REST/WebSocket API wrapping the existing Python session:

```
POST /api/session/start
  body: { case_id: "case_01" }
  returns: { session_id, chief_complaint, patient_demographics }

POST /api/session/{session_id}/turn
  body: { patient_answer: "yes my leg hurts" }
  returns: {
    doctor_question: "...",
    differential: [{dx, prob}, ...],
    confidence_to_stop: 0.55,
    session_complete: false,
    critique: {
      question_quality_score: 0.90,
      differential_quality_score: 0.40,
      weakness_category: "poor_differential",
      would_have_asked: null
    }
  }

GET /api/session/{session_id}/report
  returns: full critic_report JSON
```

Suggested backend framework: **FastAPI** (already Python, easy to wrap existing session code)

### Implementation order
1. Design in Stitch → export HTML/CSS
2. Build FastAPI backend wrapping `LiveCriticSession`
3. Wire up frontend to API (vanilla JS or lightweight React)
4. Deploy locally first, then optionally to Render/Railway

---

## Known issues / things to fix before website

- `gpt-oss-120b` (Cerebras) doesn't follow v0.5 differential updating rules — use Groq 70b as primary doctor
- Groq 70b daily limit: 100k TPD. For a demo/website, monitor usage. Each 6-turn session uses ~30-40k tokens.
- Critic Groq 70b: shares same 100k TPD pool as doctor. For 6-turn session, critic adds ~15-20k tokens. Total ~50-60k per session → ~2 full sessions per day on free tier.
- For more concurrent sessions, add Cerebras as doctor fallback or purchase Groq credits.

---

## Run command (current working demo)

```bash
cd /Users/mhervin/Desktop/MedicalConvo/loop1
PYTHONPATH=src .venv/bin/python3 demo_with_critic.py
```
