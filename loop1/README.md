# loop1

AI medical dialogue agent for structured patient intake.

## Setup

```bash
# 1. Copy the env template and add your key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=<your key from console.groq.com>

# 2. Install dependencies (Python 3.11+, uv required)
uv sync

# 3. Sanity check — one LLM call + one log event
uv run python scripts/hello_world.py

# 4. Run tests
uv run pytest
```

## Running a full session

```bash
uv run python scripts/run_session.py --profile test_cases/case_01.json
```

## Phase status

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Setup, schemas, hello world | ✅ |
| 1 | Single-turn `generate_turn()` | ⬜ |
| 2 | Multi-turn CLI loop + logging | ⬜ |
| 3 | Context compression | ⬜ |
| 4 | Static ICL exemplars | ⬜ |
| 5 | Dynamic FAISS retrieval + MMR | ⬜ |
| 6 | Hardening + replay mode | ⬜ |
