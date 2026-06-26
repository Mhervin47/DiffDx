# DiffDx

An AI-powered medical diagnostic assistant that uses an actor–critic architecture to conduct and evaluate multi-turn diagnostic conversations.

## Overview

DiffDx simulates a clinical diagnostic dialogue. A patient describes their symptoms through a web portal, and an AI doctor conducts a structured conversation to narrow down a differential diagnosis — asking follow-up questions, tracking probabilities, and knowing when it has enough confidence to stop.

The system is built in two layers:

- **Actor (Loop 1)** — the doctor LLM that drives the conversation
- **Critic (Loop 2)** — an evaluator LLM that scores the actor's performance and generates training signal

## How the Actor works

The actor is a Groq-hosted **Llama 3.3 70B** model prompted to behave as a diagnostic physician. Every turn it receives:

- The patient's structured profile (age, sex, chief complaint, symptoms, history)
- The full conversation history (compressed after turn 3 to avoid prompt bloat)
- 3 retrieved few-shot exemplars showing good diagnostic reasoning
- The current differential diagnosis with probabilities

It outputs strict JSON every turn:

```json
{
  "current_differential": [{ "dx": "migraine", "prob": 0.6 }],
  "biggest_uncertainty": "...",
  "chosen_question": "...",
  "rationale": "...",
  "confidence_to_stop": 0.71,
  "should_stop": false
}
```

The session ends when `confidence_to_stop` crosses **0.75** or after **15 turns** maximum.

**Context compression** — after the first few turns, older turns are summarised into a running note rather than passed raw, keeping the prompt size stable across long sessions.

**Exemplar retrieval** — at each turn, the 3 most relevant hand-curated exemplars are retrieved using sentence embeddings and MMR (maximum marginal relevance) for diversity, then injected into the prompt as few-shot demonstrations.

## How the Critic works

After a session ends, the critic LLM reviews the full session log and scores the actor on:

- Question relevance and efficiency
- Whether the differential evolved sensibly across turns
- Whether the stopping decision was appropriate
- Safety flag detection (did it catch red-flag symptoms?)

These scores are stored alongside the session. High-scoring sessions become new exemplars. High/low score pairs become preference pairs for future **DPO fine-tuning** of the actor.

## Patient and Doctor portals

The web app wraps this engine in a clinical workflow:

- **Patients** — register, book appointments with a doctor, upload reports (PDFs, images), start a diagnostic session, request a second opinion
- **Doctors** — manage their availability calendar, view upcoming and past appointments, read AI-generated session summaries, flag cases for review

## Tech stack

| Layer | Technology |
| --- | --- |
| Backend API | FastAPI (Python) |
| Frontend | Vanilla HTML/CSS/JS |
| Actor LLM | Groq — llama-3.1-8b-instant |
| Critic LLM | OpenRouter — Gemma 4 31B (free tier) |
| Embeddings | Sentence Transformers + FAISS |
| Database | PostgreSQL (Supabase) / SQLite locally |
| Deployment | Render |

## Running locally

```bash
cd loop1
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # add GROQ_API_KEY at minimum
PYTHONPATH=src python run_web.py
```

Then open <http://localhost:8000>.

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | Powers the actor LLM |
| `DATABASE_URL` | No | PostgreSQL URL — uses SQLite locally if unset |
| `RESEND_API_KEY` | No | Appointment confirmation emails |

## Deployment

Configured for Render via `render.yaml`. Connect the `Mhervin47/DiffDx` GitHub repo, set the environment variables in the Render dashboard, and deploy. Auto-deploys on every push to `main`.
