from __future__ import annotations

import json
import logging
import re

from loop1.config import config
from loop1.llm import call_llm
from loop1.schemas import ClosingTurn, DiagnosisEntry, PatientProfile

_log = logging.getLogger(__name__)

_CLOSING_SYSTEM_PROMPT = """You are concluding a diagnostic consultation. Write a closing statement addressed directly to the patient in plain, jargon-free language.

Return ONLY this JSON object — no markdown fences, no prose:

{
  "leading_diagnosis": "string — one plain-language sentence naming the most likely diagnosis",
  "differential_summary": "string — brief mention of top 2-3 possibilities in plain language. Do NOT include percentages or probability numbers.",
  "recommended_next_steps": ["list of concrete next steps the patient should take"],
  "unresolved_patient_questions": ["list of questions the patient raised that were not fully answered — empty list if none"]
}

Rules:
- Speak to the patient, not about them.
- No Latin terms, abbreviations, or clinical shorthand.
- recommended_next_steps must be actionable (e.g., "See your GP within 48 hours" not "Follow up").
- If the diagnosis is uncertain, say so plainly.
- unresolved_patient_questions: only include questions the patient actually asked during the session.
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def generate_closing_turn(
    profile: PatientProfile,
    final_differential: list[DiagnosisEntry],
    model: str | None = None,
) -> ClosingTurn | None:
    """
    Generate a plain-language closing statement for the patient.
    Returns None on any failure (API or parse error) — caller must not crash.
    """
    if model is None:
        model = config["models"]["doctor"]

    differential_text = ", ".join(
        f"{d.dx} ({d.prob:.0%})" for d in final_differential[:3]
    )
    profile_summary = json.dumps(
        {
            "chief_complaint": profile.chief_complaint,
            "symptoms": [s.model_dump() for s in profile.symptoms],
            "ruled_in": profile.ruled_in,
            "ruled_out": profile.ruled_out,
            "running_summary": profile.running_summary,
        },
        indent=2,
    )

    user_content = (
        f"## Final differential\n{differential_text}\n\n"
        f"## Patient profile summary\n{profile_summary}\n\n"
        "Now produce the closing statement JSON."
    )

    messages = [
        {"role": "system", "content": _CLOSING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    max_retries = 2
    for attempt in range(max_retries):
        try:
            raw = call_llm(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=2048,
            )
        except Exception as exc:
            _log.warning("Closing turn LLM call failed (attempt %d): %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                return None
            continue

        cleaned = _strip_fences(raw)
        try:
            data = json.loads(cleaned)
            return ClosingTurn(
                leading_diagnosis=data["leading_diagnosis"],
                differential_summary=data["differential_summary"],
                recommended_next_steps=list(data.get("recommended_next_steps", [])),
                unresolved_patient_questions=list(data.get("unresolved_patient_questions", [])),
                generated_by=model,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            _log.warning("Closing turn parse error (attempt %d): %s", attempt + 1, exc)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Parse failed on attempt {attempt + 1}: {exc}. "
                        "Reply with valid JSON only — no fences, no prose."
                    ),
                },
            ]

    return None
