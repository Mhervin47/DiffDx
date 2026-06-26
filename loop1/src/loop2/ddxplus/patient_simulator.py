from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from loop1.llm import call_llm
from loop2.ddxplus.schemas import DDXPlusPatient

_log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "prompts" / "simulated_patient_v0_1.txt"
_SIMULATOR_MODEL = "groq/llama-3.1-8b-instant"
_MAX_LEAKAGE_RETRIES = 3


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _build_patient_record_json(record: DDXPlusPatient) -> str:
    """Build a simulator-readable JSON view of the patient record (no ground truth)."""
    return json.dumps(
        {
            "age": record.age,
            "sex": record.sex,
            "initial_complaint": record.initial_evidence,
            "symptoms": record.symptoms,
            "antecedents": record.antecedents,
        },
        indent=2,
    )


def _leakage_tokens(pathology: str) -> list[str]:
    """
    Return significant words from the pathology name that must not appear in responses.
    Filters out short/common words (len <= 3).
    """
    words = re.split(r"[\s/\-()]+", pathology.lower())
    return [w for w in words if len(w) > 3]


def _check_leakage(response: str, pathology: str) -> bool:
    """Return True if the response contains any significant token from the pathology name."""
    tokens = _leakage_tokens(pathology)
    lower = response.lower()
    return any(tok in lower for tok in tokens)


class SimulatedPatient:
    """
    LLM-backed patient simulator grounded in a DDXPlus patient record.

    Guardrails:
    - Never invents symptoms not in the record
    - Never reveals the ground-truth pathology name
    - Temperature 0 for reproducibility
    - Tracks disclosed symptoms across the session
    """

    def __init__(self, record: DDXPlusPatient) -> None:
        self.record = record
        self.disclosed_symptoms: set[str] = set()
        self._conversation_history: list[tuple[str, str]] = []
        self._prompt_template = _load_prompt_template()
        self._patient_record_json = _build_patient_record_json(record)

    def _render_history(self) -> str:
        if not self._conversation_history:
            return "(none yet)"
        lines: list[str] = []
        for doctor_q, patient_a in self._conversation_history:
            lines.append(f"Doctor: {doctor_q}")
            lines.append(f"Patient: {patient_a}")
        return "\n".join(lines)

    def _call_simulator(self, doctor_question: str) -> str:
        prompt = (
            self._prompt_template
            .replace("{patient_record_json}", self._patient_record_json)
            .replace("{conversation_history}", self._render_history())
            .replace("{doctor_question}", doctor_question)
        )
        messages = [{"role": "user", "content": prompt}]
        return call_llm(
            model=_SIMULATOR_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=256,
        )

    def initial_complaint(self) -> str:
        """
        Return a natural-language opening statement based on the patient's initial evidence.
        Records the response in conversation history.
        """
        question = (
            "Please describe why you are here today — what brought you to see the doctor?"
        )
        for attempt in range(_MAX_LEAKAGE_RETRIES):
            response = self._call_simulator(question)
            if not _check_leakage(response, self.record.ground_truth_pathology):
                self._conversation_history.append((question, response))
                return response
            _log.warning(
                "Leakage detected in initial_complaint (attempt %d/%d), regenerating.",
                attempt + 1,
                _MAX_LEAKAGE_RETRIES,
            )

        # Final attempt: use a safe fallback derived directly from initial_evidence
        fallback = (
            f"I've been having {self.record.initial_evidence.lower().rstrip('?')}. "
            "It's been bothering me and I'm worried."
        )
        _log.warning("Using fallback initial_complaint after %d leakage retries.", _MAX_LEAKAGE_RETRIES)
        self._conversation_history.append((question, fallback))
        return fallback

    def answer(self, doctor_question: str) -> str:
        """
        Answer a doctor's question grounded in the patient record.
        Never leaks the ground-truth pathology name.
        Updates conversation history and disclosed_symptoms.
        """
        for attempt in range(_MAX_LEAKAGE_RETRIES):
            response = self._call_simulator(doctor_question)
            if not _check_leakage(response, self.record.ground_truth_pathology):
                self._conversation_history.append((doctor_question, response))
                self._update_disclosed(response)
                return response
            _log.warning(
                "Leakage detected in answer (attempt %d/%d), regenerating.",
                attempt + 1,
                _MAX_LEAKAGE_RETRIES,
            )

        # Fallback: safe generic "I don't know what's wrong" response
        fallback = "I'm not sure, I just don't feel well. Can you tell me what you think it might be?"
        _log.warning("Using fallback answer after %d leakage retries.", _MAX_LEAKAGE_RETRIES)
        self._conversation_history.append((doctor_question, fallback))
        return fallback

    def _update_disclosed(self, response: str) -> None:
        """Track which symptom names from the record appear in the response."""
        lower = response.lower()
        for symptom_name in self.record.symptoms:
            # Match on significant words from symptom name
            tokens = [w for w in re.split(r"[\s/\-(),:?]+", symptom_name.lower()) if len(w) > 3]
            if tokens and all(tok in lower for tok in tokens[:2]):
                self.disclosed_symptoms.add(symptom_name)
