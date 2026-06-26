from __future__ import annotations

import re

EMERGENCY_PHRASES: list[str] = [
    "can't breathe",
    "cannot breathe",
    "face drooping",
    "unresponsive",
    "severe bleeding",
    "won't stop bleeding",
    "suicidal",
    "want to die",
    "kill myself",
    "throat closing",
    "throat swelling",
    "seizure",
    "overdose",
]

_EMERGENCY_PATTERN = re.compile(
    "|".join(re.escape(p) for p in EMERGENCY_PHRASES),
    re.IGNORECASE,
)

EMERGENCY_MESSAGE = (
    "⚠️  Emergency language detected. "
    "Please call emergency services (112/911) immediately.\n"
    "This system is not a substitute for emergency medical care."
)


def check_safety(patient_input: str) -> str | None:
    """Return the first matched emergency phrase (case-insensitive), or None."""
    match = _EMERGENCY_PATTERN.search(patient_input)
    if match:
        matched_text = match.group(0).lower()
        for phrase in EMERGENCY_PHRASES:
            if phrase.lower() == matched_text:
                return phrase
        return matched_text
    return None
