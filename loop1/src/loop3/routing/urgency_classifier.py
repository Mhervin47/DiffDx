from __future__ import annotations

import logging

from loop3.routing.map_loader import lookup_disease
from loop3.routing.routing_schema import UrgencyLevel

_log = logging.getLogger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 0.4


def classify_urgency(
    disease: str,
    confidence: float,
    specialty_map: dict,
) -> UrgencyLevel:
    """
    Return the urgency level for a disease given final diagnostic confidence.

    Rules:
    - If disease not in map: return ROUTINE and log a warning.
    - If urgency is EMERGENCY: always return EMERGENCY regardless of confidence.
    - If urgency is ROUTINE and confidence < 0.4: bump to URGENT
      (low confidence means a human must confirm — don't let it slide to routine).
    - Otherwise: return map value as-is.
    """
    entry = lookup_disease(disease, specialty_map)

    if entry is None:
        _log.warning(
            "Disease %r not found in specialty map — falling back to ROUTINE urgency.",
            disease,
        )
        return UrgencyLevel.ROUTINE

    raw_urgency = UrgencyLevel(entry["urgency"])

    if raw_urgency == UrgencyLevel.EMERGENCY:
        return UrgencyLevel.EMERGENCY

    if raw_urgency == UrgencyLevel.ROUTINE and confidence < _LOW_CONFIDENCE_THRESHOLD:
        _log.debug(
            "Low confidence (%.2f) for routine disease %r — bumping to URGENT.",
            confidence,
            disease,
        )
        return UrgencyLevel.URGENT

    return raw_urgency
