from __future__ import annotations

import logging

from loop3.routing.ambiguity_handler import get_routing_options, is_ambiguous
from loop3.routing.map_loader import load_disease_specialty_map
from loop3.routing.routing_schema import (
    ROUTING_MODEL_VERSION,
    RoutingDecision,
    RoutingOption,
    UrgencyLevel,
)

_log = logging.getLogger(__name__)

_URGENCY_RANK = {
    UrgencyLevel.EMERGENCY: 0,
    UrgencyLevel.URGENT: 1,
    UrgencyLevel.ROUTINE: 2,
}


def _highest_urgency_first(options: list[RoutingOption]) -> list[RoutingOption]:
    """
    Sort options so the highest urgency comes first.
    Safety first: an emergency always surfaces as primary regardless of probability weight.
    Ties broken by confidence_weight descending.
    """
    return sorted(options, key=lambda o: (_URGENCY_RANK[o.urgency], -o.confidence_weight))


def route(
    session_id: str,
    final_differential: list[tuple[str, float]],
    final_confidence: float,
    patient_id: str | None = None,
) -> RoutingDecision:
    """
    Main routing entry point.

    Takes the final differential from a Loop 1 session and returns a
    RoutingDecision with specialty, urgency, appointment type, and ambiguity flag.

    The highest-urgency option is always primary_routing — safety first.
    """
    specialty_map = load_disease_specialty_map()

    options = get_routing_options(
        differential=final_differential,
        specialty_map=specialty_map,
        final_confidence=final_confidence,
    )

    options = _highest_urgency_first(options)

    ambiguous = is_ambiguous(final_differential, specialty_map)

    primary = options[0]
    alternatives = options[1:]

    _log.info(
        "Routing session %s → %s (%s)%s",
        session_id,
        primary.specialty,
        primary.urgency.value,
        " [AMBIGUOUS]" if ambiguous else "",
    )

    return RoutingDecision(
        session_id=session_id,
        patient_id=patient_id,
        is_ambiguous=ambiguous,
        primary_routing=primary,
        alternative_routings=alternatives,
        final_differential=final_differential,
        final_confidence=final_confidence,
        routing_model_version=ROUTING_MODEL_VERSION,
    )
