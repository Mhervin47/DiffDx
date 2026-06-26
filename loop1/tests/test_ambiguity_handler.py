"""Tests for the emergency-surfacing rule added to the ambiguity handler."""
from loop3.routing.ambiguity_handler import get_routing_options, is_ambiguous
from loop3.routing.map_loader import load_disease_specialty_map
from loop3.routing.routing_schema import UrgencyLevel


def test_dual_emergency_specialties_both_surface():
    """
    case_01 scenario: Cardiology (0.60 weight) and Pulmonology (0.40 weight) are both
    emergency. The probability threshold alone wouldn't flag ambiguity (0.60 == 1.5 × 0.40),
    but the emergency-surfacing rule must include both and set is_ambiguous=True.
    """
    specialty_map = load_disease_specialty_map()
    diff = [
        ("Acute Coronary Syndrome", 0.55),  # Cardiology — emergency
        ("Aortic Dissection", 0.05),         # Cardiology — emergency
        ("Pulmonary embolism", 0.30),        # Pulmonology — emergency
        ("Pneumonia", 0.10),                 # Pulmonology — urgent
    ]

    options = get_routing_options(diff, specialty_map, final_confidence=0.4)
    specialties = [o.specialty for o in options]

    assert "Cardiology" in specialties
    assert "Pulmonology" in specialties
    assert is_ambiguous(diff, specialty_map) is True
    assert options[0].specialty == "Cardiology"   # higher weight stays primary
    assert options[0].urgency == UrgencyLevel.EMERGENCY
    assert options[1].urgency == UrgencyLevel.EMERGENCY
