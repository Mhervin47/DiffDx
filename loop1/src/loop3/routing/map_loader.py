from __future__ import annotations

import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)

_DEFAULT_MAP_PATH = Path(__file__).parent.parent.parent.parent / "data" / "disease_specialty_map.json"

_REQUIRED_KEYS = {"specialty", "urgency", "appointment_type", "reasoning"}

_map_cache: dict | None = None


def load_disease_specialty_map(path: str | Path = _DEFAULT_MAP_PATH) -> dict:
    """Load and validate the disease → specialty mapping from JSON."""
    global _map_cache
    path = Path(path)

    if _map_cache is not None and path == _DEFAULT_MAP_PATH:
        return _map_cache

    if not path.exists():
        raise FileNotFoundError(f"Disease specialty map not found: {path}")

    with open(path, encoding="utf-8") as f:
        data: dict = json.load(f)

    missing_keys: list[str] = []
    for disease, entry in data.items():
        absent = _REQUIRED_KEYS - set(entry.keys())
        if absent:
            missing_keys.append(f"{disease!r}: missing {absent}")

    if missing_keys:
        raise ValueError(
            f"Disease specialty map has invalid entries:\n" + "\n".join(missing_keys)
        )

    if path == _DEFAULT_MAP_PATH:
        _map_cache = data

    return data


# Common medical abbreviations → canonical map keys (case-insensitive matching handles the rest)
_ALIASES: dict[str, str] = {
    "pe": "Pulmonary embolism",
    "pulmonary embolus": "Pulmonary embolism",
    "mi": "Possible NSTEMI / STEMI",
    "nstemi": "Possible NSTEMI / STEMI",
    "stemi": "Possible NSTEMI / STEMI",
    "heart attack": "Possible NSTEMI / STEMI",
    "acs": "Possible NSTEMI / STEMI",
    "acute coronary syndrome": "Possible NSTEMI / STEMI",
    "myocardial infarction": "Possible NSTEMI / STEMI",
    "afib": "Atrial fibrillation",
    "af": "Atrial fibrillation",
    "a-fib": "Atrial fibrillation",
    "copd exacerbation": "Acute COPD exacerbation / infection",
    "copd": "Acute COPD exacerbation / infection",
    "asthma": "Bronchospasm / acute asthma exacerbation",
    "asthma attack": "Bronchospasm / acute asthma exacerbation",
    "uti": "Urinary tract infection",
    "dvt": "DVT",
    "deep vein thrombosis": "DVT",
    "gerd": "GERD",
    "acid reflux": "GERD",
    "gbs": "Guillain-Barré syndrome",
    "guillain barre": "Guillain-Barré syndrome",
    "sle": "SLE",
    "lupus": "SLE",
    "tb": "Tuberculosis",
    "appendix rupture": "Ruptured appendix",
    "ruptured appendix": "Ruptured appendix",
    "panic disorder": "Panic attack",
    "hiv": "HIV (initial infection)",
    "pah": "Pulmonary arterial hypertension",
    "psvt": "PSVT",
    "svt": "PSVT",
    "ckd": "Chronic kidney disease",
    "dm": "Diabetes mellitus",
    "diabetes": "Diabetes mellitus",
    "hypothyroid": "Hypothyroidism",
    "hyperthyroid": "Hyperthyroidism",
    "pneumothorax": "Spontaneous pneumothorax",
    "ptx": "Spontaneous pneumothorax",
    "anaphylactic shock": "Anaphylaxis",
    "allergic reaction": "Anaphylaxis",
    "epiglottis inflammation": "Epiglottitis",
    "cauda equina": "Cauda equina syndrome",
    "cvst": "CVST",
    "cerebral venous sinus thrombosis": "CVST",
    "celiac": "Celiac disease",
    "crohns": "Crohn's disease",
    "crohn's": "Crohn's disease",
    "inflammatory bowel disease": "Crohn's disease",
    "ibd": "Crohn's disease",
    "pancreatitis": "Acute pancreatitis",
    "boerhaave": "Boerhaave syndrome",
    "esophageal rupture": "Boerhaave syndrome",
    "myocarditis": "Myocarditis",
    "pericarditis": "Pericarditis",
    "pulmonary edema": "Acute pulmonary edema",
    "flash pulmonary edema": "Acute pulmonary edema",
    "pulmonary fibrosis": "Pulmonary fibrosis",
    "ipf": "Pulmonary fibrosis",
    "sarcoid": "Sarcoidosis",
    "vasculitis": "Systemic vasculitis",
    "myasthenia": "Myasthenia gravis",
    "mg": "Myasthenia gravis",
    "nephrotic": "Nephrotic syndrome",
    "nephritis": "Acute nephritis",
    "pyelonephritis": "Pyelonephritis",
    "kidney infection": "Pyelonephritis",
    "biliary colic": "Biliary colic",
    "gallstone": "Biliary colic",
    "cholecystitis": "Biliary colic",
    "sinusitis": "Acute rhinosinusitis",
    "otitis media": "Acute otitis media",
    "ear infection": "Acute otitis media",
    "cellulitis": "Cellulitis",
    "phlebitis": "Phlebitis",
    "anemia": "Anemia",
    "anaemia": "Anemia",
}


_STRIP_PREFIXES = (
    "acute ", "chronic ", "possible ", "probable ", "suspected ", "likely ",
    "early ", "severe ", "mild ", "moderate ", "new-onset ", "new onset ",
    "recurrent ", "bilateral ", "unilateral ",
)


def _candidates(disease: str) -> list[str]:
    """Generate normalized variants of a disease name by stripping common prefixes."""
    base = disease.strip().lower()
    results = [base]
    for prefix in _STRIP_PREFIXES:
        if base.startswith(prefix):
            results.append(base[len(prefix):])
    return results


def lookup_disease(disease: str, specialty_map: dict) -> dict | None:
    """
    Look up a disease in the specialty map with four levels of matching:
    1. Exact case-insensitive match (including prefix-stripped variants)
    2. Alias table (including prefix-stripped variants)
    3. Bidirectional substring match
    Returns the entry dict or None if not found.
    """
    for candidate in _candidates(disease):
        # Level 1: exact case-insensitive
        for key, value in specialty_map.items():
            if key.strip().lower() == candidate:
                return value

        # Level 2: alias table
        canonical = _ALIASES.get(candidate)
        if canonical:
            for key, value in specialty_map.items():
                if key.strip().lower() == canonical.lower():
                    return value

    # Level 3: bidirectional substring — map key inside disease name, or vice versa
    normalized = disease.strip().lower()
    best_match: dict | None = None
    best_len = 0
    for key, value in specialty_map.items():
        key_norm = key.strip().lower()
        # Skip very short keys to avoid spurious matches
        if len(key_norm) < 5:
            continue
        if key_norm in normalized or normalized in key_norm:
            if len(key_norm) > best_len:
                best_match = value
                best_len = len(key_norm)

    return best_match
