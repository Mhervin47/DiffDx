from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from loop2.ddxplus.schemas import DDXPlusPatient

_DDXPLUS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "DDXPlus_Raw"

_SPLIT_FILE = {
    "test": "release_test_patients",
    "train": "release_train_patients",
    "validate": "release_validate_patients",
}

# Diseases covered by the Phase 6 exemplar pool — used for in_exemplar_pool flag
EXEMPLAR_POOL_DISEASES: frozenset[str] = frozenset({
    "CVST",
    "Measles",
    "Pulmonary embolism",
    "Appendicitis",
    "Cellulitis",
    "Pericarditis",
    "Biliary colic",
    "Cauda equina syndrome",
    "DVT",
    "Inflammatory back",
    "Pregnancy",
    "SLE",
    "Possible NSTEMI / STEMI",  # stimulant ACS exemplar
    "Anaphylaxis",
})

_evidences_ontology: dict[str, Any] | None = None


def _load_evidences_ontology(ddxplus_dir: Path = _DDXPLUS_DIR) -> dict[str, Any]:
    global _evidences_ontology
    if _evidences_ontology is None:
        path = ddxplus_dir / "release_evidences.json"
        with open(path, encoding="utf-8") as f:
            _evidences_ontology = json.load(f)
    return _evidences_ontology


def decode_evidence(ev_string: str, ontology: dict[str, Any]) -> tuple[str, Any]:
    """
    Decode a single evidence string into (human_readable_name, value).

    Formats:
      Binary:       "E_91"          → value = 1 (present)
      Categorical:  "E_10_@_2"      → value = "2"
      Multi-choice: "E_54_@_V_112"  → value = decoded english string or raw code
    """
    parts = ev_string.split("_@_")
    code = parts[0]
    raw_value = parts[1] if len(parts) > 1 else None

    entry = ontology.get(code, {})
    name: str = entry.get("question_en", code)

    if raw_value is None:
        return name, 1

    # Multi-choice: raw_value starts with "V_"
    if raw_value.startswith("V_"):
        value_meaning: dict = entry.get("value_meaning", {})
        vm_entry = value_meaning.get(raw_value, {})
        decoded = vm_entry.get("en", raw_value) if isinstance(vm_entry, dict) else raw_value
        return name, decoded

    # Categorical: numeric string
    return name, raw_value


def _parse_patient_row(
    row: pd.Series,
    row_index: int,
    split: str,
    ontology: dict[str, Any],
) -> DDXPlusPatient:
    evidences_raw: list[str] = ast.literal_eval(row["EVIDENCES"])
    differential_raw: list[list] = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])

    symptoms: dict[str, Any] = {}
    antecedents: dict[str, Any] = {}

    for ev_str in evidences_raw:
        code = ev_str.split("_@_")[0]
        entry = ontology.get(code, {})
        name, value = decode_evidence(ev_str, ontology)
        if entry.get("is_antecedent", False):
            antecedents[name] = value
        else:
            symptoms[name] = value

    initial_code: str = row["INITIAL_EVIDENCE"]
    initial_name, _ = decode_evidence(initial_code, ontology)

    ground_truth_differential: list[tuple[str, float]] = [
        (str(pair[0]), float(pair[1])) for pair in differential_raw
    ]

    return DDXPlusPatient(
        patient_id=f"{split}_{row_index:06d}",
        age=int(row["AGE"]),
        sex=str(row["SEX"]),
        initial_evidence=initial_name,
        initial_evidence_code=initial_code,
        symptoms=symptoms,
        antecedents=antecedents,
        ground_truth_pathology=str(row["PATHOLOGY"]),
        ground_truth_differential=ground_truth_differential,
    )


def load_ddxplus(
    ddxplus_dir: Path | str = _DDXPLUS_DIR,
    split: str = "test",
    max_rows: int | None = None,
) -> list[DDXPlusPatient]:
    """
    Load DDXPlus patients from the CSV file for the given split.

    Args:
        ddxplus_dir: Path to the DDXPlus_Raw directory.
        split: One of "test", "train", "validate". Defaults to "test".
        max_rows: If set, load only this many rows (useful for tests).

    Returns:
        List of DDXPlusPatient objects.
    """
    ddxplus_dir = Path(ddxplus_dir)
    filename = _SPLIT_FILE.get(split)
    if filename is None:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {list(_SPLIT_FILE)}")

    path = ddxplus_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"DDXPlus file not found: {path}")

    ontology = _load_evidences_ontology(ddxplus_dir)

    df = pd.read_csv(path, nrows=max_rows)

    patients: list[DDXPlusPatient] = []
    for idx, row in df.iterrows():
        patient = _parse_patient_row(row, int(idx), split, ontology)  # type: ignore[arg-type]
        patients.append(patient)

    return patients


def is_in_exemplar_pool(pathology: str) -> bool:
    """Return True if the pathology string matches a disease in the exemplar pool."""
    normalized = pathology.strip().lower()
    return any(normalized == d.lower() for d in EXEMPLAR_POOL_DISEASES)
