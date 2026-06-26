"""
Curate the 20-patient DDXPlus eval set for Phase 7.

Stratification:
  5 in-pool  — diseases covered by a Loop 1 exemplar (happy-path baseline)
  10 out-of-pool — diseases in DDXPlus but NOT in the exemplar pool (coverage stress test)
  5 ambiguous — diseases with tight top-2 differentials (discrimination test)

Run:
    python scripts/curate_eval_set.py

Output: data/ddxplus_eval_set.json (committed to git)
"""
from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from loop2.ddxplus.loader import (
    _load_evidences_ontology,
    _parse_patient_row,
    _DDXPLUS_DIR,
)

SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Selection targets
# ---------------------------------------------------------------------------

IN_POOL = [
    "Pulmonary embolism",
    "Pericarditis",
    "Possible NSTEMI / STEMI",
    "SLE",
    "Unstable angina",     # adjacent to stimulant ACS exemplar; triggers same retrieval path
]

OUT_OF_POOL = [
    "Pneumonia",
    "Influenza",
    "Panic attack",
    "Atrial fibrillation",
    "Cluster headache",
    "Spontaneous pneumothorax",
    "Tuberculosis",
    "Bronchospasm / acute asthma exacerbation",
    "Guillain-Barré syndrome",
    "Acute pulmonary edema",
]

AMBIGUOUS = [
    "GERD",            # overlaps with NSTEMI/angina on chest-pain profile
    "Anemia",          # overlaps with multiple on fatigue/weakness
    "Bronchitis",      # overlaps with pneumonia/URTI
    "PSVT",            # overlaps with panic attack/AFib on palpitations
    "Stable angina",   # 0% gt_top1 in DDXPlus — always behind unstable angina/NSTEMI
]

RATIONALES: dict[str, str] = {
    "Pulmonary embolism":
        "In-pool exemplar disease. Tests whether retrieval correctly surfaces PE exemplar "
        "and doctor applies DVT/travel/breathlessness reasoning pattern.",
    "Pericarditis":
        "In-pool exemplar disease. Pericarditis exemplar covers friction rub + positional pain; "
        "tests whether doctor screens for those features.",
    "Possible NSTEMI / STEMI":
        "In-pool exemplar disease (stimulant ACS exemplar). Tests ACS red-flag screening "
        "and appropriate confidence given high-stakes diagnosis.",
    "SLE":
        "In-pool exemplar disease. Multi-system presentation; tests whether doctor "
        "tracks malar rash + arthritis + renal features together.",
    "Unstable angina":
        "Adjacent to in-pool ACS exemplar. Tests whether stimulant_acs exemplar retrieves "
        "and applies correctly to a related but distinct ACS variant.",
    "Pneumonia":
        "Out-of-pool. Common but absent from exemplar bank; tests generalisation "
        "to a frequent respiratory condition.",
    "Influenza":
        "Out-of-pool. High-volume upper-respiratory; tests whether doctor differentiates "
        "from URTI/bronchitis without exemplar guidance.",
    "Panic attack":
        "Out-of-pool. Psychiatric mimicker of cardiac disease; tests differential breadth "
        "when no exemplar covers this presentation.",
    "Atrial fibrillation":
        "Out-of-pool. Cardiac arrhythmia; tests palpitation + stroke-risk differential "
        "without exemplar support.",
    "Cluster headache":
        "Out-of-pool. High gt_top1 (100%) but low ambiguity — should be easy even without "
        "exemplar; tests whether doctor reaches correct dx quickly.",
    "Spontaneous pneumothorax":
        "Out-of-pool. Acute chest pain + dyspnea overlap with PE/ACS; tests whether doctor "
        "screens percussion/breath-sounds features.",
    "Tuberculosis":
        "Out-of-pool. Only 17% gt_top1 in DDXPlus — the differential is hard even with "
        "ground truth. Tests handling of chronic cough + night sweats presentation.",
    "Bronchospasm / acute asthma exacerbation":
        "Out-of-pool. Wheezing + dyspnea overlap with PE; tests respiratory discrimination "
        "without an asthma exemplar.",
    "Guillain-Barré syndrome":
        "Out-of-pool. Rare neurological; tests whether doctor elicits ascending weakness "
        "pattern without relevant exemplar.",
    "Acute pulmonary edema":
        "Out-of-pool. Overlaps with PE and NSTEMI on dyspnea + chest presentation; "
        "tests cardiac vs. fluid-overload differentiation.",
    "GERD":
        "Ambiguous. Chest pain + epigastric discomfort closely mimics NSTEMI/angina; "
        "chosen with tight top-2 gap to stress differential ordering.",
    "Anemia":
        "Ambiguous. Fatigue + pallor + dyspnea overlap with multiple conditions; "
        "chosen with tight top-2 gap.",
    "Bronchitis":
        "Ambiguous. Cough + low-grade fever overlaps with pneumonia/URTI; "
        "chosen from patients where gt is NOT top-1 in DDXPlus differential.",
    "PSVT":
        "Ambiguous. Palpitations + chest discomfort overlaps with panic attack and AFib; "
        "chosen with tight top-2 gap.",
    "Stable angina":
        "Ambiguous. 0% gt_top1 in DDXPlus — model always ranks unstable angina or NSTEMI "
        "higher. Canonical discrimination-test case.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def top2_gap(diff_str: str) -> float:
    diff = ast.literal_eval(diff_str)
    return abs(diff[0][1] - diff[1][1]) if len(diff) >= 2 else 1.0


def gt_is_top1(row: pd.Series) -> bool:
    diff = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
    return diff[0][0] == row["PATHOLOGY"]


def pick_one(
    df: pd.DataFrame,
    disease: str,
    prefer_gt_top1: bool = True,
    prefer_tight: bool = False,
    rng: random.Random | None = None,
) -> pd.Series:
    _rng = rng or random
    sub = df[df["PATHOLOGY"] == disease].copy()
    if sub.empty:
        raise ValueError(f"No patients found for disease: {disease!r}")

    sub["top2_gap"] = sub["DIFFERENTIAL_DIAGNOSIS"].apply(top2_gap)
    sub["gt_top1"] = sub.apply(gt_is_top1, axis=1)

    if prefer_gt_top1:
        filtered = sub[sub["gt_top1"]]
        if not filtered.empty:
            sub = filtered

    if prefer_tight:
        tight = sub[sub["top2_gap"] < 0.10]
        if not tight.empty:
            sub = tight

    return sub.sample(1, random_state=_rng.randint(0, 10_000)).iloc[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(_DDXPLUS_DIR / "release_test_patients")
    ontology = _load_evidences_ontology()

    rng = random.Random(SEED)
    selected: list[dict] = []

    strata = [
        ("in_pool", IN_POOL, True, False),
        ("out_of_pool", OUT_OF_POOL, True, False),
        ("ambiguous", AMBIGUOUS, False, True),
    ]

    for stratum_name, diseases, prefer_gt_top1, prefer_tight in strata:
        for disease in diseases:
            row = pick_one(df, disease, prefer_gt_top1=prefer_gt_top1,
                           prefer_tight=prefer_tight, rng=rng)
            patient = _parse_patient_row(row, int(row.name), "test", ontology)  # type: ignore[arg-type]
            entry = {
                "patient_id": patient.patient_id,
                "row_index": int(row.name),
                "stratum": stratum_name,
                "disease": disease,
                "rationale": RATIONALES[disease],
                "age": patient.age,
                "sex": patient.sex,
                "ground_truth_pathology": patient.ground_truth_pathology,
                "initial_evidence": patient.initial_evidence,
                "initial_evidence_code": patient.initial_evidence_code,
                "gt_is_top1_in_ddxplus": bool(gt_is_top1(row)),
                "top2_gap": float(top2_gap(row["DIFFERENTIAL_DIAGNOSIS"])),
                "n_evidences": len(ast.literal_eval(row["EVIDENCES"])),
                "ground_truth_differential": patient.ground_truth_differential,
                "symptoms": patient.symptoms,
                "antecedents": patient.antecedents,
            }
            selected.append(entry)
            print(f"  [{stratum_name:12s}] {disease} → {patient.patient_id} "
                  f"(gt_top1={entry['gt_is_top1_in_ddxplus']}, gap={entry['top2_gap']:.3f})")

    out_path = Path("data/ddxplus_eval_set.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    print(f"\nSaved {len(selected)} patients to {out_path}")


if __name__ == "__main__":
    main()
