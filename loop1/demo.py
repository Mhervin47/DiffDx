"""
Demo: run a single interactive diagnostic session.

Usage:
    cd /Users/mhervin/Desktop/MedicalConvo/loop1
    PYTHONPATH=src .venv/bin/python3 demo.py

The doctor will ask questions. Type the patient's answers.
Type 'quit' at any prompt to end early.
"""
import json
import sys
from pathlib import Path

# ── load env ──────────────────────────────────────────────────────────────────
try:
    import dotenv
    dotenv.load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loop1.schemas import PatientProfile
from loop1.session import Session

# ── load profile ──────────────────────────────────────────────────────────────
profile_path = Path(__file__).parent / "test_cases" / "case_01.json"
profile = PatientProfile.model_validate(json.loads(profile_path.read_text()))

# ── run ───────────────────────────────────────────────────────────────────────
session = Session(profile)
session.run()
