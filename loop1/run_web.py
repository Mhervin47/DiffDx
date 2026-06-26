#!/usr/bin/env python3
"""
Launch the MedicalConvo web UI.

Usage:
    cd /Users/mhervin/Desktop/MedicalConvo/loop1
    PYTHONPATH=src .venv/bin/python3 run_web.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "web.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["web", "src"],
        reload_excludes=["web/data/*", "web/data/**/*"],
        log_level="info",
    )
