from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")


def _load() -> dict[str, Any]:
    path = _ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


config: dict[str, Any] = _load()
