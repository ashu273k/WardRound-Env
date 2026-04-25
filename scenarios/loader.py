"""Deterministic scenario loader for root-level scenarios/."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

DIFFICULTY_FILES = {
    "easy": "easy.json",
    "medium": "medium.json",
    "hard": "hard.json",
}

SCENARIOS_DIR = Path(__file__).resolve().parent


def load_scenario(difficulty: str) -> Dict[str, Any]:
    """Load a scenario by difficulty level."""
    if difficulty not in DIFFICULTY_FILES:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    path = SCENARIOS_DIR / DIFFICULTY_FILES[difficulty]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_difficulties() -> List[str]:
    """Return available difficulty levels."""
    return list(DIFFICULTY_FILES.keys())


def sample_patient(difficulty: str, seed: int = 42) -> Dict[str, Any]:
    """Return one deterministic patient for a difficulty and seed."""
    data = load_scenario(difficulty)
    patients = data.get("patients", [])
    if not patients:
        raise ValueError(f"No patients in scenario: {difficulty}")
    rng = random.Random(seed)
    return patients[rng.randrange(len(patients))]
