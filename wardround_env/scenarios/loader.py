"""Deterministic scenario loader for easy/medium/hard cases."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List


DIFFICULTY_FILES = {
    "easy": "easy.json",
    "medium": "medium.json",
    "hard": "hard.json",
}

SCENARIOS_DIR = Path(__file__).resolve().parent


def _read_cases(difficulty: str) -> List[Dict[str, Any]]:
    if difficulty not in DIFFICULTY_FILES:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    path = SCENARIOS_DIR / DIFFICULTY_FILES[difficulty]
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Scenario file must contain a list: {path}")
    return data


def list_cases(difficulty: str) -> List[Dict[str, Any]]:
    """Return all cases for a difficulty."""
    return _read_cases(difficulty)


def sample_case(difficulty: str, seed: int) -> Dict[str, Any]:
    """Return one deterministic case for difficulty and seed."""
    cases = _read_cases(difficulty)
    if not cases:
        raise ValueError(f"No cases available for difficulty={difficulty}")
    rng = random.Random(seed)
    idx = rng.randrange(len(cases))
    return cases[idx]


def get_case_by_id(difficulty: str, case_id: str) -> Dict[str, Any]:
    """Return one case by id for deterministic test selection."""
    cases = _read_cases(difficulty)
    for case in cases:
        if case.get("id") == case_id:
            return case
    raise ValueError(f"Case id not found for {difficulty}: {case_id}")

