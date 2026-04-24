"""Core OpenEnv environment for WardRound-Env."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..agents import ScriptedWardTeam
    from ..models import Action, Observation, Patient, WardRoundState
except ImportError:
    from agents import ScriptedWardTeam
    from models import Action, Observation, Patient, WardRoundState


class WardRoundEnvironment(Environment[Action, Observation, WardRoundState]):
    """Deterministic turn-based ward-round environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    DEFAULT_SEED: int = 42

    def __init__(self) -> None:
        super().__init__()
        self._rng = random.Random(self.DEFAULT_SEED)
        self._team = ScriptedWardTeam(self._rng)
        self.task_data = self._load_scenarios()
        self._state = WardRoundState(episode_id=str(uuid4()), step_count=0)

    def _load_scenarios(self) -> dict[str, dict[str, Any]]:
        """Load scenario files so team-authored data is used directly."""
        base_dir = Path(__file__).resolve().parent.parent
        scenarios_dir = base_dir / "scenarios"
        loaded: dict[str, dict[str, Any]] = {}
        for task_id in ("easy", "medium", "hard"):
            scenario_path = scenarios_dir / f"{task_id}.json"
            with scenario_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            loaded[task_id] = {
                "patients": [Patient(**p) for p in payload["patients"]],
                "consultant_style": payload["consultant_style"],
                "time_limit": int(payload["time_limit"]),
            }
        return loaded

    def _build_observation(
        self,
        *,
        feedback: str,
        consultant_opinion: str | None = None,
        nurse_status: str | None = "ready",
        family_concern: str | None = None,
        done: bool = False,
        reward: float | None = None,
    ) -> Observation:
        current_idx = min(
            self._state.current_patient_index,
            max(0, len(self._state.patients) - 1),
        )
        pending = self._state.patients[current_idx + 1 :] if not done else []
        return Observation(
            current_patient=self._state.patients[current_idx],
            pending_patients=pending,
            consultant_opinion=consultant_opinion,
            nurse_status=nurse_status,
            family_concern=family_concern,
            time_remaining=self._state.time_remaining,
            goal=f"Lead a successful ward round for task: {self._state.task_id}",
            last_feedback=feedback,
            done=done,
            reward=reward,
            metadata={
                "task_id": self._state.task_id,
                "step_count": self._state.step_count,
                "decisions": dict(self._state.decisions),
            },
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        task_id = kwargs.get("task_id", "easy")
        if task_id not in self.task_data:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Expected one of {list(self.task_data)}."
            )

        effective_seed = self.DEFAULT_SEED if seed is None else seed
        self._rng.seed(effective_seed)

        data = self.task_data[task_id]
        self._state = WardRoundState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            current_patient_index=0,
            patients=[patient.model_copy(deep=True) for patient in data["patients"]],
            time_remaining=data["time_limit"],
            decisions={},
            started=True,
        )
        self._reset_rubric()

        return self._build_observation(
            feedback="Ward round started. Present the first patient.",
            consultant_opinion="Please begin with a concise case presentation.",
            nurse_status="ready",
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        if not self._state.started:
            raise RuntimeError("Environment not started. Call reset() before step().")

        self._state.step_count += 1
        self._state.time_remaining = max(0, self._state.time_remaining - 1)

        current_patient = self._state.patients[self._state.current_patient_index]
        self._state.decisions[current_patient.id] = action.action_type

        reward = 0.1
        team_reply = self._team.respond(
            action=action,
            patient=current_patient,
            consultant_style=self.task_data[self._state.task_id]["consultant_style"],
        )
        consultant_feedback = team_reply.consultant_opinion
        nurse_status = team_reply.nurse_status

        if action.action_type == "decide_treatment":
            reward += 0.2
            if self._state.current_patient_index < len(self._state.patients) - 1:
                self._state.current_patient_index += 1
        elif action.action_type in {"present_case", "answer_question"}:
            reward += 0.05
        elif action.action_type in {"escalate", "reassure_patient"}:
            reward += 0.05

        done = (
            self._state.time_remaining == 0
            or self._state.current_patient_index >= len(self._state.patients) - 1
            and action.action_type == "decide_treatment"
        )

        feedback = (
            f"Action '{action.action_type}' applied to patient '{action.patient_id}'."
        )
        observation = self._build_observation(
            feedback=feedback,
            consultant_opinion=consultant_feedback,
            nurse_status=nurse_status,
            family_concern=team_reply.family_concern,
            done=done,
            reward=reward,
        )
        observation.reward = self._apply_rubric(action, observation) or reward
        return self._apply_transform(observation)

    @property
    def state(self) -> State:
        return self._state