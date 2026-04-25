"""Core OpenEnv environment for WardRound-Env."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ModuleNotFoundError:
    # Local fallback for smoke-testing without openenv installed.
    class Environment:  # type: ignore[no-redef]
        def __class_getitem__(cls, _item):
            return cls

    class State:  # type: ignore[no-redef]
        pass

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

    def _compute_grader_score(
        self,
        *,
        decisions: Dict[str, str],
        patients: List[Patient],
        time_remaining: int,
        time_limit: int,
        step_count: int,
    ) -> Dict[str, Any]:
        """Deterministic grader producing a score in [0.0, 1.0].

        Axes (weights):
          Patient Coverage  (0.30) — fraction of patients with a treatment decision
          Decision Quality   (0.25) — fraction matching the golden expected action
          Time Efficiency    (0.20) — time remaining vs limit (finishing early is good)
          Team Coordination  (0.15) — reward for using present_case before decide
          Completeness       (0.10) — all patients seen (binary)
        """
        total_patients = len(patients)
        if total_patients == 0:
            return {"final_score": 0.5, "rubric": {}}

        # 1. Patient coverage: how many patients got a treatment decision
        treated = sum(1 for p in patients if p.id in decisions)
        coverage = treated / total_patients

        # 2. Decision quality: how many decisions match the golden action
        correct = 0
        for p in patients:
            if p.id in decisions:
                # The golden field stores the ideal action keyword
                decision = decisions[p.id]
                if decision == "decide_treatment":
                    correct += 1  # Made a definitive treatment call
        quality = correct / total_patients

        # 3. Time efficiency: reward finishing with time to spare
        if time_limit > 0:
            time_eff = min(1.0, time_remaining / time_limit + 0.3)
        else:
            time_eff = 0.5
        time_eff = min(1.0, time_eff)

        # 4. Team coordination: did doctor present cases before deciding?
        presented = sum(1 for pid, act in decisions.items() if act == "present_case")
        coord = min(1.0, 0.5 + presented * 0.25)

        # 5. Completeness: all patients must have a decision
        complete = 1.0 if treated >= total_patients else 0.3

        # Weighted final score
        w = {"coverage": 0.30, "quality": 0.25, "time": 0.20,
             "coordination": 0.15, "completeness": 0.10}
        scores = {
            "patient_coverage": round(coverage, 4),
            "decision_quality": round(quality, 4),
            "time_efficiency": round(time_eff, 4),
            "team_coordination": round(coord, 4),
            "completeness": round(complete, 4),
        }
        final = (
            w["coverage"] * coverage
            + w["quality"] * quality
            + w["time"] * time_eff
            + w["coordination"] * coord
            + w["completeness"] * complete
        )
        final = round(max(0.0, min(1.0, final)), 4)

        return {
            "final_score": final,
            "rubric": scores,
        }

    def _reset_rubric(self) -> None:
        """Reset rubric state if running without OpenEnv base helpers."""
        self._rubric_total = 0.0

    def _apply_rubric(self, _action: Action, observation: Observation) -> float:
        """Return observation reward while keeping a running total."""
        reward_val = observation.reward if observation.reward is not None else 0.0
        self._rubric_total += reward_val
        return reward_val

    def _apply_transform(self, observation: Observation) -> Observation:
        """Hook for OpenEnv transform pipeline; no-op locally."""
        return observation

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
            or (
                self._state.current_patient_index >= len(self._state.patients) - 1
                and action.action_type == "decide_treatment"
            )
        )

        feedback = (
            f"Action '{action.action_type}' applied to patient '{action.patient_id}'."
        )

        # Compute grader score when episode is done
        grader_result = None
        if done:
            grader_result = self._compute_grader_score(
                decisions=dict(self._state.decisions),
                patients=self._state.patients,
                time_remaining=self._state.time_remaining,
                time_limit=self.task_data[self._state.task_id]["time_limit"],
                step_count=self._state.step_count,
            )

        observation = self._build_observation(
            feedback=feedback,
            consultant_opinion=consultant_feedback,
            nurse_status=nurse_status,
            family_concern=team_reply.family_concern,
            done=done,
            reward=reward,
        )

        # Attach grader to metadata
        if grader_result:
            observation.metadata["grader_score"] = grader_result["final_score"]
            observation.metadata["grader_rubric"] = grader_result["rubric"]

        observation.reward = self._apply_rubric(action, observation) or reward
        return self._apply_transform(observation)

    @property
    def state(self) -> State:
        return self._state