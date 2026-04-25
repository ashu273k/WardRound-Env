"""Core OpenEnv environment for WardRound-Env with Strict Causal RL."""

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
    """Deterministic, Strict Causal RL ward-round environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    DEFAULT_SEED: int = 42

    def __init__(self) -> None:
        super().__init__()
        self._rng = random.Random(self.DEFAULT_SEED)
        self._team = ScriptedWardTeam(self._rng)
        self.task_data = self._load_scenarios()
        self._state = WardRoundState(episode_id=str(uuid4()), step_count=0)

    def _load_scenarios(self) -> dict[str, dict[str, Any]]:
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

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        task_id = kwargs.get("task_id", "easy")
        if task_id not in self.task_data:
            raise ValueError(f"Unknown task_id '{task_id}'.")

        effective_seed = self.DEFAULT_SEED if seed is None else seed
        self._rng.seed(effective_seed)

        data = self.task_data[task_id]
        
        # Base state initialization
        self._state = WardRoundState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            current_patient_index=0,
            patients=[patient.model_copy(deep=True) for patient in data["patients"]],
            time_remaining=data["time_limit"],
            decisions={},
            vitals_known={},
            labs_known={},
            consultant_approved={},
            vital_stability={},
            is_dead={},
            treated={},
            started=True,
        )
        
        for p in self._state.patients:
            self._state.vital_stability[p.id] = 1.0
            self._state.is_dead[p.id] = False
            self._state.treated[p.id] = False

        self._reset_rubric()

        return self._build_observation(
            feedback="Ward round started. Proceed with causality.",
            consultant_opinion="I expect thorough lab analysis before requesting approval.",
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
            raise RuntimeError("Environment not started.")

        self._state.step_count += 1
        self._state.time_remaining = max(0, self._state.time_remaining - 1)

        # Base step penalty to extinguish chatter-farming
        reward = -0.01
        
        # Get target patient
        p_id = action.patient_id
        target_patient = next((p for p in self._state.patients if p.id == p_id), self._state.patients[self._state.current_patient_index])
        actual_pid = target_patient.id

        # Multi-Agent State Tracking variables
        deaths_in_ward = sum(1 for v in self._state.is_dead.values() if v)
        trust_penalty = deaths_in_ward > 0
        
        # --- ACTION PROCESSOR ---
        consultant_feedback = "I am observing your decisions."
        nurse_status = "Awaiting orders."
        family_concern = "Watching anxiously."
        feedback = f"Action '{action.action_type}' applied."

        if self._state.is_dead[actual_pid]:
            # IRREVERSIBILITY
            reward -= 0.01
            feedback = "Patient is deceased. Action had no effect."
            nurse_status = "Code Blue complete. Time of death recorded."
            consultant_feedback = "This is a profound failure."
            family_concern = "Family is devastated and hostile."
        elif self._state.treated[actual_pid]:
            reward -= 0.01
            feedback = "Patient is already treated."
        else:
            if action.action_type == "ask_nurse":
                self._state.vitals_known[actual_pid] = True
                reward += 0.1
                nurse_status = "Vitals are stable but trending down."
                feedback = "Vitals recorded."

            elif action.action_type == "request_test":
                if not self._state.vitals_known.get(actual_pid, False):
                    reward -= 0.5
                    self._state.vital_stability[actual_pid] -= 0.4
                    nurse_status = "Error: You must check vitals before ordering invasive labs!"
                    feedback = "Unsafe sequence. Vitals missing."
                else:
                    self._state.labs_known[actual_pid] = True
                    reward += 0.2
                    nurse_status = "Labs drawn and resulted."

            elif action.action_type == "ask_consultant":
                # Multi-Agent Gatekeeper
                has_vitals = self._state.vitals_known.get(actual_pid, False)
                has_labs = self._state.labs_known.get(actual_pid, False)
                if trust_penalty and not (has_vitals and has_labs):
                     consultant_feedback = "Another patient died under your care. I will NOT approve treatment until you have absolutely all data. Get vitals and labs."
                     reward -= 0.5
                elif has_vitals and has_labs:
                     self._state.consultant_approved[actual_pid] = True
                     consultant_feedback = "I agree with your workup. You are approved to decide treatment."
                     reward += 0.2
                else:
                     consultant_feedback = "I cannot approve treatment blindly. You are missing core data."
                     reward -= 0.1

            elif action.action_type == "decide_treatment":
                # MULTI-AGENT CONSEQUENCE (True Dependency)
                if not self._state.consultant_approved.get(actual_pid, False):
                    self._state.is_dead[actual_pid] = True
                    self._state.vital_stability[actual_pid] = 0.0
                    reward -= 1.0
                    feedback = "FATAL ERROR: Administered unapproved treatment."
                    consultant_feedback = "You acted without my authorization! The patient crashed!"
                else:
                    # Valid treatment
                    self._state.treated[actual_pid] = True
                    self._state.vital_stability[actual_pid] = 1.0
                    reward += 0.5
                    consultant_feedback = "Treatment successful."

            # Other useless actions don't get points
            elif action.action_type in {"present_case", "answer_question", "reassure_patient", "escalate"}:
                reward -= 0.01  # Burn time step penalty
                
            self._state.decisions[actual_pid] = action.action_type

        # CASCADING DETERIORATION
        for p in self._state.patients:
            if not self._state.treated[p.id] and not self._state.is_dead[p.id]:
                self._state.vital_stability[p.id] -= 0.1
                if self._state.vital_stability[p.id] <= 0.0:
                    self._state.is_dead[p.id] = True
                    reward -= 1.0  # Death Penalty but NO done=True

        # TERMINATION CHECK
        all_resolved = all(self._state.treated.get(p.id) or self._state.is_dead.get(p.id) for p in self._state.patients)
        done = (self._state.time_remaining <= 0) or all_resolved

        grader_result = None
        if done:
            grader_result = self._compute_grader_score()

        observation = self._build_observation(
            feedback=feedback,
            consultant_opinion=consultant_feedback,
            nurse_status=nurse_status,
            family_concern=family_concern,
            done=done,
            reward=reward,
        )

        if grader_result:
            observation.metadata["grader_score"] = grader_result["final_score"]
            observation.metadata["grader_rubric"] = grader_result["rubric"]

        observation.reward = self._apply_rubric(action, observation) or reward
        return self._apply_transform(observation)

    def _compute_grader_score(self) -> Dict[str, Any]:
        total_p = len(self._state.patients)
        if total_p == 0:
            return {"final_score": 0.0, "rubric": {}}

        alive = sum(1 for p in self._state.patients if not self._state.is_dead.get(p.id))
        survival_frac = alive / total_p
        
        correct = sum(1 for p in self._state.patients if self._state.treated.get(p.id) and not self._state.is_dead.get(p.id))
        correctness = correct / total_p
        
        time_limit = self.task_data[self._state.task_id]["time_limit"]
        time_eff = max(0.0, self._state.time_remaining / time_limit)
        
        score = (survival_frac * 0.6) + (correctness * 0.2) + (time_eff * 0.2)
        
        if alive < total_p:
            score *= 0.8
            
        final = round(max(0.0, min(1.0, score)), 4)
        
        return {
            "final_score": final,
            "rubric": {
                "survival_fraction": survival_frac,
                "treatment_correctness": correctness,
                "time_efficiency": round(time_eff, 4)
            }
        }

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
        
        # Advance index to first unsolved patient
        idx = self._state.current_patient_index
        while idx < len(self._state.patients):
            pid = self._state.patients[idx].id
            if not (self._state.treated.get(pid) or self._state.is_dead.get(pid)):
                break
            idx += 1
            
        idx = min(idx, len(self._state.patients) - 1)
        self._state.current_patient_index = idx
        current_pid = self._state.patients[idx].id
        
        if self._state.is_dead.get(current_pid):
            feedback = f"Patient {current_pid} is DEAD."
        elif self._state.vital_stability.get(current_pid, 1.0) < 0.3:
            feedback = f"Patient {current_pid} is CRASHING."

        pending = self._state.patients[idx + 1 :] if not done else []
        return Observation(
            current_patient=self._state.patients[idx],
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
            },
        )

    def _reset_rubric(self) -> None:
        self._rubric_total = 0.0

    def _apply_rubric(self, _action: Action, observation: Observation) -> float:
        reward_val = observation.reward if observation.reward is not None else 0.0
        self._rubric_total += reward_val
        return reward_val

    def _apply_transform(self, observation: Observation) -> Observation:
        return observation

    @property
    def state(self) -> State:
        return self._state