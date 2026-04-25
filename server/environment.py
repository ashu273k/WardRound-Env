"""Core OpenEnv environment for WardRound-Env with Strict Causal RL and Multi-Agent Reasoning."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
    """Advanced Multi-Agent Reasoning environment with hidden traits and conflicting incentives."""

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
            if not scenario_path.exists():
                continue
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
            task_id = "easy" # Fallback

        effective_seed = self.DEFAULT_SEED if seed is None else seed
        self._rng.seed(effective_seed)

        data = self.task_data[task_id]
        
        # Determine hidden traits
        c_type = self._rng.choice(["conservative", "aggressive", "risk_averse"])
        
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
            consultant_type=c_type,
            family_states={},
            started=True,
        )
        
        for p in self._state.patients:
            self._state.vital_stability[p.id] = self._rng.uniform(0.65, 0.95) # Start closer to danger
            self._state.is_dead[p.id] = False
            self._state.treated[p.id] = False
            self._state.family_states[p.id] = {
                "resistance": self._rng.uniform(0.3, 0.9), # Higher starting resistance
                "emotional_state": self._rng.choice(["angry", "scared", "cooperative"])
            }

        self._reset_rubric()

        return self._build_observation(
            feedback=f"Ward round started. Mission: Lead the team effectively.",
            consultant_opinion="I'm ready for the round. Present the first case.",
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

        # Base step penalty
        reward = -0.01
        
        p_id = action.patient_id
        # Fallback to current patient if id is invalid or generic
        if p_id not in [p.id for p in self._state.patients]:
             p_id = self._state.patients[self._state.current_patient_index].id
        
        actual_pid = p_id

        # --- CAUSAL AGENT INTERACTION ---
        state_vars = {
            "vitals_known": self._state.vitals_known.get(actual_pid, False),
            "labs_known": self._state.labs_known.get(actual_pid, False),
            "vital_stability": self._state.vital_stability[actual_pid],
            "trust_score": self._state.trust_score
        }
        
        team_reply = self._team.respond(
            action=action,
            patient=next(p for p in self._state.patients if p.id == actual_pid),
            state_vars=state_vars,
            consultant_personality=self._state.consultant_type,
            family_state=self._state.family_states[actual_pid],
            decision_history=dict(self._state.decisions)
        )
        
        self._state.trust_score = max(0.0, min(1.0, self._state.trust_score + team_reply.trust_delta))

        consultant_feedback = team_reply.consultant_opinion
        nurse_status = team_reply.nurse_status
        family_concern = team_reply.family_concern
        feedback = f"Action '{action.action_type}' applied."

        # Apply resistance penalty (time cost) - DOUBLE IMPACT
        if team_reply.resistance_penalty > 0:
            penalty_steps = int(team_reply.resistance_penalty * 10)
            self._state.time_remaining = max(0, self._state.time_remaining - penalty_steps)
            reward -= 0.15 # Higher penalty for social friction

        if self._state.is_dead[actual_pid]:
            reward -= 0.05
            feedback = "Patient is deceased."
        elif self._state.treated[actual_pid]:
            reward -= 0.01
            feedback = "Already treated."
        else:
            if action.action_type == "ask_nurse":
                self._state.vitals_known[actual_pid] = True
                reward += 0.05
            elif action.action_type == "request_test":
                if not self._state.vitals_known.get(actual_pid, False):
                    reward -= 0.2
                    self._state.vital_stability[actual_pid] -= 0.2
                else:
                    self._state.labs_known[actual_pid] = True
                    reward += 0.1
            elif action.action_type == "ask_consultant":
                if team_reply.approval_granted:
                    self._state.consultant_approved[actual_pid] = True
                    reward += 0.2
                else:
                    # Penalize checking without data
                    reward -= 0.1
            elif action.action_type == "decide_treatment":
                # Check for approval OR check for emergency override
                can_treat = self._state.consultant_approved.get(actual_pid, False)
                stability = self._state.vital_stability[actual_pid]
                
                # PATH A: Approved path (Safe)
                if can_treat:
                    self._state.treated[actual_pid] = True
                    self._state.vital_stability[actual_pid] = 1.0
                    reward += 1.0
                # PATH B: Emergency Override (Risky)
                elif stability < 0.3 and action.action_type == "decide_treatment":
                    # Override is allowed but risky. 50% chance of high success, 50% chance of death.
                    if self._rng.random() < 0.5:
                        self._state.treated[actual_pid] = True
                        self._state.vital_stability[actual_pid] = 1.0
                        self._state.trust_score = min(1.0, self._state.trust_score + 0.3) # Massive trust gain on heroic save
                        reward += 0.5
                        consultant_feedback = "I didn't approve, but you saved them. Risky move, Doctor."
                    else:
                        self._state.is_dead[actual_pid] = True
                        self._state.vital_stability[actual_pid] = 0.0
                        self._state.trust_score = max(0.0, self._state.trust_score - 0.4)
                        reward -= 2.0
                        feedback = "OVERRIDE FAILED. Patient expired."
                else:
                    # Unauthorized treatment on stable patient = death and high penalty
                    self._state.is_dead[actual_pid] = True
                    self._state.vital_stability[actual_pid] = 0.0
                    reward -= 2.0
                    feedback = "UNAUTHORIZED clinical decision. Patient expired."
            
            elif action.action_type == "reassure_patient":
                # Only reward the first reassurance per patient
                if self._state.family_states[actual_pid].get("reassured"):
                    reward -= 0.05
                else:
                    self._state.family_states[actual_pid]["reassured"] = True
                    reward += 0.1

        # CASCADING DETERIORATION - BRUTAL MODE
        for p in self._state.patients:
            if not self._state.treated[p.id] and not self._state.is_dead[p.id]:
                # Dynamic decay: faster if unstable or if conflict exists
                decay = 0.10 
                if self._state.vital_stability[p.id] < 0.6: decay = 0.25 
                self._state.vital_stability[p.id] -= decay
                
                # Check for stability-based death
                if self._state.vital_stability[p.id] <= 0:
                    self._state.is_dead[p.id] = True
                    self._state.trust_score = max(0.0, self._state.trust_score - 0.5) # Trust collapse on death
                    reward -= 2.0 

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
        if total_p == 0: return {"final_score": 0.0, "rubric": {}}

        alive = sum(1 for p in self._state.patients if not self._state.is_dead.get(p.id))
        survival_frac = alive / total_p
        
        # Bonus for team alignment and efficiency
        time_limit = self.task_data[self._state.task_id]["time_limit"]
        time_eff = max(0.0, self._state.time_remaining / time_limit)
        
        score = (survival_frac * 0.7) + (time_eff * 0.3)
        if alive < total_p: score *= 0.5 # Strict penalty for any mortality
            
        return {
            "final_score": round(max(0.0, min(1.0, score)), 4),
            "rubric": {
                "survival": survival_frac,
                "time_efficiency": time_eff,
                "consultant_personality": self._state.consultant_type
            }
        }

    def _build_observation(self, **kwargs) -> Observation:
        idx = self._state.current_patient_index
        while idx < len(self._state.patients):
            pid = self._state.patients[idx].id
            if not (self._state.treated.get(pid) or self._state.is_dead.get(pid)):
                break
            idx += 1
        idx = min(idx, len(self._state.patients) - 1)
        self._state.current_patient_index = idx

        observation = Observation(
            current_patient=self._state.patients[idx],
            pending_patients=self._state.patients[idx+1:] if not kwargs.get("done") else [],
            consultant_opinion=kwargs.get("consultant_opinion"),
            nurse_status=kwargs.get("nurse_status"),
            family_concern=kwargs.get("family_concern"),
            time_remaining=self._state.time_remaining,
            goal=f"Task: {self._state.task_id}. Manage the team and save the patients.",
            last_feedback=kwargs.get("feedback"),
            done=kwargs.get("done", False),
            reward=kwargs.get("reward"),
            metadata={"step": self._state.step_count}
        )
        return observation

    def _reset_rubric(self) -> None:
        self._rubric_total = 0.0

    def _apply_rubric(self, _action: Action, observation: Observation) -> float:
        val = observation.reward or 0.0
        self._rubric_total += val
        return val

    def _apply_transform(self, observation: Observation) -> Observation:
        return observation

    @property
    def state(self) -> State:
        return self._state