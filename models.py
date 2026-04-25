"""Typed models for the WardRound-Env OpenEnv interface."""

from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
    from openenv.core.env_server.types import State
except ModuleNotFoundError:
    # Local fallback so inference/tests can run without openenv installed.
    class OpenEnvAction(BaseModel):
        pass

    class OpenEnvObservation(BaseModel):
        done: bool = False
        reward: float | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        episode_id: str = ""
        step_count: int = 0


ActionType = Literal[
    "present_case",
    "answer_question",
    "ask_nurse",
    "ask_consultant",
    "decide_treatment",
    "reassure_patient",
    "request_test",
    "escalate",
]


class Patient(BaseModel):
    """Serializable patient payload used in observations and internal state."""

    id: str
    name: str
    age: int = Field(..., ge=0)
    condition: str
    golden: str = Field(..., description="Scenario-specific key expected action")


class Action(OpenEnvAction):
    """Action issued by the Junior Doctor policy."""

    action_type: ActionType
    patient_id: str
    content: str = ""
    target: str | None = Field(default=None, description="consultant/nurse/patient")
    reason: str | None = None


class Observation(OpenEnvObservation):
    """Structured observation visible to the policy."""

    current_patient: Patient
    pending_patients: list[Patient]
    consultant_opinion: str | None = None
    nurse_status: str | None = None
    family_concern: str | None = None
    time_remaining: int = Field(..., ge=0)
    goal: str
    last_feedback: str


class Reward(BaseModel):
    """Optional reward payload for future grader integration."""

    value: float
    info: dict[str, Any] = Field(default_factory=dict)


class WardRoundState(State):
    """Internal deterministic state for one ward-round episode."""

    task_id: str = "easy"
    current_patient_index: int = 0
    patients: list[Patient] = Field(default_factory=list)
    time_remaining: int = 0
    decisions: dict[str, str] = Field(default_factory=dict)
    
    # Causal tracking properties
    vitals_known: dict[str, bool] = Field(default_factory=dict)
    labs_known: dict[str, bool] = Field(default_factory=dict)
    consultant_approved: dict[str, bool] = Field(default_factory=dict)
    vital_stability: dict[str, float] = Field(default_factory=dict)
    is_dead: dict[str, bool] = Field(default_factory=dict)
    treated: dict[str, bool] = Field(default_factory=dict)
    
    # Hidden Multi-Agent States
    consultant_type: str = "conservative"  # conservative, aggressive, risk_averse
    family_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    trust_score: float = 0.5  # 0.0 (incompetent) to 1.0 (expert)
    
    started: bool = False