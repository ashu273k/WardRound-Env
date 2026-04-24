"""Typed OpenEnv client for WardRound-Env."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import Action, Observation


class WardRoundEnvClient(EnvClient[Action, Observation, State]):
    """Client wrapper for connecting to a running WardRound-Env server."""

    def _step_payload(self, action: Action) -> dict[str, Any]:
        payload = action.model_dump()
        payload.pop("metadata", None)
        return payload

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[Observation]:
        obs_payload = payload.get("observation", {})
        observation = Observation(**obs_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(**payload)
