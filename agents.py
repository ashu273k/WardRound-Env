"""Scripted multi-agent responses for ward-round interactions."""

from __future__ import annotations

import random
from dataclasses import dataclass

from models import Action, Patient


@dataclass(frozen=True)
class AgentReplies:
    consultant_opinion: str
    nurse_status: str
    family_concern: str | None = None


class ScriptedWardTeam:
    """Rule-based team behavior used for deterministic baseline simulation."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def respond(
        self,
        *,
        action: Action,
        patient: Patient,
        consultant_style: str,
    ) -> AgentReplies:
        consultant = self._consultant_reply(action, consultant_style)
        nurse = self._nurse_reply(action)
        family = self._family_reply(action, patient)
        return AgentReplies(
            consultant_opinion=consultant,
            nurse_status=nurse,
            family_concern=family,
        )

    def _consultant_reply(self, action: Action, style: str) -> str:
        if action.action_type == "present_case":
            return "Concise and clear. Continue." if style == "supportive" else "Include risk stratification."
        if action.action_type == "decide_treatment":
            return "Plan accepted, execute now." if style != "aggressive" else "Defend your plan with evidence."
        if action.action_type == "escalate":
            return "Escalation is appropriate. Keep me updated."
        return "Proceed to the next focused step."

    def _nurse_reply(self, action: Action) -> str:
        if action.action_type in {"request_test", "decide_treatment"}:
            return "Orders acknowledged and queued."
        if action.action_type == "ask_nurse":
            return "Bedside status stable; awaiting actionable order."
        return "Nursing team ready."

    def _family_reply(self, action: Action, patient: Patient) -> str | None:
        if action.action_type == "reassure_patient":
            return f"Family appreciates the update for {patient.name}."
        if "delirium" in patient.condition.lower() and action.action_type != "reassure_patient":
            return "Family requests explanation about confusion and medication risks."
        return None
