"""Rule-based multi-agent behaviors for WardRound-Env.

This module is intentionally deterministic and seed-controlled so that
rollouts are reproducible for training and grading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Optional


CONSULTANT_STYLES = ("supportive", "conservative", "aggressive")
DIFFICULTIES = ("easy", "medium", "hard")


def _stable_text_seed(text: str) -> int:
    """Return a deterministic integer for a given string."""
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


@dataclass
class AgentMessage:
    """Structured response from one scripted agent."""

    role: str
    tone: str
    text: str
    intent: str
    tags: List[str] = field(default_factory=list)


@dataclass
class MultiAgentTurnResult:
    """Aggregated output after all scripted agents react."""

    consultant: AgentMessage
    nurse: AgentMessage
    patient_family: AgentMessage
    state_updates: Dict[str, Any]


class SeniorConsultantAgent:
    """Scripted consultant with style-based behavior."""

    def __init__(self, style: str = "supportive", seed: int = 0) -> None:
        if style not in CONSULTANT_STYLES:
            raise ValueError(f"Unknown consultant style: {style}")
        self.style = style
        self.rng = random.Random(seed + _stable_text_seed(f"consultant:{style}"))

    def respond(
        self,
        *,
        doctor_action: Dict[str, Any],
        case_context: Dict[str, Any],
        round_state: Dict[str, Any],
    ) -> AgentMessage:
        presented_items = set(doctor_action.get("presented_facts", []))
        missing_critical = [
            item for item in case_context.get("critical_facts", []) if item not in presented_items
        ]
        unsafe_orders = [
            order for order in doctor_action.get("orders", []) if order in case_context.get("contraindicated_orders", [])
        ]

        if unsafe_orders:
            text = (
                f"I am concerned. The order '{unsafe_orders[0]}' is unsafe here. "
                "Reassess risk and present a safer plan now."
            )
            return AgentMessage(
                role="consultant",
                tone="firm",
                text=text,
                intent="safety_challenge",
                tags=["safety", "contraindication"],
            )

        if missing_critical:
            if self.style == "supportive":
                text = (
                    f"Good start. Please also include {missing_critical[0]} so the team has a complete picture."
                )
                tone = "coaching"
            elif self.style == "conservative":
                text = (
                    f"Before proceeding, clarify {missing_critical[0]}. "
                    "I do not want assumptions in this patient."
                )
                tone = "cautious"
            else:
                text = (
                    f"You skipped a key point: {missing_critical[0]}. "
                    "Why should I trust this plan without it?"
                )
                tone = "confrontational"
            return AgentMessage(
                role="consultant",
                tone=tone,
                text=text,
                intent="information_probe",
                tags=["completeness"],
            )

        disagreement = case_context.get("consultant_disagreement", False)
        if disagreement:
            preferred = case_context.get("consultant_preferred_plan", "a more conservative plan")
            if self.style == "aggressive":
                text = f"I disagree with your direction. I want {preferred} and a justification if you refuse."
            elif self.style == "conservative":
                text = f"I prefer {preferred}. Explain your risk mitigation before we continue."
            else:
                text = f"I see your logic, but I would lean toward {preferred}. Can you justify your choice?"
            return AgentMessage(
                role="consultant",
                tone="challenging",
                text=text,
                intent="plan_negotiation",
                tags=["conflict", "decision"],
            )

        affirmations = [
            "Presentation is clear. Continue.",
            "Reasonable plan. Keep the team aligned and proceed.",
            "Good. You addressed the key safety points.",
        ]
        text = affirmations[self.rng.randrange(len(affirmations))]
        return AgentMessage(
            role="consultant",
            tone="neutral",
            text=text,
            intent="acknowledge",
            tags=["alignment"],
        )


class NurseAgent:
    """Practical nurse that validates actionable execution."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed + _stable_text_seed("nurse"))

    def respond(
        self,
        *,
        doctor_action: Dict[str, Any],
        case_context: Dict[str, Any],
        round_state: Dict[str, Any],
    ) -> AgentMessage:
        orders = doctor_action.get("orders", [])
        required = case_context.get("required_nursing_actions", [])
        pending = [action for action in required if action not in orders]

        if pending:
            text = (
                f"I still need explicit orders for '{pending[0]}'. "
                "Please confirm urgency and timing so I can execute."
            )
            return AgentMessage(
                role="nurse",
                tone="practical",
                text=text,
                intent="request_clarification",
                tags=["tasking", "workflow"],
            )

        if round_state.get("time_remaining_min", 999) < 10:
            text = "We are tight on time. Please prioritize stat actions and defer non-urgent items."
            return AgentMessage(
                role="nurse",
                tone="urgent",
                text=text,
                intent="time_pressure",
                tags=["time"],
            )

        confirmations = [
            "Orders are clear. I will execute and report back.",
            "Understood. Tasks are actionable and assigned.",
            "Noted. Nursing tasks are feasible right now.",
        ]
        return AgentMessage(
            role="nurse",
            tone="task_oriented",
            text=confirmations[self.rng.randrange(len(confirmations))],
            intent="confirm_execution",
            tags=["coordination"],
        )


class PatientFamilyAgent:
    """Emotion-aware patient/family behavior with deterministic escalation."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed + _stable_text_seed("patient_family"))

    def respond(
        self,
        *,
        doctor_action: Dict[str, Any],
        case_context: Dict[str, Any],
        round_state: Dict[str, Any],
    ) -> AgentMessage:
        concerns = case_context.get("family_concerns", [])
        empathy_level = doctor_action.get("empathy_level", "low")
        explained = set(doctor_action.get("concerns_addressed", []))
        unresolved = [item for item in concerns if item not in explained]

        if unresolved:
            if empathy_level == "high":
                text = f"I appreciate your explanation, but I am still worried about {unresolved[0]}."
                tone = "worried"
            else:
                text = f"You are not hearing us. We are scared about {unresolved[0]} and need clear answers."
                tone = "distressed"
            return AgentMessage(
                role="patient_family",
                tone=tone,
                text=text,
                intent="seek_reassurance",
                tags=["emotion", "communication"],
            )

        ethical_trigger = case_context.get("ethical_conflict", False)
        if ethical_trigger and not doctor_action.get("shared_decision_making", False):
            text = (
                "This decision feels forced. We need to discuss options and consent before moving forward."
            )
            return AgentMessage(
                role="patient_family",
                tone="demanding",
                text=text,
                intent="ethical_pushback",
                tags=["ethics", "consent"],
            )

        calm_lines = [
            "Thank you for explaining. We feel more comfortable now.",
            "That makes sense to us. Please keep us updated.",
            "We understand the plan and agree to proceed.",
        ]
        return AgentMessage(
            role="patient_family",
            tone="calmer",
            text=calm_lines[self.rng.randrange(len(calm_lines))],
            intent="accept_plan",
            tags=["trust"],
        )


class MultiAgentCoordinator:
    """Coordinates deterministic responses from all scripted agents."""

    def __init__(self, *, seed: int = 0, consultant_style: str = "supportive") -> None:
        self.seed = seed
        self.consultant = SeniorConsultantAgent(style=consultant_style, seed=seed)
        self.nurse = NurseAgent(seed=seed)
        self.patient_family = PatientFamilyAgent(seed=seed)

    def run_turn(
        self,
        *,
        doctor_action: Dict[str, Any],
        case_context: Dict[str, Any],
        round_state: Dict[str, Any],
    ) -> MultiAgentTurnResult:
        consultant_msg = self.consultant.respond(
            doctor_action=doctor_action,
            case_context=case_context,
            round_state=round_state,
        )
        nurse_msg = self.nurse.respond(
            doctor_action=doctor_action,
            case_context=case_context,
            round_state=round_state,
        )
        family_msg = self.patient_family.respond(
            doctor_action=doctor_action,
            case_context=case_context,
            round_state=round_state,
        )

        state_updates = {
            "team_alignment_delta": self._alignment_delta(consultant_msg, nurse_msg, family_msg),
            "unresolved_family_concerns": family_msg.intent in {"seek_reassurance", "ethical_pushback"},
            "nursing_blocked": nurse_msg.intent == "request_clarification",
            "safety_alert": consultant_msg.intent == "safety_challenge",
        }
        return MultiAgentTurnResult(
            consultant=consultant_msg,
            nurse=nurse_msg,
            patient_family=family_msg,
            state_updates=state_updates,
        )

    @staticmethod
    def _alignment_delta(
        consultant_msg: AgentMessage,
        nurse_msg: AgentMessage,
        family_msg: AgentMessage,
    ) -> float:
        delta = 0.0
        if consultant_msg.intent in {"acknowledge"}:
            delta += 0.10
        if consultant_msg.intent in {"plan_negotiation", "information_probe"}:
            delta -= 0.05
        if consultant_msg.intent == "safety_challenge":
            delta -= 0.20

        if nurse_msg.intent == "confirm_execution":
            delta += 0.05
        if nurse_msg.intent in {"request_clarification", "time_pressure"}:
            delta -= 0.05

        if family_msg.intent == "accept_plan":
            delta += 0.10
        if family_msg.intent in {"seek_reassurance", "ethical_pushback"}:
            delta -= 0.10
        return max(-0.4, min(0.4, delta))


def consultant_style_for_difficulty(difficulty: str) -> str:
    """Default consultant style by difficulty tier."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    mapping = {"easy": "supportive", "medium": "conservative", "hard": "aggressive"}
    return mapping[difficulty]


def build_multi_agent_coordinator(
    *, difficulty: str, seed: int = 0, consultant_style: Optional[str] = None
) -> MultiAgentCoordinator:
    """Factory used by environment reset for deterministic setup."""
    style = consultant_style or consultant_style_for_difficulty(difficulty)
    return MultiAgentCoordinator(seed=seed, consultant_style=style)

