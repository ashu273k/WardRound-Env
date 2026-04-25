"""Scripted multi-agent responses for ward-round interactions.

Author: Abhijeet (Multi-Agent Simulation & Tasks)

Provides rule-based, deterministic behavior for three scripted hospital
agents: Senior Consultant, Nurse, and Patient/Family.  Each agent
responds based on the Junior Doctor's action, the current patient, and
the consultant's personality style.

Consultant Styles
─────────────────
  supportive   — coaching, encouraging, gives second chances
  conservative — cautious, demands evidence before action
  aggressive   — confrontational, challenges competence, time-pressures
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from models import Action, Patient


# ═══════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AgentReplies:
    """Aggregated responses from the full ward team."""
    consultant_opinion: str
    nurse_status: str
    family_concern: str | None = None
    team_alignment: float = 0.5  # 0=conflict, 1=harmony


# ═══════════════════════════════════════════════════════════════════════════
#  Consultant Agent — style-driven personality
# ═══════════════════════════════════════════════════════════════════════════

class ConsultantAgent:
    """Senior Consultant with personality-based response logic."""

    STYLE_RESPONSES = {
        "present_case": {
            "supportive": [
                "Concise and clear. Continue with the plan.",
                "Good summary. Make sure to mention the allergy status.",
            ],
            "conservative": [
                "Include risk stratification before we proceed.",
                "I need more data. What about trending vitals?",
            ],
            "aggressive": [
                "That was incomplete. Why didn't you include labs?",
                "I expect a tighter presentation. Restructure and try again.",
            ],
        },
        "decide_treatment": {
            "supportive": [
                "Plan accepted, execute now.",
                "Reasonable approach. Proceed and update me in 2 hours.",
            ],
            "conservative": [
                "Plan is cautious but safe. I approve with monitoring.",
                "Acceptable, but add a safety net order just in case.",
            ],
            "aggressive": [
                "Defend your plan with evidence before I approve.",
                "Why this over the alternative? Justify your reasoning.",
            ],
        },
        "escalate": {
            "supportive": [
                "Escalation is appropriate. Keep me updated.",
            ],
            "conservative": [
                "Escalation noted. Document your reasoning clearly.",
            ],
            "aggressive": [
                "You should have handled this yourself. But fine, escalate.",
            ],
        },
        "ask_consultant": {
            "supportive": [
                "Good question. Here's my take: stay the course but reassess in 4 hours.",
            ],
            "conservative": [
                "My recommendation: err on the side of caution. Order the test first.",
            ],
            "aggressive": [
                "You should know this. But: the evidence favors intervention.",
            ],
        },
    }

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def respond(self, action: Action, style: str) -> str:
        action_key = action.action_type
        style_map = self.STYLE_RESPONSES.get(action_key, {})
        options = style_map.get(style)
        if options:
            return options[self._rng.randrange(len(options))]
        # Fallback varies by style
        fallbacks = {
            "supportive": "Proceed to the next focused step.",
            "conservative": "Pause. Clarify your rationale before continuing.",
            "aggressive": "I'm not convinced. Move faster and be precise.",
        }
        return fallbacks.get(style, "Proceed.")


# ═══════════════════════════════════════════════════════════════════════════
#  Nurse Agent — practical, task-oriented
# ═══════════════════════════════════════════════════════════════════════════

class NurseAgent:
    """Practical bedside nurse focused on actionable orders."""

    def respond(self, action: Action) -> str:
        if action.action_type in {"request_test", "decide_treatment"}:
            return "Orders acknowledged and queued for execution."
        if action.action_type == "ask_nurse":
            return "Bedside status: vitals stable, patient comfortable, awaiting orders."
        if action.action_type == "reassure_patient":
            return "Nursing team standing by. Family has been notified of the update."
        if action.action_type == "escalate":
            return "Preparing for escalation protocol. Equipment on standby."
        return "Nursing team ready and awaiting actionable orders."


# ═══════════════════════════════════════════════════════════════════════════
#  Patient / Family Agent — emotional, concern-driven
# ═══════════════════════════════════════════════════════════════════════════

class PatientFamilyAgent:
    """Emotion-driven patient/family agent with condition-based triggers."""

    CONDITION_CONCERNS = {
        "delirium": "Family requests explanation about confusion and medication risks.",
        "acute": "Family is anxious about the urgency. Please explain the timeline.",
        "pain": "Patient is distressed. Family asks about pain management options.",
        "septic": "Family is frightened. They want to understand the prognosis.",
        "stroke": "Family is panicking. They need clear information about recovery chances.",
        "ethical": "Family insists on being part of the decision-making process.",
    }

    def respond(self, action: Action, patient: Patient) -> str | None:
        if action.action_type == "reassure_patient":
            return f"Family appreciates the update for {patient.name}. They feel more at ease."

        # Trigger concern based on patient condition keywords
        condition_lower = patient.condition.lower()
        for keyword, concern in self.CONDITION_CONCERNS.items():
            if keyword in condition_lower and action.action_type != "reassure_patient":
                return concern

        # No concern raised for stable / routine patients
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  ScriptedWardTeam — coordinator used by environment.py
# ═══════════════════════════════════════════════════════════════════════════

class ScriptedWardTeam:
    """Rule-based team behavior used for deterministic baseline simulation.

    Coordinates three scripted agents (Consultant, Nurse, Patient/Family)
    and returns aggregated responses for each doctor action.
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._consultant = ConsultantAgent(rng)
        self._nurse = NurseAgent()
        self._family = PatientFamilyAgent()

    def respond(
        self,
        *,
        action: Action,
        patient: Patient,
        consultant_style: str,
    ) -> AgentReplies:
        consultant_opinion = self._consultant.respond(action, consultant_style)
        nurse_status = self._nurse.respond(action)
        family_concern = self._family.respond(action, patient)

        # Compute team alignment based on consultant reaction
        alignment = 0.5
        if consultant_style == "supportive":
            alignment = 0.8
        elif consultant_style == "aggressive":
            alignment = 0.3
        if action.action_type == "decide_treatment":
            alignment += 0.1
        if family_concern:
            alignment -= 0.1
        alignment = max(0.0, min(1.0, alignment))

        return AgentReplies(
            consultant_opinion=consultant_opinion,
            nurse_status=nurse_status,
            family_concern=family_concern,
            team_alignment=alignment,
        )
