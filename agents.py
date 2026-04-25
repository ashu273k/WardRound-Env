"""Scripted multi-agent responses with conflicting incentives and hidden traits.

Author: Abhijeet (Multi-Agent Interaction Logic)

This module provides complex, rule-based behavior for three scripted agents:
Senior Consultant, Nurse, and Patient/Family. Agents have hidden traits that
influence approval and compliance, forcing the Junior Doctor to use Theory of Mind.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models import Action, Patient


@dataclass(frozen=True)
class AgentReplies:
    """Aggregated responses from the full ward team with mechanical effects."""
    consultant_opinion: str
    nurse_status: str
    family_concern: Optional[str] = None
    approval_granted: bool = False
    team_alignment: float = 0.5  # 0=conflict, 1=harmony
    resistance_penalty: float = 0.0
    trust_delta: float = 0.0


class ConsultantAgent:
    """Senior Consultant with hidden personality-based approval logic."""

    # Personality characteristics:
    # conservative: Needs vitals + labs. Very safe.
    # aggressive:   Prioritizes speed. Might approve without labs if present_case is good.
    # risk_averse:  Needs vitals + labs + stable patient. Rejects if patient is crashing.

    STYLE_TRAITS = {
        "conservative": "I demand a full data set before we commit to intervention.",
        "aggressive": "Time is tissue. If you're sure, act now. But don't miss.",
        "risk_averse": "The patient is too unstable for aggressive moves without deep certainty.",
    }

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def get_approval(self, action: Action, traits: Dict[str, Any], personality: str, decision_history: Dict[str, str]) -> bool:
        """Determines if the consultant approves the treatment plan."""
        vitals = traits.get("vitals_known", False)
        labs = traits.get("labs_known", False)
        stability = traits.get("vital_stability", 1.0)
        
        # Check if they presented the case properly (Theme #1)
        presented = decision_history.get(action.patient_id) == "present_case"

        if personality == "conservative":
            # Conservative needs full workup AND formal case presentation
            return vitals and labs and presented
        elif personality == "aggressive":
            # Aggressive wants action. Approves with vitals if stability is reasonable
            return vitals and stability > 0.5
        elif personality == "risk_averse":
            # Risk averse needs full workup plus reasonable stability
            return vitals and labs and presented and stability > 0.4
        return False

    def respond(self, action: Action, personality: str, approved: bool) -> str:
        """Generates personality-driven feedback."""
        if action.action_type == "ask_consultant":
            if approved:
                return f"[{personality.upper()}] Your workup is sufficient. I authorize the decision."
            else:
                if personality == "conservative":
                    return "[CONSERVATIVE] You lack definitive labs. I will not sign off on this."
                elif personality == "aggressive":
                    return "[AGGRESSIVE] You seem hesitant. Either treat or get out of the way."
                else:
                    return "[RISK-AVERSE] The situation is too volatile. I need more reassurance."
        
        if action.action_type == "decide_treatment":
            if approved:
                return "Proceed. I take full responsibility for this clinical path."
            else:
                return "Wait! I have not cleared this treatment. This is unauthorized!"
                
        return self.STYLE_TRAITS.get(personality, "I am observing.")


class PatientFamilyAgent:
    """Emotion-driven family agent that creates friction or support."""

    EMOTIONAL_PROFILES = ["angry", "scared", "cooperative"]

    def respond(self, action: Action, patient: Patient, family_state: Dict[str, Any]) -> tuple[Optional[str], float]:
        """Returns (concern_text, resistance_penalty)."""
        emotion = family_state.get("emotional_state", "scared")
        resistance = family_state.get("resistance", 0.5)
        
        concern = None
        penalty = 0.0
        
        if action.action_type == "reassure_patient":
            family_state["resistance"] = max(0.0, resistance - 0.2)
            family_state["emotional_state"] = "cooperative"
            return "Family feels heard. Their anxiety subsides slightly.", 0.0

        if emotion == "angry":
            concern = f"Family is shouting! 'Why is {patient.name} still not getting better? Do something!'"
            penalty = 0.2 # Extra time cost / friction
        elif emotion == "scared":
            concern = "Family is sobbing. They are asking if they should call other relatives."
            penalty = 0.1
        
        # Treatment without reassurance or explanation causes resistance
        if action.action_type == "decide_treatment" and resistance > 0.6:
            concern = "Family blocks the door! 'We don't trust this plan! Explain yourselves!'"
            penalty = 0.5 # Major friction
            
        return concern, penalty


class NurseAgent:
    """Supportive but literal bedside agent."""

    def respond(self, action: Action, vitals_known: bool) -> str:
        if action.action_type == "ask_nurse":
            if not vitals_known:
                return "The vitals are trending. I'll get them for you now."
            return "Vitals are on the monitor. Should I prepare for labs?"
        
        if action.action_type == "request_test":
            if not vitals_known:
                return "Doctor, we typically check vitals before sending labs. Proceed anyway?"
            return "Labs sent. I'll let you know the moment they hit the portal."
            
        return "Standing by for bedside orders."


class ScriptedWardTeam:
    """Coordinates complex interaction dynamics and multi-agent reasoning."""

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
        state_vars: Dict[str, Any], # vitals_known, labs_known, vital_stability
        consultant_personality: str,
        family_state: Dict[str, Any],
        decision_history: Dict[str, str],
    ) -> AgentReplies:
        
        # 1. Determine Approval
        # Base threshold modified by trust_score
        trust = state_vars.get("trust_score", 0.5)
        
        approved = self._consultant.get_approval(action, state_vars, consultant_personality, decision_history)
        
        # Override approval if trust is exceptionally high OR family is very cooperative
        if trust > 0.8 and state_vars.get("vitals_known"):
            approved = True
            
        # 2. Get Responses
        consultant_opinion = self._consultant.respond(action, consultant_personality, approved)
        nurse_status = self._nurse.respond(action, state_vars.get("vitals_known", False))
        family_concern, resistance_penalty = self._family.respond(action, patient, family_state)
        
        # 3. Compute trust delta for environment to apply
        trust_delta = 0.0
        if action.action_type == "reassure_patient": trust_delta += 0.1
        if action.action_type == "present_case": trust_delta += 0.05
        if family_state.get("emotional_state") == "angry": trust_delta -= 0.05
        
        # 4. Compute Alignment
        alignment = trust
        if approved: alignment += 0.1
        
        return AgentReplies(
            consultant_opinion=consultant_opinion,
            nurse_status=nurse_status,
            family_concern=family_concern,
            approval_granted=approved,
            team_alignment=max(0.0, min(1.0, alignment)),
            resistance_penalty=resistance_penalty,
            trust_delta=trust_delta,
        )
