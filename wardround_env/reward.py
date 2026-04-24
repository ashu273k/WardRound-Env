"""Shaped reward function for WardRound-Env.

Author: Ashu (Reward System & Training)

This module is STANDALONE — it does not depend on models.py or
environment.py.  Ram can import ``compute_step_reward()`` into
his environment's ``step()`` method.

Reward Design
─────────────
8 components, each normalised to [-1.0, +1.0].  The weighted sum
gives a total per-step reward roughly in [-1.0, +1.0].

┌────────────────────┬────────┬─────────┬──────────────────────────────────┐
│ Component          │ Weight │ Signal  │ Description                      │
├────────────────────┼────────┼─────────┼──────────────────────────────────┤
│ Presentation       │  0.15  │ Dense   │ Fraction of critical facts shown │
│ Clinical Reasoning │  0.15  │ Dense   │ Correct orders − unsafe penalty  │
│ Coordination       │  0.15  │ Dense   │ Nurse can execute; clear orders  │
│ Empathy            │  0.10  │ Dense   │ Family concerns + empathy level  │
│ Safety             │  0.20  │ Sparse  │ Big negative for unsafe orders   │
│ Time Efficiency    │  0.10  │ Dense   │ Tasks done early vs late         │
│ Conflict Handling  │  0.10  │ Dense   │ Team alignment improvement       │
│ Anti-Loop Penalty  │  0.05  │ Penalty │ Exponential penalty for repeats  │
└────────────────────┴────────┴─────────┴──────────────────────────────────┘

Anti-loop: hash each action; penalty = -min(1.0, 0.3 × 2^(repeats−1)).
Safety has the highest weight (0.20) because a single unsafe order
should dominate the learning signal.

Usage (by Ram in environment.py)
────────────────────────────────
    from wardround_env.reward import compute_step_reward

    reward = compute_step_reward(
        action        = doctor_action_dict,
        case_context  = loaded_scenario,
        state_updates = coordinator.run_turn(...).state_updates,
        tracker       = episode_tracker,   # mutable, kept across turns
    )
    # reward is a dict with all 8 components + "total"
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List


# ═══════════════════════════════════════════════════════════════════════════
# Reward weights — MUST sum to 1.0
# ═══════════════════════════════════════════════════════════════════════════
WEIGHTS = {
    "presentation":       0.15,
    "clinical_reasoning": 0.15,
    "coordination":       0.15,
    "empathy":            0.10,
    "safety":             0.20,
    "time_efficiency":    0.10,
    "conflict_handling":  0.10,
    "anti_loop_penalty":  0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Reward weights must sum to 1.0"


# ═══════════════════════════════════════════════════════════════════════════
# Tracker — mutable state that accumulates across turns in one episode
# ═══════════════════════════════════════════════════════════════════════════

def new_tracker(case_context: Dict[str, Any], max_turns: int) -> Dict[str, Any]:
    """Create a fresh tracker dict for a new episode.

    Call this once in ``reset()`` and pass the same dict to every
    ``compute_step_reward()`` call during the episode.
    """
    return {
        # Facts & tasks
        "critical_facts_required": list(case_context.get("critical_facts", [])),
        "presented_facts_so_far":  [],
        "pending_tasks":           list(case_context.get("required_nursing_actions", [])),
        "completed_tasks":         [],
        # Family concerns
        "family_concerns_total":   list(case_context.get("family_concerns", [])),
        "concerns_resolved":       [],
        # Safety
        "safety_violations":       0,
        "contraindicated_issued":  [],
        # Team dynamics
        "team_alignment":          0.5,   # starts neutral
        # Time
        "turns_elapsed":           0,
        "max_turns":               max_turns,
        # Anti-loop hashes
        "action_hashes":           [],
        # Patient trajectory
        "patient_trajectory":      "stable",  # "improving" | "stable" | "worsening"
        # Ethical / disagreement
        "ethical_conflict":        case_context.get("ethical_conflict", False),
        "shared_decision_made":    False,
        "consultant_disagreement": case_context.get("consultant_disagreement", False),
        "disagreement_resolved":   False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Helper: deterministic action hash for loop detection
# ═══════════════════════════════════════════════════════════════════════════

def _action_hash(action: Dict[str, Any]) -> int:
    """Return a deterministic integer for an action dict."""
    canonical = (
        action.get("communication", "").strip().lower()
        + "|" + ",".join(sorted(action.get("orders", [])))
        + "|" + ",".join(sorted(action.get("presented_facts", [])))
    )
    return int(hashlib.md5(canonical.encode()).hexdigest()[:12], 16)


# ═══════════════════════════════════════════════════════════════════════════
# Tracker update — call BEFORE computing reward each turn
# ═══════════════════════════════════════════════════════════════════════════

def update_tracker(
    tracker: Dict[str, Any],
    action: Dict[str, Any],
    case_context: Dict[str, Any],
    state_updates: Dict[str, Any],
) -> None:
    """Mutate *tracker* in place with the latest action & agent reactions.

    Must be called exactly once per turn, before ``compute_step_reward``.
    """
    t = tracker  # alias for brevity

    # ── Time ─────────────────────────────────────────────────────────
    t["turns_elapsed"] += 1

    # ── Presented facts ──────────────────────────────────────────────
    for fact in action.get("presented_facts", []):
        if fact not in t["presented_facts_so_far"]:
            t["presented_facts_so_far"].append(fact)

    # ── Orders → task completion + safety check ──────────────────────
    contraindicated = case_context.get("contraindicated_orders", [])
    for order in action.get("orders", []):
        if order in contraindicated:
            t["safety_violations"] += 1
            if order not in t["contraindicated_issued"]:
                t["contraindicated_issued"].append(order)
        if order in t["pending_tasks"]:
            t["pending_tasks"].remove(order)
            t["completed_tasks"].append(order)

    # ── Concerns addressed ───────────────────────────────────────────
    for concern in action.get("concerns_addressed", []):
        if concern in t["family_concerns_total"] and concern not in t["concerns_resolved"]:
            t["concerns_resolved"].append(concern)

    # ── Team alignment ───────────────────────────────────────────────
    delta = state_updates.get("team_alignment_delta", 0.0)
    t["team_alignment"] = max(0.0, min(1.0, t["team_alignment"] + delta))

    # ── Patient trajectory ───────────────────────────────────────────
    if t["safety_violations"] >= 2:
        t["patient_trajectory"] = "worsening"
    else:
        total = len(t["completed_tasks"]) + len(t["pending_tasks"])
        if total > 0 and len(t["completed_tasks"]) / total >= 0.5:
            t["patient_trajectory"] = "improving"

    # ── Ethical / disagreement tracking ──────────────────────────────
    if action.get("shared_decision_making", False) and t["ethical_conflict"]:
        t["shared_decision_made"] = True

    if t["consultant_disagreement"] and delta > 0 and not state_updates.get("safety_alert", False):
        t["disagreement_resolved"] = True

    # ── Anti-loop hash ───────────────────────────────────────────────
    t["action_hashes"].append(_action_hash(action))


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API — compute_step_reward
# ═══════════════════════════════════════════════════════════════════════════

def compute_step_reward(
    action: Dict[str, Any],
    case_context: Dict[str, Any],
    state_updates: Dict[str, Any],
    tracker: Dict[str, Any],
) -> Dict[str, float]:
    """Compute the shaped per-step reward with 8 components.

    Parameters
    ----------
    action : dict
        The Junior Doctor's action this turn.  Expected keys:
        ``presented_facts``, ``orders``, ``concerns_addressed``,
        ``empathy_level`` ("low"/"medium"/"high"),
        ``shared_decision_making`` (bool), ``communication`` (str).
    case_context : dict
        The loaded scenario (from scenarios/*.json).
    state_updates : dict
        Output of ``MultiAgentCoordinator.run_turn().state_updates``.
        Keys: ``team_alignment_delta``, ``unresolved_family_concerns``,
        ``nursing_blocked``, ``safety_alert``.
    tracker : dict
        Mutable episode tracker created by ``new_tracker()``.
        Must have been updated via ``update_tracker()`` before calling.

    Returns
    -------
    dict
        Keys are the 8 component names + ``"total"``.
        Each component is in [-1.0, +1.0]; total is weighted sum.
    """
    t = tracker

    # ── 1. PRESENTATION (dense) ──────────────────────────────────────
    #    Fraction of critical facts presented THIS turn.
    total_critical = len(t["critical_facts_required"])
    if total_critical > 0:
        new_facts = [
            f for f in action.get("presented_facts", [])
            if f in t["critical_facts_required"]
        ]
        presentation = len(new_facts) / total_critical
    else:
        presentation = 0.5  # nothing required → neutral

    # ── 2. CLINICAL REASONING (dense) ────────────────────────────────
    #    Reward correct orders, heavy penalty for unsafe ones.
    required = case_context.get("required_nursing_actions", [])
    contraindicated = case_context.get("contraindicated_orders", [])
    orders = action.get("orders", [])

    correct = [o for o in orders if o in required]
    unsafe  = [o for o in orders if o in contraindicated]

    clinical = (len(correct) / len(required)) if required else 0.5
    clinical -= len(unsafe) * 0.5        # heavy penalty per unsafe order
    clinical = max(-1.0, min(1.0, clinical))

    # ── 3. COORDINATION (dense) ──────────────────────────────────────
    #    Nurse confirms vs. is blocked.
    nursing_blocked = state_updates.get("nursing_blocked", False)
    if not nursing_blocked and orders:
        coordination = 0.8        # orders clear + actionable
    elif not nursing_blocked and not orders:
        coordination = 0.0        # nothing ordered but nothing blocked
    else:
        coordination = -0.5       # nurse needs clarification

    # ── 4. EMPATHY (dense) ───────────────────────────────────────────
    concerns = case_context.get("family_concerns", [])
    addressed = [c for c in action.get("concerns_addressed", []) if c in concerns]
    unresolved = state_updates.get("unresolved_family_concerns", True)

    empathy = 0.0
    if concerns:
        empathy += len(addressed) / len(concerns) * 0.6
    empathy += {"low": -0.2, "medium": 0.1, "high": 0.3}.get(
        action.get("empathy_level", "low"), 0.0
    )
    if not unresolved:
        empathy += 0.2            # family is satisfied
    empathy = max(-1.0, min(1.0, empathy))

    # ── 5. SAFETY (sparse, high-impact) ──────────────────────────────
    if unsafe:
        safety = -1.0             # catastrophic: issued contraindicated order
    elif t["safety_violations"] > 0:
        safety = -0.3             # lingering penalty from past violations
    else:
        safety = 0.5              # clean record

    # ── 6. TIME EFFICIENCY (dense) ───────────────────────────────────
    total_tasks = len(t["completed_tasks"]) + len(t["pending_tasks"])
    completion = (len(t["completed_tasks"]) / total_tasks) if total_tasks > 0 else 1.0
    turns_ratio = t["turns_elapsed"] / t["max_turns"] if t["max_turns"] > 0 else 1.0

    if turns_ratio <= 0.5:
        time_eff = completion     # early → reward any progress
    else:
        time_eff = completion - (turns_ratio - 0.5)  # late → penalise slowness
    time_efficiency = max(-1.0, min(1.0, time_eff))

    # ── 7. CONFLICT HANDLING (dense) ─────────────────────────────────
    alignment_delta = state_updates.get("team_alignment_delta", 0.0)
    conflict = alignment_delta / 0.4     # normalise [-0.4, 0.4] → [-1, 1]
    if state_updates.get("safety_alert", False):
        conflict -= 0.5                  # safety alert is partly doctor's fault
    conflict_handling = max(-1.0, min(1.0, conflict))

    # ── 8. ANTI-LOOP PENALTY ─────────────────────────────────────────
    h = _action_hash(action)
    # count repeats before current turn (current hash already appended)
    repeat_count = t["action_hashes"][:-1].count(h)
    anti_loop = -min(1.0, 0.3 * (2 ** (repeat_count - 1))) if repeat_count > 0 else 0.0

    # ── WEIGHTED TOTAL ───────────────────────────────────────────────
    components = {
        "presentation":       round(presentation, 4),
        "clinical_reasoning": round(clinical, 4),
        "coordination":       round(coordination, 4),
        "empathy":            round(empathy, 4),
        "safety":             round(safety, 4),
        "time_efficiency":    round(time_efficiency, 4),
        "conflict_handling":  round(conflict_handling, 4),
        "anti_loop_penalty":  round(anti_loop, 4),
    }

    total = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    components["total"] = round(total, 4)

    return components


# ═══════════════════════════════════════════════════════════════════════════
# Termination check — optional helper for Ram's environment
# ═══════════════════════════════════════════════════════════════════════════

def check_termination(tracker: Dict[str, Any]) -> tuple:
    """Check whether the episode should end.

    Returns
    -------
    (done: bool, reason: str)
        reason is one of "completed", "time_out", "safety_failure", or "".
    """
    t = tracker

    # All objectives met → success
    all_facts   = all(f in t["presented_facts_so_far"] for f in t["critical_facts_required"])
    all_tasks   = len(t["pending_tasks"]) == 0
    all_concerns = all(c in t["concerns_resolved"] for c in t["family_concerns_total"])
    if all_facts and all_tasks and all_concerns:
        return True, "completed"

    # Out of turns
    if t["turns_elapsed"] >= t["max_turns"]:
        return True, "time_out"

    # Catastrophic safety failure (3+ violations)
    if t["safety_violations"] >= 3:
        t["patient_trajectory"] = "worsening"
        return True, "safety_failure"

    return False, ""
