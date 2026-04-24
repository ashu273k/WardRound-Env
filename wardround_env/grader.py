"""Deterministic grader for WardRound-Env.

Author: Ashu (Reward System & Training)

Takes a complete episode tracker (from reward.py) and produces a
single score in [0.0, 1.0] with a transparent sub-score rubric.

Properties
──────────
• **Pure function**: no side effects.
• **Deterministic**: same inputs → same score, always.
• **No external calls**: no network, no LLM, no randomness.
• **Strict**: only counts *actionable orders*, not just communication.

Scoring Axes (weights)
──────────────────────
┌──────────────────────┬────────┬──────────────────────────────────────────┐
│ Axis                 │ Weight │ What it measures                         │
├──────────────────────┼────────┼──────────────────────────────────────────┤
│ Patient Outcome      │  0.30  │ Safety violations + patient trajectory   │
│ Plan Correctness     │  0.25  │ Required nursing actions actually done   │
│ Team Agreement       │  0.20  │ Final team_alignment value               │
│ Time Efficiency      │  0.10  │ Turns used vs. max_turns                 │
│ Communication Quality│  0.15  │ Critical facts + family concerns covered │
└──────────────────────┴────────┴──────────────────────────────────────────┘

Usage
─────
    from wardround_env.grader import grade_episode
    result = grade_episode(tracker, termination_reason="completed")
    print(result["final_score"])   # 0.0 – 1.0
    for line in result["rubric"]:
        print(line)
"""

from __future__ import annotations

from typing import Any, Dict, List


# ═══════════════════════════════════════════════════════════════════════════
# Axis weights — MUST sum to 1.0
# ═══════════════════════════════════════════════════════════════════════════
AX_PATIENT_OUTCOME  = 0.30
AX_PLAN_CORRECTNESS = 0.25
AX_TEAM_AGREEMENT   = 0.20
AX_TIME_EFFICIENCY  = 0.10
AX_COMMUNICATION    = 0.15

assert abs(
    AX_PATIENT_OUTCOME + AX_PLAN_CORRECTNESS + AX_TEAM_AGREEMENT
    + AX_TIME_EFFICIENCY + AX_COMMUNICATION - 1.0
) < 1e-9, "Grader axis weights must sum to 1.0"


# ═══════════════════════════════════════════════════════════════════════════
# Sub-score helpers
# ═══════════════════════════════════════════════════════════════════════════

def _patient_outcome(tracker: Dict[str, Any]) -> float:
    """0.0–1.0 based on safety and patient trajectory.

    • Each safety violation subtracts 0.25 (clamped to 0).
    • Trajectory bonus: improving → 1.0, stable → 0.7, worsening → 0.2.
    """
    traj_map = {"improving": 1.0, "stable": 0.7, "worsening": 0.2}
    base = traj_map.get(tracker["patient_trajectory"], 0.5)
    penalty = tracker["safety_violations"] * 0.25
    return max(0.0, min(1.0, base - penalty))


def _plan_correctness(tracker: Dict[str, Any]) -> float:
    """0.0–1.0 = fraction of required tasks completed.

    +0.1 bonus if no contraindicated orders were ever issued.
    """
    total = len(tracker["completed_tasks"]) + len(tracker["pending_tasks"])
    if total == 0:
        return 1.0                         # nothing required → perfect
    frac = len(tracker["completed_tasks"]) / total
    if not tracker["contraindicated_issued"]:
        frac = min(1.0, frac + 0.1)        # clean-execution bonus
    return max(0.0, min(1.0, frac))


def _team_agreement(tracker: Dict[str, Any]) -> float:
    """Simply the final team_alignment (already 0.0–1.0)."""
    return max(0.0, min(1.0, tracker["team_alignment"]))


def _time_efficiency(tracker: Dict[str, Any], termination: str) -> float:
    """Reward finishing early, penalise running out.

    • ≤50% turns used  → 1.0
    • ≤75%             → 0.9
    • ≤100%            → linear down to 0.5
    • time-out / safety → 0.2
    """
    if tracker["max_turns"] == 0:
        return 0.5
    if termination in ("time_out", "safety_failure"):
        return 0.2
    ratio = tracker["turns_elapsed"] / tracker["max_turns"]
    if ratio <= 0.5:
        return 1.0
    if ratio <= 0.75:
        return 0.9
    return max(0.5, 1.0 - ratio)


def _communication(tracker: Dict[str, Any]) -> float:
    """50/50 blend of critical-facts coverage and family-concerns resolved."""
    # Facts
    total_f = len(tracker["critical_facts_required"])
    if total_f > 0:
        covered = sum(1 for f in tracker["critical_facts_required"]
                      if f in tracker["presented_facts_so_far"])
        facts_score = covered / total_f
    else:
        facts_score = 1.0

    # Concerns
    total_c = len(tracker["family_concerns_total"])
    if total_c > 0:
        resolved = sum(1 for c in tracker["family_concerns_total"]
                       if c in tracker["concerns_resolved"])
        concerns_score = resolved / total_c
    else:
        concerns_score = 1.0

    return 0.5 * facts_score + 0.5 * concerns_score


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API — grade_episode
# ═══════════════════════════════════════════════════════════════════════════

def grade_episode(
    tracker: Dict[str, Any],
    termination_reason: str = "",
) -> Dict[str, Any]:
    """Grade a completed episode.

    Parameters
    ----------
    tracker : dict
        The episode tracker dict from ``reward.new_tracker`` that was
        mutated across all turns via ``reward.update_tracker``.
    termination_reason : str
        One of ``"completed"``, ``"time_out"``, ``"safety_failure"``, ``""``.

    Returns
    -------
    dict with keys:
        ``patient_outcome``, ``plan_correctness``, ``team_agreement``,
        ``time_efficiency``, ``communication_quality`` — each in [0, 1].
        ``final_score`` — weighted sum in [0, 1].
        ``rubric`` — list of human-readable breakdown strings.
    """
    po = _patient_outcome(tracker)
    pc = _plan_correctness(tracker)
    ta = _team_agreement(tracker)
    te = _time_efficiency(tracker, termination_reason)
    cq = _communication(tracker)

    final = (
        AX_PATIENT_OUTCOME  * po
        + AX_PLAN_CORRECTNESS * pc
        + AX_TEAM_AGREEMENT   * ta
        + AX_TIME_EFFICIENCY  * te
        + AX_COMMUNICATION    * cq
    )
    final = max(0.0, min(1.0, final))

    # ── Human-readable rubric ────────────────────────────────────────
    rubric: List[str] = [
        f"Patient Outcome:    {po:.3f}  (w={AX_PATIENT_OUTCOME})",
        f"  - Safety violations: {tracker['safety_violations']}",
        f"  - Trajectory: {tracker['patient_trajectory']}",
        f"Plan Correctness:   {pc:.3f}  (w={AX_PLAN_CORRECTNESS})",
        f"  - Completed {len(tracker['completed_tasks'])}"
        f"/{len(tracker['completed_tasks']) + len(tracker['pending_tasks'])} tasks",
        f"  - Unsafe orders: {tracker['contraindicated_issued']}",
        f"Team Agreement:     {ta:.3f}  (w={AX_TEAM_AGREEMENT})",
        f"  - Final alignment: {tracker['team_alignment']:.2f}",
        f"Time Efficiency:    {te:.3f}  (w={AX_TIME_EFFICIENCY})",
        f"  - Turns: {tracker['turns_elapsed']}/{tracker['max_turns']}",
        f"  - Termination: {termination_reason or 'normal'}",
        f"Communication:      {cq:.3f}  (w={AX_COMMUNICATION})",
        f"  - Facts: {len(tracker['presented_facts_so_far'])}"
        f"/{len(tracker['critical_facts_required'])}",
        f"  - Concerns: {len(tracker['concerns_resolved'])}"
        f"/{len(tracker['family_concerns_total'])}",
        f"─────────────────────────────────",
        f"FINAL SCORE:        {final:.4f}",
    ]

    return {
        "patient_outcome":      round(po, 4),
        "plan_correctness":     round(pc, 4),
        "team_agreement":       round(ta, 4),
        "time_efficiency":      round(te, 4),
        "communication_quality": round(cq, 4),
        "final_score":          round(final, 4),
        "rubric":               rubric,
    }
