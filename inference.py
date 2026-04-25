"""Inference script for WardRound-Env.

Runs all 3 tasks (easy, medium, hard) and prints a clear GRADER SCORE
for each one. Supports both local and remote server modes.

Usage
─────
    # Local mode (default):
    python inference.py

    # Single task:
    python inference.py --task-id easy --seed 42

    # Remote server mode:
    python inference.py --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import os
import sys

from models import Action
from server.environment import WardRoundEnvironment


# ═══════════════════════════════════════════════════════════════════════════
#  Local episode runner
# ═══════════════════════════════════════════════════════════════════════════

def run_episode_local(task_id: str, seed: int) -> dict:
    """Run one episode locally and return results with grader score."""
    env = WardRoundEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    print(f"  Episode: {env.state.episode_id}")
    print(f"  Patient: {obs.current_patient.name} (ID: {obs.current_patient.id})")
    print(f"  Condition: {obs.current_patient.condition}")
    print(f"  Time budget: {obs.time_remaining} turns")
    print()

    # Deterministic baseline policy: present case → decide treatment
    # per patient until all patients are seen
    step = 0
    while not obs.done:
        patient = obs.current_patient

        # Step 1: Present the case
        action = Action(
            action_type="present_case",
            patient_id=patient.id,
            content=f"Presenting {patient.name}, {patient.age}y, {patient.condition}.",
        )
        obs = env.step(action)
        step += 1
        print(
            f"  Step {step}: present_case -> reward={obs.reward:.3f} "
            f"| consultant='{obs.consultant_opinion}'"
        )
        if obs.done:
            break

        # Step 2: Reassure patient/family if concern is raised
        if obs.family_concern:
            action = Action(
                action_type="reassure_patient",
                patient_id=patient.id,
                content="Addressing family concerns and explaining the care plan.",
            )
            obs = env.step(action)
            step += 1
            print(
                f"  Step {step}: reassure_patient -> reward={obs.reward:.3f} "
                f"| family='{obs.family_concern}'"
            )
            if obs.done:
                break

        # Step 3: Decide treatment
        action = Action(
            action_type="decide_treatment",
            patient_id=patient.id,
            content="Initiating treatment plan based on clinical assessment.",
            reason="Evidence-based guideline approach.",
        )
        obs = env.step(action)
        step += 1
        print(
            f"  Step {step}: decide_treatment -> reward={obs.reward:.3f} "
            f"| consultant='{obs.consultant_opinion}'"
        )
        if obs.done:
            break

    # Extract grader score from metadata
    grader_score = obs.metadata.get("grader_score", None)
    grader_rubric = obs.metadata.get("grader_rubric", {})

    return {
        "task_id": task_id,
        "seed": seed,
        "steps": step,
        "decisions": dict(env.state.decisions),
        "grader_score": grader_score,
        "grader_rubric": grader_rubric,
        "final_reward": obs.reward,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Remote server mode (via OpenEnv client)
# ═══════════════════════════════════════════════════════════════════════════

def run_episode_remote(server_url: str, task_id: str, seed: int) -> dict:
    """Run one episode against a remote WardRound-Env server."""
    try:
        from client import WardRoundEnvClient
    except ImportError:
        print("  ⚠ client.py not available. Falling back to local mode.")
        return run_episode_local(task_id, seed)

    try:
        client = WardRoundEnvClient(base_url=server_url)
        result = client.reset(task_id=task_id, seed=seed)
        print(f"  Connected to server at {server_url}")
        print(f"  Task: {task_id} | Seed: {seed}")

        # Simple 2-step policy via client
        actions = [
            Action(action_type="present_case", patient_id="P001", content="Case summary."),
            Action(action_type="decide_treatment", patient_id="P001", content="Treatment plan."),
        ]
        for i, action in enumerate(actions, 1):
            result = client.step(action)
            print(f"  Step {i}: {action.action_type} -> done={result.done}")
            if result.done:
                break

        return {
            "task_id": task_id,
            "seed": seed,
            "steps": i,
            "grader_score": getattr(result.observation, "metadata", {}).get("grader_score"),
            "final_reward": result.reward,
        }
    except Exception as e:
        print(f"  ⚠ Server connection failed: {e}")
        print("  Falling back to local mode.")
        return run_episode_local(task_id, seed)


# ═══════════════════════════════════════════════════════════════════════════
#  Main — runs all tasks and prints grader scores
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WardRound-Env inference — run episodes and report grader scores."
    )
    parser.add_argument(
        "--task-id", default=None, choices=["easy", "medium", "hard"],
        help="Run a single task. If omitted, runs all 3 tasks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--server-url", type=str, default=None,
        help="URL of a running WardRound-Env server (e.g. http://localhost:8000). "
             "Uses ENV_BASE_URL env var if set.",
    )
    args = parser.parse_args()

    # Support ENV_BASE_URL environment variable
    server_url = args.server_url or os.environ.get("ENV_BASE_URL")
    tasks = [args.task_id] if args.task_id else ["easy", "medium", "hard"]
    run_fn = (
        lambda tid, s: run_episode_remote(server_url, tid, s)
        if server_url
        else run_episode_local
    )

    print()
    print("=" * 60)
    print("  WardRound-Env — Inference")
    print("=" * 60)

    results = []
    for task_id in tasks:
        print(f"\n{'-' * 60}")
        print(f"  TASK: {task_id.upper()}")
        print(f"{'-' * 60}")
        print()

        if server_url:
            result = run_episode_remote(server_url, task_id, args.seed)
        else:
            result = run_episode_local(task_id, args.seed)
        results.append(result)
        print()

    # ── Summary table ─────────────────────────────────────────────────
    print("=" * 60)
    print("  GRADER SCORES SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<10} {'Score':>10} {'Steps':>8} {'Decisions':>12}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*12}")

    for r in results:
        score = r["grader_score"]
        score_str = f"{score:.4f}" if score is not None else "N/A"
        decisions = len(r.get("decisions", {}))
        print(f"  {r['task_id']:<10} {score_str:>10} {r['steps']:>8} {decisions:>12}")

    # Print rubric detail for each task
    for r in results:
        rubric = r.get("grader_rubric", {})
        if rubric:
            print(f"\n  Rubric for {r['task_id'].upper()}:")
            for axis, val in rubric.items():
                label = axis.replace("_", " ").title()
                bar = "#" * int(val * 20) + "." * (20 - int(val * 20))
                print(f"    {label:<22} {bar} {val:.4f}")

    print(f"\n{'=' * 60}")
    avg_score = sum(
        r["grader_score"] for r in results if r["grader_score"] is not None
    ) / max(len(results), 1)
    print(f"  Average Grader Score: {avg_score:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
