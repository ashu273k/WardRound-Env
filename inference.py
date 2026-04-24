"""Baseline deterministic inference script for WardRound-Env."""

from __future__ import annotations

import argparse

from models import Action
from server.environment import WardRoundEnvironment


def run_episode(task_id: str, seed: int) -> None:
    env = WardRoundEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    print(f"Episode: {env.state.episode_id} task={task_id} seed={seed}")
    print(f"Start patient: {obs.current_patient.id} | time_remaining={obs.time_remaining}")

    policy_actions = [
        Action(action_type="present_case", patient_id=obs.current_patient.id, content="Concise case summary."),
        Action(action_type="decide_treatment", patient_id=obs.current_patient.id, content="Initiate treatment plan."),
    ]

    for idx, action in enumerate(policy_actions, start=1):
        obs = env.step(action)
        print(
            f"step={idx} action={action.action_type} reward={obs.reward} "
            f"done={obs.done} consultant='{obs.consultant_opinion}'"
        )
        if obs.done:
            break

    print(f"Final step_count={env.state.step_count} decisions={env.state.decisions}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WardRound-Env baseline inference.")
    parser.add_argument("--task-id", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_episode(task_id=args.task_id, seed=args.seed)


if __name__ == "__main__":
    main()
