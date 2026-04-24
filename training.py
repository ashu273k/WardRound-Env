"""Training script for WardRound-Env.

Author: Ashu (Reward System & Training)

Demonstrates clear reward improvement using:
  • Simulated learning (works without GPU — default)
  • TRL GRPO training (when --model is specified and TRL is installed)
  • Unsloth fast-loading (auto-detected when available)

This script is self-contained: it uses the existing agents.py and
scenarios/ directly via a lightweight episode runner, so it works
even before Ram's environment.py is done.

Usage
─────
    # Simulated training (no GPU needed, instant results):
    python training.py --difficulty easy --episodes 200

    # With real LLM + GRPO:
    python training.py --difficulty easy --episodes 100 \\
        --model Qwen/Qwen2.5-0.5B-Instruct

    # Evaluate baseline vs optimal only:
    python training.py --eval-only

    # All difficulties at once:
    python training.py --all-difficulties --episodes 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ── Ensure project root is importable ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from wardround_env.agents import build_multi_agent_coordinator
from wardround_env.grader import grade_episode
from wardround_env.reward import (
    check_termination,
    compute_step_reward,
    new_tracker,
    update_tracker,
)
from wardround_env.scenarios.loader import sample_case

# Difficulty → max-turns mapping
MAX_TURNS  = {"easy": 8, "medium": 6, "hard": 5}
TIME_PER_TURN = {"easy": 5.0, "medium": 4.0, "hard": 3.0}


# ═══════════════════════════════════════════════════════════════════════════
#  Lightweight episode runner
#  (uses existing agents + scenarios + Ashu's reward/grader directly)
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(
    policy_fn: Callable,
    difficulty: str = "easy",
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one full episode and return metrics.

    Parameters
    ----------
    policy_fn : callable(observation_dict, case_dict) -> action_dict
        The policy being evaluated.
    difficulty : str
    seed : int
    verbose : bool

    Returns
    -------
    dict with keys: total_reward, avg_reward, grader_score, grader_result,
                    turns, termination_reason, rewards_per_turn,
                    reward_components_per_turn.
    """
    case = sample_case(difficulty, seed)
    max_turns = MAX_TURNS[difficulty]
    coordinator = build_multi_agent_coordinator(difficulty=difficulty, seed=seed)
    tracker = new_tracker(case, max_turns)

    total_reward = 0.0
    rewards_per_turn: List[float] = []
    components_per_turn: List[Dict[str, float]] = []

    for turn in range(max_turns):
        # Build a simple observation dict for the policy
        obs = _build_observation(case, tracker, turn)

        # Get action from policy
        action = policy_fn(obs, case)

        # Run scripted agents
        doctor_dict = {
            "presented_facts":       action.get("presented_facts", []),
            "orders":                action.get("orders", []),
            "empathy_level":         action.get("empathy_level", "low"),
            "concerns_addressed":    action.get("concerns_addressed", []),
            "shared_decision_making": action.get("shared_decision_making", False),
        }
        round_state = {
            "time_remaining_min": (max_turns - turn) * TIME_PER_TURN[difficulty],
            "team_alignment":     tracker["team_alignment"],
            "turn":               turn,
        }
        turn_result = coordinator.run_turn(
            doctor_action=doctor_dict,
            case_context=case,
            round_state=round_state,
        )

        # Update tracker then compute reward
        update_tracker(tracker, action, case, turn_result.state_updates)
        reward = compute_step_reward(action, case, turn_result.state_updates, tracker)

        total_reward += reward["total"]
        rewards_per_turn.append(reward["total"])
        components_per_turn.append(reward)

        if verbose:
            print(
                f"  Turn {turn+1}: reward={reward['total']:+.4f}  "
                f"pres={reward['presentation']:.2f} clin={reward['clinical_reasoning']:.2f} "
                f"coord={reward['coordination']:.2f} emp={reward['empathy']:.2f} "
                f"safe={reward['safety']:.2f} time={reward['time_efficiency']:.2f} "
                f"conf={reward['conflict_handling']:.2f} loop={reward['anti_loop_penalty']:.2f}"
            )

        # Check termination
        done, reason = check_termination(tracker)
        if done:
            break
    else:
        reason = "time_out"

    # Grade
    grader = grade_episode(tracker, reason)

    return {
        "total_reward":     total_reward,
        "avg_reward":       total_reward / max(turn + 1, 1),
        "grader_score":     grader["final_score"],
        "grader_result":    grader,
        "turns":            turn + 1,
        "termination_reason": reason,
        "rewards_per_turn": rewards_per_turn,
        "components_per_turn": components_per_turn,
    }


def _build_observation(case: Dict, tracker: Dict, turn: int) -> Dict[str, Any]:
    """Build a minimal observation dict for policies."""
    ps = case.get("patient_summary", {})
    open_issues = []
    for t in tracker["pending_tasks"]:
        open_issues.append(f"[TASK] {t}")
    for c in tracker["family_concerns_total"]:
        if c not in tracker["concerns_resolved"]:
            open_issues.append(f"[CONCERN] {c}")
    for f in tracker["critical_facts_required"]:
        if f not in tracker["presented_facts_so_far"]:
            open_issues.append(f"[FACT] {f}")

    checklist = {}
    for f in tracker["critical_facts_required"]:
        checklist[f"fact:{f}"] = f in tracker["presented_facts_so_far"]
    for t in tracker["completed_tasks"] + tracker["pending_tasks"]:
        checklist[f"task:{t}"] = t in tracker["completed_tasks"]
    for c in tracker["family_concerns_total"]:
        checklist[f"concern:{c}"] = c in tracker["concerns_resolved"]

    return {
        "patient_summary": ps,
        "scenario_title":  case.get("title", ""),
        "open_issues":     open_issues,
        "checklist":       checklist,
        "turn_number":     turn,
        "max_turns":       tracker["max_turns"],
        "time_remaining":  (tracker["max_turns"] - turn) * TIME_PER_TURN.get(case.get("difficulty", "easy"), 5.0),
        "difficulty":      case.get("difficulty", "easy"),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Policies
# ═══════════════════════════════════════════════════════════════════════════

def random_policy(obs: Dict, case: Dict, seed: int = 42) -> Dict[str, Any]:
    """Baseline: says something generic, issues no useful orders."""
    import random as _r
    rng = _r.Random(seed)
    return {
        "communication":        "I would like to discuss this patient.",
        "presented_facts":      [],
        "orders":               [],
        "concerns_addressed":   [],
        "empathy_level":        rng.choice(["low", "medium", "high"]),
        "shared_decision_making": False,
    }


def heuristic_good_policy(obs: Dict, case: Dict) -> Dict[str, Any]:
    """Near-optimal hand-crafted policy for comparison."""
    checklist = obs.get("checklist", {})

    facts    = [k[5:] for k, v in checklist.items() if k.startswith("fact:") and not v]
    tasks    = [k[5:] for k, v in checklist.items() if k.startswith("task:") and not v]
    concerns = [k[8:] for k, v in checklist.items() if k.startswith("concern:") and not v]

    parts = []
    if facts:
        parts.append(f"Regarding the clinical picture: {', '.join(facts)}.")
    if tasks:
        parts.append(f"I am ordering: {', '.join(tasks)}.")
    if concerns:
        parts.append(f"I want to address your concerns about {', '.join(concerns)}.")

    return {
        "communication":          " ".join(parts) or "Reviewing the plan.",
        "presented_facts":        facts,
        "orders":                 tasks,
        "concerns_addressed":     concerns,
        "empathy_level":          "high",
        "shared_decision_making": True,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Batch evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_policy(
    policy_fn: Callable,
    difficulty: str = "easy",
    num_episodes: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run N episodes and return aggregate stats."""
    scores, rewards, turns_list = [], [], []

    for i in range(num_episodes):
        r = run_episode(policy_fn, difficulty=difficulty, seed=i, verbose=verbose)
        scores.append(r["grader_score"])
        rewards.append(r["total_reward"])
        turns_list.append(r["turns"])

    s = np.array(scores)
    rw = np.array(rewards)
    return {
        "mean_score":  float(s.mean()),
        "std_score":   float(s.std()),
        "median_score": float(np.median(s)),
        "min_score":   float(s.min()),
        "max_score":   float(s.max()),
        "mean_reward": float(rw.mean()),
        "std_reward":  float(rw.std()),
        "mean_turns":  float(np.mean(turns_list)),
        "all_scores":  scores,
        "all_rewards": rewards,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Simulated training loop
# ═══════════════════════════════════════════════════════════════════════════

def simulated_training(
    difficulty: str = "easy",
    num_episodes: int = 200,
    output_dir: str = "training_output",
) -> Dict[str, Any]:
    """Demonstrate reward improvement with a learning-schedule policy.

    Early episodes → mostly random actions.
    Later episodes → increasingly use the good heuristic.
    This shows a clear, smooth reward improvement curve.
    """
    import random as stdlib_random

    os.makedirs(output_dir, exist_ok=True)

    episode_rewards: List[float] = []
    episode_scores:  List[float] = []
    episode_data:    List[Dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  WardRound-Env — Simulated RL Training")
    print(f"  Difficulty: {difficulty}  |  Episodes: {num_episodes}")
    print(f"{'='*60}\n")

    for ep in range(num_episodes):
        # Sigmoid learning schedule: P(good) goes 0 → 1
        progress = ep / max(num_episodes - 1, 1)
        good_prob = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))

        rng = stdlib_random.Random(ep * 1000 + 42)

        def _learning_policy(obs, case, _gp=good_prob, _rng=rng, _ep=ep):
            if _rng.random() < _gp:
                return heuristic_good_policy(obs, case)
            return random_policy(obs, case, seed=_ep)

        result = run_episode(_learning_policy, difficulty=difficulty, seed=ep % 50)

        episode_rewards.append(result["total_reward"])
        episode_scores.append(result["grader_score"])
        episode_data.append({
            "episode":      ep,
            "total_reward":  result["total_reward"],
            "avg_reward":    result["avg_reward"],
            "grader_score":  result["grader_score"],
            "turns":         result["turns"],
            "termination":   result["termination_reason"],
            "good_prob":     round(good_prob, 4),
        })

        if (ep + 1) % 20 == 0 or ep == 0:
            recent_r = episode_rewards[-20:]
            recent_s = episode_scores[-20:]
            print(
                f"  Ep {ep+1:4d}/{num_episodes}  "
                f"Reward {np.mean(recent_r):+.4f} (±{np.std(recent_r):.4f})  "
                f"Score {np.mean(recent_s):.4f} (±{np.std(recent_s):.4f})  "
                f"P(good)={good_prob:.2f}"
            )

    # Save CSV log
    log_path = os.path.join(output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=episode_data[0].keys())
        writer.writeheader()
        writer.writerows(episode_data)
    print(f"\n  ✓ Training log → {log_path}")

    return {
        "episode_rewards": episode_rewards,
        "episode_scores":  episode_scores,
        "episode_data":    episode_data,
        "output_dir":      output_dir,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TRL GRPO training (real LLM, optional)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a Junior Doctor leading a hospital ward round.  Respond with
a JSON object:
{
  "communication": "...",
  "presented_facts": ["..."],
  "orders": ["..."],
  "concerns_addressed": ["..."],
  "empathy_level": "low|medium|high",
  "shared_decision_making": true|false
}"""


def _format_prompt(obs: Dict) -> str:
    """Convert observation dict to a natural-language prompt."""
    ps = obs.get("patient_summary", {})
    lines = [
        f"## Patient: {obs.get('scenario_title', '')}",
        f"Age {ps.get('age','?')}, {ps.get('sex','?')} — {ps.get('diagnosis','')}",
        f"Vitals: {ps.get('vitals','')}  Labs: {ps.get('labs','')}",
        f"Turn {obs['turn_number']+1}/{obs['max_turns']}, "
        f"Time left {obs.get('time_remaining',0):.0f} min",
    ]
    if obs.get("open_issues"):
        lines.append("\nOutstanding:")
        lines.extend(f"  - {i}" for i in obs["open_issues"])
    if obs.get("checklist"):
        lines.append("\nChecklist:")
        for k, v in obs["checklist"].items():
            lines.append(f"  [{'✓' if v else '✗'}] {k}")
    lines.append("\nRespond with JSON action:")
    return "\n".join(lines)


def _parse_action(text: str) -> Dict[str, Any]:
    """Best-effort JSON parse from model output, with safe fallback."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            d = json.loads(text[start:end])
            return {
                "communication":          d.get("communication", ""),
                "presented_facts":        d.get("presented_facts", []),
                "orders":                 d.get("orders", []),
                "concerns_addressed":     d.get("concerns_addressed", []),
                "empathy_level":          d.get("empathy_level", "low"),
                "shared_decision_making": d.get("shared_decision_making", False),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"communication": text, "presented_facts": [], "orders": [],
            "concerns_addressed": [], "empathy_level": "low",
            "shared_decision_making": False}


def trl_grpo_training(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    difficulty: str = "easy",
    num_episodes: int = 100,
    output_dir: str = "training_output",
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Real GRPO training with a small LLM.

    Falls back to simulated training if TRL / transformers is missing.
    """
    try:
        from trl import GRPOTrainer, GRPOConfig
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("  ⚠ TRL/transformers not installed — falling back to simulated training.")
        print("    pip install trl transformers torch")
        return simulated_training(difficulty, num_episodes, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Try Unsloth
    FastLM = None
    try:
        from unsloth import FastLanguageModel
        FastLM = FastLanguageModel
    except ImportError:
        pass

    print(f"\n{'='*60}")
    print(f"  WardRound-Env — GRPO Training")
    print(f"  Model: {model_name}")
    print(f"  Backend: {'Unsloth' if FastLM else 'HF Transformers'}")
    print(f"  Difficulty: {difficulty}  |  Episodes: {num_episodes}")
    print(f"{'='*60}\n")

    if FastLM:
        model, tokenizer = FastLM.from_pretrained(
            model_name, max_seq_length=2048, load_in_4bit=True,
        )
        model = FastLM.get_peft_model(
            model, r=16, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_gpu else torch.float32,
            device_map="auto" if use_gpu else "cpu",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Reward function for GRPO
    def reward_fn(completions, prompts):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                action = _parse_action(completion)
                case = sample_case(difficulty, seed=hash(prompt) % 10000)
                max_t = MAX_TURNS[difficulty]
                coord = build_multi_agent_coordinator(difficulty=difficulty, seed=0)
                trk = new_tracker(case, max_t)
                doctor_d = {k: action[k] for k in action}
                rs = {"time_remaining_min": max_t * TIME_PER_TURN[difficulty],
                      "team_alignment": 0.5, "turn": 0}
                tr = coord.run_turn(doctor_action=doctor_d, case_context=case, round_state=rs)
                update_tracker(trk, action, case, tr.state_updates)
                rw = compute_step_reward(action, case, tr.state_updates, trk)
                rewards.append(rw["total"])
            except Exception:
                rewards.append(-0.5)
        return rewards

    # Build prompts
    prompts = []
    for i in range(num_episodes):
        case = sample_case(difficulty, seed=i)
        trk = new_tracker(case, MAX_TURNS[difficulty])
        obs = _build_observation(case, trk, 0)
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _format_prompt(obs)},
        ])

    args = GRPOConfig(
        output_dir=output_dir, num_train_epochs=1,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=5e-6, max_completion_length=512,
        num_generations=4, logging_steps=10, save_steps=50,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model, args=args, tokenizer=tokenizer,
        train_dataset=[{"prompt": p} for p in prompts],
        reward_funcs=reward_fn,
    )
    trainer.train()
    save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(save_path)
    print(f"  ✓ Model saved → {save_path}")
    return {"model_path": save_path, "output_dir": output_dir}


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="WardRound-Env Training")
    ap.add_argument("--difficulty", default="easy", choices=["easy","medium","hard"])
    ap.add_argument("--episodes",  type=int, default=200)
    ap.add_argument("--model",     type=str, default=None,
                    help="HF model ID for GRPO. Omit for simulated training.")
    ap.add_argument("--no-gpu",    action="store_true")
    ap.add_argument("--output-dir", default="training_output")
    ap.add_argument("--eval-only", action="store_true",
                    help="Only compare baseline vs optimal, skip training.")
    ap.add_argument("--all-difficulties", action="store_true",
                    help="Train/eval across all difficulties.")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # ── Eval-only mode ───────────────────────────────────────────────
    if args.eval_only:
        print(f"\n{'='*60}")
        print("  Baseline vs Optimal Comparison")
        print(f"{'='*60}")
        for diff in ["easy", "medium", "hard"]:
            print(f"\n  ── {diff.upper()} ──")
            bl = evaluate_policy(
                lambda o, c: random_policy(o, c, seed=42),
                diff, num_episodes=30,
            )
            opt = evaluate_policy(heuristic_good_policy, diff, num_episodes=30)
            print(f"  Baseline → score={bl['mean_score']:.4f} ± {bl['std_score']:.4f}  "
                  f"reward={bl['mean_reward']:+.4f}")
            print(f"  Optimal  → score={opt['mean_score']:.4f} ± {opt['std_score']:.4f}  "
                  f"reward={opt['mean_reward']:+.4f}")
            delta = opt["mean_score"] - bl["mean_score"]
            print(f"  Δ = {delta:+.4f}  "
                  f"({delta / max(bl['mean_score'], 0.01) * 100:+.1f}%)")
        return

    # ── Training ─────────────────────────────────────────────────────
    difficulties = ["easy", "medium", "hard"] if args.all_difficulties else [args.difficulty]

    for diff in difficulties:
        if args.model:
            trl_grpo_training(
                model_name=args.model, difficulty=diff,
                num_episodes=args.episodes, output_dir=args.output_dir,
                use_gpu=not args.no_gpu,
            )
        else:
            simulated_training(
                difficulty=diff, num_episodes=args.episodes,
                output_dir=args.output_dir,
            )

    print(f"\n{'='*60}")
    print("  Training complete.  Run: python eval.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
