"""Evaluation & visualization for WardRound-Env.

Author: Ashu (Reward System & Training)

Generates:
  1. Reward curve over training episodes
  2. Grader score distribution (before vs after)
  3. Sub-score radar chart
  4. Per-difficulty comparison bar chart
  5. Per-step reward component breakdown
  6. Text summary of before/after behavior

Usage
─────
    # Full eval (trains + plots):
    python eval.py --full

    # Quick eval (fewer episodes):
    python eval.py --quick

    # Use existing training log:
    python eval.py --training-log training_output/training_log.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from training import (
    evaluate_policy,
    heuristic_good_policy,
    random_policy,
    run_episode,
    simulated_training,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Matplotlib setup
# ═══════════════════════════════════════════════════════════════════════════

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 1: Reward curve
# ═══════════════════════════════════════════════════════════════════════════

def plot_reward_curve(
    rewards: List[float],
    scores: List[float],
    path: str,
    window: int = 20,
) -> str:
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150)
    fig.suptitle("WardRound-Env Training Progress", fontsize=16, fontweight="bold")

    eps = np.arange(len(rewards))

    # Rewards
    ax1.plot(eps, rewards, alpha=0.2, color="#6366f1", lw=0.8)
    if len(rewards) >= window:
        sm = np.convolve(rewards, np.ones(window)/window, "valid")
        ax1.plot(np.arange(window-1, len(rewards)), sm,
                 color="#6366f1", lw=2.5, label=f"Reward (MA-{window})")
    ax1.set_ylabel("Total Episode Reward"); ax1.set_xlabel("Episode")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_title("Shaped Reward Signal")

    # Scores
    ax2.plot(eps, scores, alpha=0.2, color="#f59e0b", lw=0.8)
    if len(scores) >= window:
        sm = np.convolve(scores, np.ones(window)/window, "valid")
        ax2.plot(np.arange(window-1, len(scores)), sm,
                 color="#f59e0b", lw=2.5, label=f"Grader Score (MA-{window})")
    ax2.set_ylabel("Grader Score [0,1]"); ax2.set_xlabel("Episode")
    ax2.set_ylim(-0.05, 1.05); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title("Deterministic Grader Score")

    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  ✓ Reward curve → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 2: Score distribution histogram
# ═══════════════════════════════════════════════════════════════════════════

def plot_score_distribution(
    baseline: List[float],
    trained: List[float],
    path: str,
) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bins = np.linspace(0, 1, 21)
    ax.hist(baseline, bins, alpha=0.6, color="#ef4444",
            label=f"Baseline (μ={np.mean(baseline):.3f})", edgecolor="white", lw=0.5)
    ax.hist(trained, bins, alpha=0.6, color="#22c55e",
            label=f"Trained (μ={np.mean(trained):.3f})", edgecolor="white", lw=0.5)
    ax.set_xlabel("Grader Score"); ax.set_ylabel("Count")
    ax.set_title("Score Distribution: Baseline vs Trained", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  ✓ Score distribution → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 3: Radar chart (sub-scores)
# ═══════════════════════════════════════════════════════════════════════════

def plot_radar(bl_sub: Dict[str,float], tr_sub: Dict[str,float], path: str) -> str:
    plt = _plt()
    cats = list(bl_sub.keys())
    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    bl_vals = [bl_sub[c] for c in cats] + [bl_sub[cats[0]]]
    tr_vals = [tr_sub[c] for c in cats] + [tr_sub[cats[0]]]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150, subplot_kw=dict(polar=True))
    ax.plot(angles, bl_vals, "o-", lw=2, color="#ef4444", label="Baseline", ms=6)
    ax.fill(angles, bl_vals, alpha=0.15, color="#ef4444")
    ax.plot(angles, tr_vals, "o-", lw=2, color="#22c55e", label="Trained", ms=6)
    ax.fill(angles, tr_vals, alpha=0.15, color="#22c55e")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Sub-Score Breakdown", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  ✓ Radar chart → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 4: Per-difficulty bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_difficulty_bars(results: Dict[str, Dict], path: str) -> str:
    plt = _plt()
    diffs = list(results.keys())
    bl_m = [results[d]["bl_mean"] for d in diffs]
    tr_m = [results[d]["tr_mean"] for d in diffs]
    bl_s = [results[d]["bl_std"]  for d in diffs]
    tr_s = [results[d]["tr_std"]  for d in diffs]

    x = np.arange(len(diffs)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    b1 = ax.bar(x-w/2, bl_m, w, yerr=bl_s, label="Baseline",
                color="#ef4444", alpha=0.8, capsize=5, edgecolor="white")
    b2 = ax.bar(x+w/2, tr_m, w, yerr=tr_s, label="Trained",
                color="#22c55e", alpha=0.8, capsize=5, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in diffs], fontsize=12)
    ax.set_ylabel("Mean Grader Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Performance by Difficulty", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(alpha=0.3, axis="y")
    for bar in list(b1)+list(b2):
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,5), textcoords="offset points", ha="center", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  ✓ Difficulty comparison → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 5: Per-step reward components (single episode)
# ═══════════════════════════════════════════════════════════════════════════

def plot_reward_components(difficulty: str, path: str) -> str:
    plt = _plt()
    result = run_episode(heuristic_good_policy, difficulty=difficulty, seed=0)
    comps = result["components_per_turn"]

    names = ["presentation","clinical_reasoning","coordination","empathy",
             "safety","time_efficiency","conflict_handling","anti_loop_penalty"]
    colors = ["#6366f1","#22c55e","#f59e0b","#ec4899",
              "#ef4444","#06b6d4","#8b5cf6","#6b7280"]

    turns = np.arange(1, len(comps)+1)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for name, clr in zip(names, colors):
        ax.plot(turns, [c[name] for c in comps], "o-",
                label=name.replace("_"," ").title(), color=clr, lw=2, ms=5)
    ax.set_xlabel("Turn"); ax.set_ylabel("Component Value")
    ax.set_title(f"Reward Components ({difficulty.capitalize()})", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, loc="lower left")
    ax.grid(alpha=0.3); ax.axhline(0, color="gray", ls="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  ✓ Reward components ({difficulty}) → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Text: behavior comparison
# ═══════════════════════════════════════════════════════════════════════════

def behavior_comparison(path: str) -> str:
    lines = ["="*70, "  WardRound-Env: Behavior Comparison", "="*70, ""]

    for diff in ["easy", "medium", "hard"]:
        lines += [f"\n{'─'*60}", f"  DIFFICULTY: {diff.upper()}", f"{'─'*60}"]

        bl = run_episode(lambda o,c: random_policy(o,c,0), diff, seed=0)
        tr = run_episode(heuristic_good_policy, diff, seed=0)

        for label, r in [("BASELINE (random)", bl), ("TRAINED (heuristic)", tr)]:
            g = r["grader_result"]
            lines += [
                f"\n  {label}:",
                f"    Reward:       {r['total_reward']:+.4f}",
                f"    Grader score: {r['grader_score']:.4f}",
                f"    Turns:        {r['turns']}",
                f"    Termination:  {r['termination_reason']}",
                f"    Patient:      {g['patient_outcome']:.3f}",
                f"    Plan:         {g['plan_correctness']:.3f}",
                f"    Team:         {g['team_agreement']:.3f}",
                f"    Time:         {g['time_efficiency']:.3f}",
                f"    Communication:{g['communication_quality']:.3f}",
            ]

        d = tr["grader_score"] - bl["grader_score"]
        lines.append(f"\n  IMPROVEMENT: {d:+.4f} "
                     f"({d/max(bl['grader_score'],0.01)*100:+.1f}%)")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(report)
    print(f"\n  ✓ Behavior comparison → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def full_eval(
    num_episodes: int = 50,
    output_dir: str = "training_output",
    training_log: str = None,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  WardRound-Env — Full Evaluation")
    print(f"{'='*60}")

    # ── 1. Reward curve ──────────────────────────────────────────────
    if training_log and os.path.exists(training_log):
        print(f"\n  Loading training log: {training_log}")
        rws, scs = [], []
        with open(training_log) as f:
            for row in csv.DictReader(f):
                rws.append(float(row["total_reward"]))
                scs.append(float(row["grader_score"]))
    else:
        print("\n  No log found — running simulated training…")
        res = simulated_training("easy", 200, output_dir)
        rws, scs = res["episode_rewards"], res["episode_scores"]

    plot_reward_curve(rws, scs, os.path.join(output_dir, "reward_curve.png"))

    # ── 2. Score distribution ────────────────────────────────────────
    print("\n  Evaluating baseline…")
    bl = evaluate_policy(lambda o,c: random_policy(o,c,42), "easy", num_episodes)
    print("  Evaluating trained…")
    tr = evaluate_policy(heuristic_good_policy, "easy", num_episodes)

    plot_score_distribution(bl["all_scores"], tr["all_scores"],
                            os.path.join(output_dir, "score_distribution.png"))

    # ── 3. Radar chart ───────────────────────────────────────────────
    from wardround_env.grader import grade_episode as _ge
    from wardround_env.reward import new_tracker as _nt

    bl_sub = {k: 0.0 for k in ["patient_outcome","plan_correctness",
              "team_agreement","time_efficiency","communication"]}
    tr_sub = dict(bl_sub)
    n = min(30, num_episodes)

    for i in range(n):
        r_bl = run_episode(lambda o,c: random_policy(o,c,i), "easy", seed=i)
        r_tr = run_episode(heuristic_good_policy, "easy", seed=i)
        for k in bl_sub:
            bl_sub[k] += r_bl["grader_result"].get(k, r_bl["grader_result"].get("communication_quality", 0))
            tr_sub[k] += r_tr["grader_result"].get(k, r_tr["grader_result"].get("communication_quality", 0))
    bl_sub = {k: v/n for k,v in bl_sub.items()}
    tr_sub = {k: v/n for k,v in tr_sub.items()}

    plot_radar(bl_sub, tr_sub, os.path.join(output_dir, "radar_chart.png"))

    # ── 4. Per-difficulty bars ───────────────────────────────────────
    print("\n  Per-difficulty comparison…")
    diff_res = {}
    for d in ["easy","medium","hard"]:
        b = evaluate_policy(lambda o,c: random_policy(o,c,42), d, min(30,num_episodes))
        t = evaluate_policy(heuristic_good_policy, d, min(30,num_episodes))
        diff_res[d] = {"bl_mean": b["mean_score"], "bl_std": b["std_score"],
                       "tr_mean": t["mean_score"], "tr_std": t["std_score"]}
    plot_difficulty_bars(diff_res, os.path.join(output_dir, "difficulty_comparison.png"))

    # ── 5. Reward components ─────────────────────────────────────────
    print("\n  Reward component breakdowns…")
    for d in ["easy","medium","hard"]:
        plot_reward_components(d, os.path.join(output_dir, f"reward_components_{d}.png"))

    # ── 6. Behavior comparison text ──────────────────────────────────
    behavior_comparison(os.path.join(output_dir, "behavior_comparison.txt"))

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✓ Evaluation complete!")
    print(f"{'='*60}")
    print(f"  Baseline: {bl['mean_score']:.4f} ± {bl['std_score']:.4f}")
    print(f"  Trained:  {tr['mean_score']:.4f} ± {tr['std_score']:.4f}")
    print(f"  Δ = {tr['mean_score']-bl['mean_score']:+.4f}")
    print(f"\n  All outputs in: {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="WardRound-Env Evaluation & Plots")
    ap.add_argument("--training-log", type=str, default=None)
    ap.add_argument("--full",  action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--num_episodes", type=int, default=50)
    ap.add_argument("--output-dir", default="training_output")
    args = ap.parse_args()

    n = 10 if args.quick else args.num_episodes
    full_eval(n, args.output_dir, args.training_log)


if __name__ == "__main__":
    main()
