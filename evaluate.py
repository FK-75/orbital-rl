"""
evaluate.py — Benchmark trained agents and generate learning curves
====================================================================
Two modes:

  1. Evaluate a single model across many episodes:
       python evaluate.py --task docking

  2. Generate a learning curve from checkpoint models:
       python evaluate.py --task docking --curve
       (saves assets/learning_curve_docking.png)

Usage:
    python evaluate.py --task docking --episodes 100
    python evaluate.py --task docking --curve
    python evaluate.py --task station_keeping --episodes 50
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from stable_baselines3 import PPO

from envs.orbital_env import OrbitalEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",     default="docking",
                   choices=["docking", "station_keeping"])
    # ADDED MODE HERE
    p.add_argument("--mode",     default="2d", choices=["2d", "3d"],
                   help="2d or 3d mode")
    p.add_argument("--episodes", type=int, default=100,
                   help="Number of evaluation episodes per model")
    p.add_argument("--curve",    action="store_true",
                   help="Generate full learning curve from evaluations.npz")
    p.add_argument("--model",    default=None,
                   help="Specific model path (uses best model if omitted)")
    return p.parse_args()


# ── Single model evaluation ───────────────────────────────────────────────────

def evaluate_model(model_path: str, task: str, mode: str, n_episodes: int) -> dict:
    model = PPO.load(model_path)
    # PASS THE MODE HERE
    env   = OrbitalEnv(task=task, mode=mode)

    rewards, distances, speeds, fuels, outcomes = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        distances.append(info["distance"])
        speeds.append(info["speed"])
        fuels.append(info["fuel"])
        outcomes.append(info.get("outcome", "timeout"))

    env.close()

    outcome_counts = {o: outcomes.count(o) for o in set(outcomes)}
    success_rate   = outcome_counts.get("docked", 0) / n_episodes * 100

    return {
        "mean_reward":   float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "success_rate":  success_rate,
        "mean_dist":     float(np.mean(distances)),
        "mean_speed":    float(np.mean(speeds)),
        "mean_fuel_left":float(np.mean(fuels)),
        "outcomes":      outcome_counts,
        "n_episodes":    n_episodes,
    }


def print_report(stats: dict, model_path: str):
    print(f"\n{'='*55}")
    print(f"  Model   : {model_path}")
    print(f"  Episodes: {stats['n_episodes']}")
    print(f"{'='*55}")
    print(f"  Mean reward     : {stats['mean_reward']:+.1f} ± {stats['std_reward']:.1f}")
    print(f"  Docking success : {stats['success_rate']:.1f}%")
    print(f"  Mean final dist : {stats['mean_dist']:.2f} m")
    print(f"  Mean final speed: {stats['mean_speed']:.3f} m/s")
    print(f"  Mean fuel left  : {stats['mean_fuel_left']:.1f} kg")
    print(f"\n  Outcomes:")
    for outcome, count in sorted(stats["outcomes"].items()):
        pct = count / stats["n_episodes"] * 100
        bar = "█" * int(pct / 2)
        print(f"    {outcome:12s}: {count:4d}  ({pct:5.1f}%)  {bar}")
    print(f"{'='*55}\n")


# ── Learning curve from evaluations.npz ──────────────────────────────────────

def plot_learning_curve(task: str):
    npz_path = Path("logs") / task / "evaluations.npz"
    if not npz_path.exists():
        print(f"No evaluations.npz found at {npz_path}. Train first.")
        return

    data      = np.load(npz_path)
    timesteps = data["timesteps"]
    results   = data["results"]       # (n_evals, n_episodes)
    ep_lengths= data["ep_lengths"]

    mean_r  = results.mean(axis=1)
    std_r   = results.std(axis=1)
    mean_l  = ep_lengths.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor="#0d0d1a",
                                    sharex=True)
    fig.suptitle(f"Training Progress — {task.replace('_', ' ').title()} Task",
                 color="white", fontsize=14, y=0.98)

    _ts = timesteps / 1_000  # → thousands

    # ── Reward panel ────────────────────────────────────────────────────────
    ax1.set_facecolor("#0d0d1a")
    ax1.fill_between(_ts, mean_r - std_r, mean_r + std_r,
                     alpha=0.2, color="#00d4ff")
    ax1.plot(_ts, mean_r, color="#00d4ff", lw=2, label="Mean reward")

    # Mark best checkpoint
    best_idx = int(np.argmax(mean_r))
    ax1.axvline(_ts[best_idx], color="#ffd700", lw=1, ls="--", alpha=0.6)
    ax1.plot(_ts[best_idx], mean_r[best_idx], "o",
             color="#ffd700", ms=8, zorder=5,
             label=f"Best ({mean_r[best_idx]:.0f} @ {timesteps[best_idx]/1e6:.2f}M)")

    ax1.set_ylabel("Mean Episode Reward", color="white")
    ax1.tick_params(colors="#888")
    ax1.spines[:].set_color("#333")
    ax1.yaxis.label.set_color("white")
    ax1.legend(facecolor="#111", labelcolor="white", fontsize=9)
    ax1.grid(axis="y", color="#222", lw=0.5)

    # ── Episode length panel ─────────────────────────────────────────────────
    ax2.set_facecolor("#0d0d1a")
    ax2.fill_between(_ts, ep_lengths.min(axis=1), ep_lengths.max(axis=1),
                     alpha=0.15, color="#ff6ec7")
    ax2.plot(_ts, mean_l, color="#ff6ec7", lw=2, label="Mean episode length")
    ax2.axhline(1000, color="#555", lw=1, ls=":", label="Max steps (timeout)")

    ax2.set_ylabel("Mean Episode Length (steps)", color="white")
    ax2.set_xlabel("Training Timesteps (thousands)", color="white")
    ax2.tick_params(colors="#888")
    ax2.spines[:].set_color("#333")
    ax2.yaxis.label.set_color("white")
    ax2.xaxis.label.set_color("white")
    ax2.legend(facecolor="#111", labelcolor="white", fontsize=9)
    ax2.grid(axis="y", color="#222", lw=0.5)

    plt.tight_layout()

    out = Path("assets") / f"learning_curve_{task}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    print(f"\n✓ Learning curve saved → {out}")
    plt.show()

    # Print summary table
    print(f"\nCheckpoint summary:")
    print(f"  {'Steps':>10}  {'Mean Reward':>12}  {'Std':>7}  {'Mean Len':>9}")
    print(f"  {'-'*44}")
    for i in range(len(timesteps)):
        marker = " ◄ best" if i == best_idx else ""
        print(f"  {timesteps[i]:>10,}  {mean_r[i]:>+12.1f}  {std_r[i]:>7.1f}  {mean_l[i]:>9.0f}{marker}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    Path("assets").mkdir(exist_ok=True)

    if args.curve:
        # Update plot path to look in the mode-specific log folder
        plot_learning_curve(f"{args.task}_{args.mode}")
        return

    # Single model evaluation
    if args.model:
        model_path = args.model
    else:
        # UPDATED CANDIDATES TO MATCH TRAIN.PY FOLDER STRUCTURE
        model_dir = Path("models") / f"{args.task}_{args.mode}"
        candidates = [
            model_dir / "best" / "best_model.zip",
            model_dir / f"ppo_{args.task}_{args.mode}_final.zip",
        ]
        
        model_path = next((str(p) for p in candidates if p.exists()), None)
        if model_path is None:
            print(f"No model found in {model_dir}. Train first: python train.py --task {args.task} --mode {args.mode}")
            return

    print(f"Evaluating {args.episodes} episodes (Task: {args.task}, Mode: {args.mode})...")
    # Pass the mode to the environment inside evaluate_model
    stats = evaluate_model(model_path, args.task, args.mode, args.episodes)
    print_report(stats, model_path)


if __name__ == "__main__":
    main()
