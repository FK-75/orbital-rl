"""
enjoy.py — Watch a trained PPO agent fly
=========================================
Usage:
    python enjoy.py --task docking
    python enjoy.py --task station_keeping
    python enjoy.py --task docking --episodes 5
    python enjoy.py --task docking --save_gif
    python enjoy.py --task docking --model models/docking/best/best_model
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs.orbital_env import OrbitalEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",     default="docking",
                   choices=["docking", "station_keeping"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--model",    default=None,
                   help="Path to .zip model (auto-detects best model if omitted)")
    p.add_argument("--save_gif", action="store_true",
                   help="Save each episode as a GIF to assets/")
    return p.parse_args()


def find_model(task: str) -> str:
    candidates = [
        Path("models") / task / "best" / "best_model.zip",
        Path("models") / task / f"ppo_{task}_final.zip",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"No trained model found for task '{task}'.\n"
        f"Run: python train.py --task {task}"
    )


def _render_frame(env: OrbitalEnv) -> np.ndarray:
    """Render current env state to an RGB numpy array."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
    traj = np.array(env._trajectory)

    ax.set_facecolor("#0a0a12")
    fig.patch.set_facecolor("#0a0a12")

    ax.plot(traj[:, 1], traj[:, 0], "-", color="#00d4ff", lw=0.7, alpha=0.5)
    ax.plot(traj[-1, 1], traj[-1, 0], "o", color="#00d4ff", ms=8)
    ax.plot(0, 0, "x", color="#ffd700", ms=12, mew=2.5)

    if env.task == "station_keeping":
        b = env.station_box
        rect = plt.Rectangle((-b, -b), 2*b, 2*b,
                              linewidth=1, edgecolor="lime",
                              facecolor="lime", alpha=0.08)
        ax.add_patch(rect)

    ax.set_xlim(-1_200, 1_200)
    ax.set_ylim(-1_200, 1_200)
    ax.set_xlabel("Along-track [m]", color="white", fontsize=8)
    ax.set_ylabel("Radial [m]",      color="white", fontsize=8)
    ax.tick_params(colors="#555")
    ax.spines[:].set_color("#222")

    dist = float(np.linalg.norm(env._state[:2]))
    ax.set_title(
        f"Step {env._steps}  |  dist={dist:.1f} m  |  fuel={env._fuel:.1f} kg",
        color="white", fontsize=8
    )
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = img[:, :, :3]  # drop alpha → RGB
    plt.close(fig)
    return img


def save_gif(frames: list, path: str, fps: int = 20):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"  ✓ GIF saved → {path}")


def run_episode(env, model, save_gif_flag: bool):
    obs, _ = env.reset()
    done = truncated = False
    total_reward = 0.0
    frames = []

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if save_gif_flag:
            frames.append(_render_frame(env))

    return total_reward, info, frames


def main():
    args  = parse_args()
    mpath = args.model or find_model(args.task)

    print(f"\nLoading: {mpath}")
    model = PPO.load(mpath)
    env   = OrbitalEnv(task=args.task, render_mode="human")
    Path("assets").mkdir(exist_ok=True)

    outcomes = {"docked": 0, "crash": 0, "runaway": 0, "timeout": 0}

    for ep in range(1, args.episodes + 1):
        reward, info, frames = run_episode(env, model, args.save_gif)
        outcome = info.get("outcome", "timeout")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

        print(f"  Episode {ep:2d} | reward={reward:+8.1f} | "
              f"outcome={outcome:10s} | dist={info['distance']:7.2f} m | "
              f"fuel={info['fuel']:.1f} kg")

        if args.save_gif and frames:
            save_gif(frames, f"assets/{args.task}_ep{ep}.gif")

    env.close()

    print(f"\nSummary over {args.episodes} episodes:")
    for k, v in outcomes.items():
        if v > 0:
            print(f"  {k:10s}: {v}")
    print()


if __name__ == "__main__":
    main()
