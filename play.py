"""
play.py — Human-playable spacecraft docking
============================================
Try to dock manually using your keyboard. This shows exactly how
hard the task is — and how impressive it is that the AI solves it.

Controls:
  Arrow keys  — radial (↑↓) and along-track (←→) thrust
  W / S       — out-of-plane thrust (3D mode only)
  SPACE       — zero throttle (coast)
  R           — reset episode
  Q / Escape  — quit

Usage:
    python play.py                    # 2D docking
    python play.py --mode 3d          # 3D docking
    python play.py --task station_keeping

Tips:
  - The target (gold ✕) is at the origin.
  - You need to arrive within 1 m at under 0.5 m/s.
  - CW drift means doing nothing is NOT stable — you will drift away.
  - Use short bursts, not sustained thrust.
"""

import argparse
import sys
import time

import matplotlib
matplotlib.use("TkAgg")          # needs an interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from envs.orbital_env import OrbitalEnv


# ── Key state ────────────────────────────────────────────────────────────────

_KEYS = {
    "up": False, "down": False, "left": False, "right": False,
    "w": False,  "s":    False,
    "reset": False, "quit": False,
}

def _on_key_press(event):
    if event.key == "up":        _KEYS["up"]    = True
    elif event.key == "down":    _KEYS["down"]  = True
    elif event.key == "left":    _KEYS["left"]  = True
    elif event.key == "right":   _KEYS["right"] = True
    elif event.key == "w":       _KEYS["w"]     = True
    elif event.key == "s":       _KEYS["s"]     = True
    elif event.key == "r":       _KEYS["reset"] = True
    elif event.key in ("q", "escape"): _KEYS["quit"] = True

def _on_key_release(event):
    if event.key == "up":        _KEYS["up"]    = False
    elif event.key == "down":    _KEYS["down"]  = False
    elif event.key == "left":    _KEYS["left"]  = False
    elif event.key == "right":   _KEYS["right"] = False
    elif event.key == "w":       _KEYS["w"]     = False
    elif event.key == "s":       _KEYS["s"]     = False


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw(ax, env, score: float, best: float, outcome: str | None,
          controls_ax, is_3d: bool):
    traj = np.array(env._trajectory)
    pos, vel = env._get_pos_vel()
    dist  = float(np.linalg.norm(pos))
    speed = float(np.linalg.norm(vel))

    ax.clear()
    ax.set_facecolor("#0a0a12")

    # Trajectory
    ax.plot(traj[:, 1], traj[:, 0], "-", color="#00d4ff", lw=0.8, alpha=0.5)
    # Chaser
    ax.plot(traj[-1, 1], traj[-1, 0], "o", color="#00d4ff", ms=9, zorder=5)
    # Target
    ax.plot(0, 0, "x", color="#ffd700", ms=14, mew=3, zorder=5)

    if env.task == "station_keeping":
        b = env.station_box
        rect = mpatches.Rectangle((-b, -b), 2*b, 2*b, lw=1.5,
                                   edgecolor="lime", facecolor="lime", alpha=0.06)
        ax.add_patch(rect)

    # Dock radius circle
    circle = plt.Circle((0, 0), env.dock_radius, color="gold",
                         fill=False, lw=1, ls="--", alpha=0.5)
    ax.add_patch(circle)

    lim = 1_200
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Along-track y [m]", color="#888", fontsize=9)
    ax.set_ylabel("Radial x [m]",      color="#888", fontsize=9)
    ax.tick_params(colors="#555")
    ax.spines[:].set_color("#222")

    # Status colour
    if dist < 50:
        col = "#00ff88"
    elif dist < 200:
        col = "#ffd700"
    else:
        col = "#ff6060"

    outcome_str = f"  [{outcome.upper()}]" if outcome else ""
    title = (f"dist={dist:.1f} m   speed={speed:.3f} m/s   "
             f"fuel={env._fuel:.1f} kg   step={env._steps}{outcome_str}")
    ax.set_title(title, color=col, fontsize=9)

    # ── Controls panel ───────────────────────────────────────────────────────
    controls_ax.clear()
    controls_ax.set_facecolor("#0a0a12")
    controls_ax.axis("off")

    lines = [
        ("Score", f"{score:+.1f}"),
        ("Best",  f"{best:+.1f}"),
        ("",      ""),
        ("↑ ↓",   "Radial thrust"),
        ("← →",   "Along-track thrust"),
    ]
    if is_3d:
        lines.append(("W S", "Cross-track thrust"))
    lines += [("R", "Reset"), ("Q / Esc", "Quit")]

    for i, (key, desc) in enumerate(lines):
        controls_ax.text(0.05, 0.95 - i * 0.11, key,
                         color="#ffd700", fontsize=9,
                         transform=controls_ax.transAxes, va="top",
                         fontfamily="monospace")
        controls_ax.text(0.45, 0.95 - i * 0.11, desc,
                         color="#aaa", fontsize=9,
                         transform=controls_ax.transAxes, va="top")

    # Active thrust indicator
    thrust_y = 0.95 - len(lines) * 0.11 - 0.05
    active = []
    if _KEYS["up"]:    active.append("▲ radial+")
    if _KEYS["down"]:  active.append("▼ radial−")
    if _KEYS["left"]:  active.append("◄ track−")
    if _KEYS["right"]: active.append("► track+")
    if _KEYS["w"]:     active.append("W z+")
    if _KEYS["s"]:     active.append("S z−")
    thrust_str = "  ".join(active) if active else "coasting"
    controls_ax.text(0.05, thrust_y, thrust_str,
                     color="#00ff88" if active else "#555",
                     fontsize=8, transform=controls_ax.transAxes, va="top")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="docking",
                   choices=["docking", "station_keeping"])
    p.add_argument("--mode", default="2d", choices=["2d", "3d"])
    args = p.parse_args()
    is_3d = args.mode == "3d"

    env = OrbitalEnv(task=args.task, mode=args.mode)
    obs, _ = env.reset()

    fig = plt.figure(figsize=(11, 6), facecolor="#0a0a12")
    fig.canvas.manager.set_window_title("Orbital RL — Human Play")
    ax          = fig.add_axes([0.05, 0.08, 0.60, 0.86])
    controls_ax = fig.add_axes([0.68, 0.08, 0.28, 0.86])

    fig.canvas.mpl_connect("key_press_event",   _on_key_press)
    fig.canvas.mpl_connect("key_release_event", _on_key_release)

    score   = 0.0
    best    = -float("inf")
    outcome = None

    print(f"\n🛰️  Orbital RL — Human Play  ({args.mode.upper()} {args.task})")
    print("   Dock the chaser (●) with the target (✕)")
    print("   Use arrow keys to thrust. R to reset. Q to quit.\n")

    dt_wall = 1 / 20   # 20 fps display

    while not _KEYS["quit"]:
        # ── Build action from keys ────────────────────────────────────────
        fx = fy = fz = 0.0
        if _KEYS["up"]:    fx =  1.0
        if _KEYS["down"]:  fx = -1.0
        if _KEYS["right"]: fy =  1.0
        if _KEYS["left"]:  fy = -1.0
        if _KEYS["w"]:     fz =  1.0
        if _KEYS["s"]:     fz = -1.0

        if is_3d:
            action = np.array([fx, fy, fz], dtype=np.float32)
        else:
            action = np.array([fx, fy], dtype=np.float32)

        # ── Reset ─────────────────────────────────────────────────────────
        if _KEYS["reset"]:
            _KEYS["reset"] = False
            obs, _ = env.reset()
            score   = 0.0
            outcome = None
            print("  ↺ Reset")

        # ── Step ──────────────────────────────────────────────────────────
        if outcome is None:
            obs, reward, done, truncated, info = env.step(action)
            score += reward

            if done or truncated:
                outcome = info.get("outcome", "timeout")
                best    = max(best, score)
                dist    = info["distance"]
                spd     = info["speed"]
                print(f"  {outcome.upper():10s} | score={score:+.1f} | "
                      f"dist={dist:.2f} m | speed={spd:.3f} m/s | "
                      f"fuel={info['fuel']:.1f} kg")

        # ── Draw ──────────────────────────────────────────────────────────
        _draw(ax, env, score, best, outcome, controls_ax, is_3d)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(dt_wall)

    env.close()
    plt.close("all")
    print(f"\n  Final best score: {best:+.1f}")
    print("  Quit.\n")


if __name__ == "__main__":
    main()
