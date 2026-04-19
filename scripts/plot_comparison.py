"""
scripts/plot_comparison.py — Side-by-side learning curve for both tasks
========================================================================
Generates a single figure showing docking and station-keeping training
curves on the same x-axis (timesteps) with independent y-axes.

The contrasting convergence patterns tell the whole story:
  - Docking:         slow, noisy, ~3M steps to converge
  - Station-keeping: fast, clean cliff at 300k steps

Usage:
    python scripts/plot_comparison.py
    python scripts/plot_comparison.py --out assets/comparison.png
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_eval(task: str, mode: str):
    # Updated path to match your folder structure: task_mode
    path = Path("logs") / f"{task}_{mode}" / "evaluations.npz"
    if not path.exists():
        print(f"  [!] Missing: {path}")
        return None
    
    d = np.load(path)
    ts      = d["timesteps"]
    results = d["results"]
    lengths = d["ep_lengths"]
    return {
        "timesteps":   ts,
        "mean_reward": results.mean(axis=1),
        "std_reward":  results.std(axis=1),
        "mean_length": lengths.mean(axis=1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="2d", choices=["2d", "3d"], 
                   help="Compare 2d or 3d results")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # Default output name includes the mode
    out_path = args.out or f"assets/comparison_{args.mode}.png"

    print(f"Loading {args.mode} logs...")
    dock = load_eval("docking", args.mode)
    sk   = load_eval("station_keeping", args.mode)

    if dock is None and sk is None:
        print(f"\nError: No evaluations.npz found for mode '{args.mode}'.")
        print(f"Check your logs/ folder. Expected subfolders: docking_{args.mode} and station_keeping_{args.mode}")
        return

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor="#0d0d1a")
    fig.suptitle("Training Comparison — Docking vs Station-Keeping",
                 color="white", fontsize=14, y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_dock_r  = fig.add_subplot(gs[0, 0])   # docking reward
    ax_dock_l  = fig.add_subplot(gs[1, 0])   # docking ep length
    ax_sk_r    = fig.add_subplot(gs[0, 1])   # station-keeping reward
    ax_sk_l    = fig.add_subplot(gs[1, 1])   # station-keeping ep length

    DOCK_COL = "#00d4ff"
    SK_COL   = "#ff6ec7"
    BG       = "#0d0d1a"

    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor(BG)
        ax.spines[:].set_color("#333")
        ax.tick_params(colors="#888", labelsize=8)
        ax.set_title(title,  color="white",  fontsize=9)
        ax.set_xlabel(xlabel, color="#888", fontsize=8)
        ax.set_ylabel(ylabel, color="#888", fontsize=8)
        ax.grid(axis="y", color="#222", lw=0.5)

    # ── Docking reward ────────────────────────────────────────────────────────
    if dock:
        ts = dock["timesteps"] / 1_000
        mr = dock["mean_reward"]
        sr = dock["std_reward"]
        best_i = int(np.argmax(mr))

        ax_dock_r.fill_between(ts, mr - sr, mr + sr,
                               alpha=0.18, color=DOCK_COL)
        ax_dock_r.plot(ts, mr, color=DOCK_COL, lw=2)
        ax_dock_r.axhline(0, color="#444", lw=0.8, ls="--")
        ax_dock_r.axvline(ts[best_i], color="#ffd700", lw=1, ls="--", alpha=0.7)
        ax_dock_r.plot(ts[best_i], mr[best_i], "o", color="#ffd700", ms=7,
                       zorder=5, label=f"Best: {mr[best_i]:.0f} @ {dock['timesteps'][best_i]/1e6:.2f}M")
        ax_dock_r.legend(facecolor="#111", labelcolor="white", fontsize=7)
        _style(ax_dock_r, "Docking — Episode Reward",
               "Timesteps (k)", "Mean Reward")

        ax_dock_l.fill_between(ts,
                               dock["mean_length"] - dock["std_reward"]*0,  # just mean line
                               dock["mean_length"],
                               alpha=0.15, color=DOCK_COL)
        ax_dock_l.plot(ts, dock["mean_length"], color=DOCK_COL, lw=2)
        ax_dock_l.axhline(1000, color="#555", lw=1, ls=":", label="Max steps")
        ax_dock_l.legend(facecolor="#111", labelcolor="white", fontsize=7)
        _style(ax_dock_l, "Docking — Episode Length",
               "Timesteps (k)", "Mean Steps")
        # Annotate: shortening episodes = faster docking
        ax_dock_l.annotate("Shorter = faster dock",
                           xy=(ts[-1], dock["mean_length"][-1]),
                           xytext=(ts[len(ts)//2], 600),
                           color="#888", fontsize=7,
                           arrowprops=dict(arrowstyle="->", color="#555"))

    # ── Station-keeping reward ────────────────────────────────────────────────
    if sk:
        ts = sk["timesteps"] / 1_000
        mr = sk["mean_reward"]
        sr = sk["std_reward"]
        best_i = int(np.argmax(mr))

        ax_sk_r.fill_between(ts, mr - sr, mr + sr,
                             alpha=0.18, color=SK_COL)
        ax_sk_r.plot(ts, mr, color=SK_COL, lw=2)
        ax_sk_r.axhline(0,    color="#444", lw=0.8, ls="--")
        ax_sk_r.axhline(1500, color="#ffd700", lw=0.8, ls="--",
                        alpha=0.5, label="Theoretical max (+1500)")
        ax_sk_r.axvline(ts[best_i], color="#ffd700", lw=1, ls="--", alpha=0.7)
        ax_sk_r.plot(ts[best_i], mr[best_i], "o", color="#ffd700", ms=7,
                     zorder=5, label=f"Best: {mr[best_i]:.0f} @ {sk['timesteps'][best_i]/1e6:.2f}M")

        # Annotate the cliff
        cliff_i = next((i for i, r in enumerate(mr) if r > 0), None)
        if cliff_i is not None:
            ax_sk_r.annotate("Agent discovers\ndrift correction",
                            xy=(ts[cliff_i], mr[cliff_i]),
                            xytext=(ts[cliff_i] + ts[-1]*0.15, mr[cliff_i] - 400),
                            color="#aaa", fontsize=7,
                            arrowprops=dict(arrowstyle="->", color="#555"))

        ax_sk_r.legend(facecolor="#111", labelcolor="white", fontsize=7)
        _style(ax_sk_r, "Station-Keeping — Episode Reward",
               "Timesteps (k)", "Mean Reward")

        ax_sk_l.fill_between(ts,
                             sk["mean_length"] * 0,
                             sk["mean_length"],
                             alpha=0.15, color=SK_COL)
        ax_sk_l.plot(ts, sk["mean_length"], color=SK_COL, lw=2)
        ax_sk_l.axhline(1000, color="#555", lw=1, ls=":", label="Max steps (success)")
        ax_sk_l.legend(facecolor="#111", labelcolor="white", fontsize=7)
        _style(ax_sk_l, "Station-Keeping — Episode Length",
               "Timesteps (k)", "Mean Steps")
        ax_sk_l.annotate("1000 steps = stays in box\nfull episode",
                         xy=(ts[-1], sk["mean_length"][-1]),
                         xytext=(ts[len(ts)//3], 600),
                         color="#888", fontsize=7,
                         arrowprops=dict(arrowstyle="->", color="#555"))

    # ── Key insight text ──────────────────────────────────────────────────────
    insight = (
        "Key insight: Station-keeping converges 3× faster than docking.\n"
        "The sharp cliff (↑) marks when the agent discovered the CW drift correction.\n"
        "Docking requires multi-scale behaviour (approach + precision) — harder to learn."
    )
    fig.text(0.5, 0.01, insight, ha="center", va="bottom",
             color="#666", fontsize=8, style="italic")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n✓ Comparison plot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
