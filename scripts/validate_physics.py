"""
scripts/validate_physics.py — Sanity-check the CW propagator
=============================================================
Run this BEFORE training. If the plots don't look like natural
orbital trajectories, fix the physics before touching RL.

Three tests:
  1. Free drift       — no thrust, should produce a closed ellipse-like path
  2. Radial impulse   — single x-burst produces a drifting figure-8
  3. Station-keeping  — along-track offset stays bounded

Usage:
    python scripts/validate_physics.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from envs.dynamics import mean_motion, propagate


def simulate(
    state0: np.ndarray,
    n: float,
    force_fn,
    dt: float = 1.0,
    steps: int = 5_000,
) -> np.ndarray:
    """Run a trajectory and return (steps, 4) array of states."""
    states = [state0.copy()]
    s = state0.copy()
    for i in range(steps):
        force = force_fn(i, s)
        s = propagate(s, n, force, dt)
        states.append(s.copy())
    return np.array(states)


def main():
    n = mean_motion(400.0)
    print(f"Mean motion n = {n:.6e} rad/s  (T_orbit ≈ {2*np.pi/n/60:.1f} min)")

    fig = plt.figure(figsize=(15, 5), facecolor="#0a0a12")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ─── Test 1: Free drift from radial offset ─────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    s0 = np.array([500.0, 0.0, 0.0, 0.0])        # 500 m radial offset
    traj1 = simulate(s0, n, lambda i, s: np.zeros(2))

    ax1.plot(traj1[:, 1], traj1[:, 0], color="#00d4ff", lw=1.0)
    ax1.plot(traj1[0, 1], traj1[0, 0], "o", color="lime",  ms=7, label="Start")
    ax1.plot(0, 0, "x",                     color="gold",  ms=10, mew=2, label="Target")
    _style_ax(ax1, "Test 1: Free Drift (no thrust)\nShould trace a closed ellipse")

    # ─── Test 2: Radial impulse ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    def radial_impulse(i, s):
        return np.array([0.05, 0.0]) if i == 0 else np.zeros(2)

    s0 = np.array([200.0, 200.0, 0.0, 0.0])
    traj2 = simulate(s0, n, radial_impulse, steps=8_000)

    ax2.plot(traj2[:, 1], traj2[:, 0], color="#ff6ec7", lw=0.8)
    ax2.plot(traj2[0, 1], traj2[0, 0], "o", color="lime", ms=7, label="Start")
    ax2.plot(0, 0, "x",                     color="gold", ms=10, mew=2, label="Target")
    _style_ax(ax2, "Test 2: Single Radial Impulse\nShould drift along-track over time")

    # ─── Test 3: Constant along-track thrust ──────────────────────────────
    ax3 = fig.add_subplot(gs[2])

    s0 = np.array([0.0, -800.0, 0.0, 0.0])
    traj3 = simulate(s0, n, lambda i, s: np.array([0.0, 0.002]), steps=3_000)

    ax3.plot(traj3[:, 1], traj3[:, 0], color="#ffd700", lw=1.0)
    ax3.plot(traj3[0, 1], traj3[0, 0], "o", color="lime", ms=7, label="Start")
    ax3.plot(0, 0, "x",                     color="gold", ms=10, mew=2, label="Target")
    _style_ax(ax3, "Test 3: Constant Along-Track Thrust\nShould spiral inward toward target")

    # ─── Summary ──────────────────────────────────────────────────────────
    plt.suptitle("CW Propagator Validation", color="white", fontsize=13, y=1.02)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "physics_validation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0a0a12")
    print(f"\n✓ Saved → {out}")
    plt.show()

    # ─── Numerical Validation ────────────────────────────────────────────
    T = int(2 * np.pi / n)
    all_pass = True

    # Check 1: Truly periodic orbit (correct ICs: vy0 = -2*n*x0)
    # These initial conditions eliminate the secular y-drift in CW equations
    # From analytical CW y(t): secular term vanishes when -6*n*x0 - 3*vy0 = 0 → vy0 = -2*n*x0
    x0 = 100.0
    vy0 = -2.0 * n * x0  # drift-free condition
    s0_periodic = np.array([x0, 0.0, 0.0, vy0])
    sfin = simulate(s0_periodic, n, lambda i, s: np.zeros(2), dt=1.0, steps=T)[-1]
    err = np.linalg.norm(sfin[:2] - s0_periodic[:2])
    print(f"\nCheck 1: Periodic orbit  |  position error after 1 orbit = {err:.3f} m")
    if err < 5.0:
        print("  ✓ PASS — closed orbit returns to start")
    else:
        print("  ✗ FAIL — periodic orbit did not close")
        all_pass = False

    # Check 2: Secular drift matches analytical CW prediction
    # For ICs [x0, 0, 0, 0]: x returns to x0, but y drifts by -6*x0*2π
    s0_drift = np.array([x0, 0.0, 0.0, 0.0])
    sfin2 = simulate(s0_drift, n, lambda i, s: np.zeros(2), dt=1.0, steps=T)[-1]
    x_err = abs(sfin2[0] - x0)
    y_analytical = -6.0 * x0 * 2.0 * np.pi   # expected secular drift
    y_err = abs(sfin2[1] - y_analytical)
    print(f"\nCheck 2: Secular drift   |  x error = {x_err:.3f} m  |  y drift error vs analytical = {y_err:.3f} m")
    if x_err < 2.0 and y_err < 50.0:
        print(f"  ✓ PASS — x returns to x0, y drifts by {sfin2[1]:.1f} m (analytical: {y_analytical:.1f} m)")
    else:
        print("  ✗ FAIL — drift does not match CW analytical solution")
        all_pass = False

    # Summary
    if all_pass:
        print("\n══════════════════════════════════════════════")
        print("  ✓ ALL CHECKS PASSED — physics is correct, safe to train.")
        print("══════════════════════════════════════════════\n")
    else:
        print("\n  ✗ SOME CHECKS FAILED — review propagator before training.\n")


def _style_ax(ax, title):
    ax.set_facecolor("#0a0a12")
    ax.tick_params(colors="#555")
    ax.set_xlabel("Along-track y [m]", color="#888", fontsize=8)
    ax.set_ylabel("Radial x [m]",      color="#888", fontsize=8)
    ax.set_title(title, color="white", fontsize=8.5)
    ax.legend(fontsize=7, facecolor="#111", labelcolor="white")
    ax.spines[:].set_color("#222")


if __name__ == "__main__":
    main()
