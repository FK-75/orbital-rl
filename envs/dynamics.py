"""
Clohessy-Wiltshire (CW) Equations — Relative Orbital Dynamics
=============================================================
Describes the motion of a chaser spacecraft relative to a target
in a circular reference orbit (Hill's Frame / LVLH frame).

Full 3D state vector: [x, y, z, vx, vy, vz]
  x  — radial       (away from Earth = positive)
  y  — along-track  (direction of orbital motion)
  z  — cross-track  (out of orbital plane)
  vx, vy, vz — corresponding velocities

In-plane equations (coupled):
  ẍ =  3n²x + 2nẏ + fx
  ÿ = -2nẋ       + fy

Out-of-plane equation (decoupled — simpler harmonic oscillator):
  z̈ = -n²z + fz

The z-axis is independent of x and y — this is the key property
that makes 3D CW tractable. An agent controlling z does not need
to worry about the x-y dynamics and vice versa.

Reference: Clohessy, W.H. & Wiltshire, R.S. (1960)
"""

import numpy as np


def mean_motion(altitude_km: float = 400.0) -> float:
    """
    Compute mean motion n [rad/s] for a circular orbit.

    n = sqrt(mu / a^3)
    mu = 3.986e14 m^3/s^2  (Earth gravitational parameter)
    a  = R_earth + altitude
    """
    mu = 3.986004418e14
    R_earth = 6_371_000.0
    a = R_earth + altitude_km * 1_000.0
    return np.sqrt(mu / a**3)


def cw_derivatives(state: np.ndarray, n: float, force: np.ndarray) -> np.ndarray:
    """
    Compute state derivatives from the full 3D CW equations.

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz]
    n : float
        Mean motion [rad/s]
    force : np.ndarray, shape (3,)
        Specific force [m/s^2] in [fx, fy, fz]

    Returns
    -------
    np.ndarray, shape (6,)
        [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state
    fx, fy, fz = force

    # In-plane (coupled via Coriolis)
    ax = 3 * n**2 * x + 2 * n * vy + fx
    ay = -2 * n * vx + fy

    # Out-of-plane (decoupled harmonic oscillator)
    az = -n**2 * z + fz

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)


def rk4_step(state: np.ndarray, n: float, force: np.ndarray, dt: float) -> np.ndarray:
    """Fourth-order Runge-Kutta integration step (3D)."""
    k1 = cw_derivatives(state,             n, force)
    k2 = cw_derivatives(state + dt/2 * k1, n, force)
    k3 = cw_derivatives(state + dt/2 * k2, n, force)
    k4 = cw_derivatives(state + dt   * k3, n, force)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def propagate(
    state: np.ndarray,
    n: float,
    force: np.ndarray,
    dt: float,
    substeps: int = 10,
) -> np.ndarray:
    """
    Propagate state over one environment step using sub-stepped RK4.

    Works for both 2D (state shape (4,), force shape (2,)) and
    3D (state shape (6,), force shape (3,)) via zero-padding.
    Always operates in 3D internally for consistency.
    """
    # Pad 2D inputs to 3D if needed (backward-compatible)
    if state.shape[0] == 4:
        state = np.array([state[0], state[1], 0.0, state[2], state[3], 0.0])
        force = np.array([force[0], force[1], 0.0])
        was_2d = True
    else:
        was_2d = False

    sub_dt = dt / substeps
    s = state.copy()
    for _ in range(substeps):
        s = rk4_step(s, n, force, sub_dt)

    if was_2d:
        return np.array([s[0], s[1], s[3], s[4]])
    return s
