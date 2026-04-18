"""
Clohessy-Wiltshire (CW) Equations — Relative Orbital Dynamics
=============================================================
Describes the motion of a chaser spacecraft relative to a target
in a circular reference orbit (Hill's Frame / LVLH frame).

State vector: [x, y, vx, vy]
  x  — radial (towards Earth = negative)
  y  — along-track (direction of orbital motion)
  vx — radial velocity
  vy — along-track velocity

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
    mu = 3.986004418e14       # m^3 / s^2
    R_earth = 6_371_000.0     # m
    a = R_earth + altitude_km * 1_000.0
    return np.sqrt(mu / a**3)


def cw_derivatives(state: np.ndarray, n: float, force: np.ndarray) -> np.ndarray:
    """
    Compute state derivatives from CW equations.

    Parameters
    ----------
    state : np.ndarray, shape (4,)
        [x, y, vx, vy]
    n : float
        Mean motion [rad/s]
    force : np.ndarray, shape (2,)
        Specific force (acceleration) [m/s^2] in [fx, fy]

    Returns
    -------
    np.ndarray, shape (4,)
        [vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    fx, fy = force

    ax = 3 * n**2 * x + 2 * n * vy + fx
    ay = -2 * n * vx + fy

    return np.array([vx, vy, ax, ay], dtype=np.float64)


def rk4_step(state: np.ndarray, n: float, force: np.ndarray, dt: float) -> np.ndarray:
    """
    Fourth-order Runge-Kutta integration step.

    Parameters
    ----------
    state : np.ndarray, shape (4,)
    n     : float  — mean motion
    force : np.ndarray, shape (2,) — constant force over dt
    dt    : float  — timestep [s]

    Returns
    -------
    np.ndarray, shape (4,) — next state
    """
    k1 = cw_derivatives(state,            n, force)
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

    Splitting dt into `substeps` reduces integration error for
    fast dynamics (high thrust or large n).
    """
    sub_dt = dt / substeps
    s = state.copy()
    for _ in range(substeps):
        s = rk4_step(s, n, force, sub_dt)
    return s
