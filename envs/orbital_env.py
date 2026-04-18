"""
OrbitalEnv — Gymnasium Environment for Orbital RL
==================================================
Supports two tasks via the `task` constructor argument:

  "docking"         — chaser must rendezvous with the origin (target spacecraft)
  "station_keeping" — chaser must stay inside a defined orbital box

Observation space (normalised, all in [-1, 1] approximately):
  [x_norm, y_norm, vx_norm, vy_norm, fuel_norm]

Action space (continuous):
  [fx, fy] — specific force (acceleration) per axis, clipped to max_thrust
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dynamics import mean_motion, propagate


# ── Constants ────────────────────────────────────────────────────────────────

_TASKS = ("docking", "station_keeping")

# Normalisation scales (match order in obs vector)
_POS_SCALE  = 1_000.0   # metres   → ~[-1,1] for ±1 km initial spread
_VEL_SCALE  = 10.0      # m/s      → ~[-1,1] for ±10 m/s
_FUEL_SCALE = 100.0     # kg       → ~[0, 1]


# ── Environment ───────────────────────────────────────────────────────────────

class OrbitalEnv(gym.Env):
    """
    Parameters
    ----------
    task : str
        "docking" or "station_keeping"
    altitude_km : float
        Orbital altitude above Earth surface (default 400 km — LEO).
    dt : float
        Simulation timestep in seconds (default 1.0 s).
    max_steps : int
        Episode horizon (default 1 000 steps).
    max_thrust : float
        Maximum specific force per axis [m/s^2] (default 0.1).
    initial_fuel : float
        Starting fuel mass [kg] (default 100 kg).
    fuel_per_thrust : float
        Fuel consumed per unit of |thrust| per timestep [kg / (m/s^2 · s)].
    dock_radius : float
        Distance threshold for a successful dock [m] (default 1.0 m).
    dock_speed : float
        Speed threshold for a successful dock [m/s] (default 0.5 m/s).
    station_box : float
        Half-width of the allowed station-keeping box [m] (default 50 m).
    render_mode : str or None
        "human" for live Matplotlib window, None to skip.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "docking",
        altitude_km: float = 400.0,
        dt: float = 1.0,
        max_steps: int = 1_000,
        max_thrust: float = 0.1,
        initial_fuel: float = 100.0,
        fuel_per_thrust: float = 0.05,
        dock_radius: float = 1.0,
        dock_speed: float = 0.5,        # FIX: was 0.1 — too tight for initial training
        station_box: float = 50.0,
        render_mode: str | None = None,
    ):
        super().__init__()

        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got '{task}'")

        self.task          = task
        self.n             = mean_motion(altitude_km)
        self.dt            = dt
        self.max_steps     = max_steps
        self.max_thrust    = max_thrust
        self.initial_fuel  = initial_fuel
        self.fuel_per_unit = fuel_per_thrust
        self.dock_radius   = dock_radius
        self.dock_speed    = dock_speed
        self.station_box   = station_box
        self.render_mode   = render_mode

        # ── Spaces ──────────────────────────────────────────────────────────
        # docking obs: [x, y, vx, vy, fuel]  — 5-dim, all normalised
        # station_keeping obs: [x, y, vx, vy, fuel, vy_correction_norm] — 6-dim
        # vy_correction = -2*n*x is the along-track velocity needed to cancel
        # the secular CW drift from radial offset x. Giving this to the agent
        # as an explicit feature makes the control problem much easier to learn.
        if task == "station_keeping":
            obs_dim = 6
        else:
            obs_dim = 5

        obs_low  = np.array([-5]*4 + [0] + ([-5] if obs_dim == 6 else []), dtype=np.float32)
        obs_high = np.array([ 5]*4 + [1] + ([ 5] if obs_dim == 6 else []), dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # action: [fx, fy] in [-1, 1], scaled internally by max_thrust
        self.action_space = spaces.Box(
            low  = np.full(2, -1.0, dtype=np.float32),
            high = np.full(2,  1.0, dtype=np.float32),
        )

        # ── Rendering state ──────────────────────────────────────────────────
        self._fig = None
        self._trajectory: list[np.ndarray] = []

    # ── Core API ─────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        if self.task == "docking":
            # Start anywhere in ±1 km — agent must navigate from far away
            pos = self.np_random.uniform(-1_000, 1_000, size=2)
            vel = self.np_random.uniform(   -2.0,   2.0, size=2)

        else:  # station_keeping
            # Start INSIDE the box, well away from the edges, with near-zero
            # velocity. This guarantees the agent sees positive reward from
            # step 1, giving it a clear signal to learn from.
            #
            # Why not start at the edges? CW secular drift rate = 6*n*x.
            # At x = station_box (50 m), drift = 0.34 m/s — the agent drifts
            # out in ~150 steps before learning anything. Starting at 0.3×
            # box width gives ~400+ steps of positive reward to learn from.
            inner = self.station_box * 0.3   # ±15 m for default 50 m box
            pos = self.np_random.uniform(-inner, inner, size=2)
            vel = self.np_random.uniform(-0.1, 0.1, size=2)  # near-stationary

        self._state = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float64)
        self._fuel  = float(self.initial_fuel)
        self._steps = 0
        self._trajectory = [self._state[:2].copy()]

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        force  = action * self.max_thrust                   # [m/s^2]

        # Fuel cost proportional to |thrust|
        thrust_magnitude  = float(np.linalg.norm(force))
        fuel_used         = self.fuel_per_unit * thrust_magnitude * self.dt
        self._fuel        = max(0.0, self._fuel - fuel_used)

        # Zero force if out of fuel
        if self._fuel <= 0.0:
            force = np.zeros(2)

        # Propagate physics
        self._state = propagate(self._state, self.n, force, self.dt)
        self._steps += 1
        self._trajectory.append(self._state[:2].copy())

        obs      = self._get_obs()
        reward   = self._compute_reward(force)
        done, truncated, info = self._check_terminal()

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, truncated, info

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        x, y, vx, vy = self._state
        obs = [
            x  / _POS_SCALE,
            y  / _POS_SCALE,
            vx / _VEL_SCALE,
            vy / _VEL_SCALE,
            self._fuel / _FUEL_SCALE,
        ]
        if self.task == "station_keeping":
            # CW drift cancellation hint: the along-track velocity the agent
            # should target to cancel secular drift from radial offset x.
            # vy_correction = -2*n*x (from CW periodic orbit condition)
            vy_correction = -2.0 * self.n * x
            obs.append(vy_correction / _VEL_SCALE)
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, force: np.ndarray) -> float:
        x, y, vx, vy = self._state
        dist  = np.sqrt(x**2 + y**2)
        speed = np.sqrt(vx**2 + vy**2)

        if self.task == "docking":
            # 1. Distance penalty — always drives agent closer
            reward = -dist / _POS_SCALE

            # 2. Proximity bonus: 1/(d + eps) gives a pole at the origin.
            #    Continuous and monotone at ALL distances — no transition
            #    discontinuity. The gradient doubles every time distance halves,
            #    so the agent is always pulled forward, never satisfied hovering.
            reward += 0.5 / (dist + 0.5)

            # 3. Speed bonus inside 50 m: reward slow approach, additive only.
            #    max(0, ...) ensures this never penalises the agent.
            if dist < 50.0:
                reward += 0.3 * max(0.0, 1.0 - speed / self.dock_speed)

            # 4. Terminal dock bonus: large one-time reward on success.
            #    This makes docking worth more than any amount of hovering.
            #    Checked here so it appears in the step reward, not just info.
            if dist < self.dock_radius and speed < self.dock_speed:
                reward += 500.0

            # 5. Small fuel penalty — discourages wasteful burns
            reward -= 0.005 * float(np.linalg.norm(force)) / self.max_thrust

        else:  # station_keeping
            half = self.station_box
            in_box = (abs(x) < half) and (abs(y) < half)

            if in_box:
                # Reward for being inside — bonus for being near centre
                centre_dist = np.sqrt(x**2 + y**2)
                reward = 1.0 + 0.5 * (1.0 - centre_dist / half)
            else:
                # Shaped penalty: closer to the box = less negative
                # Distance to nearest box edge (not centre)
                dx = max(0.0, abs(x) - half)
                dy = max(0.0, abs(y) - half)
                edge_dist = np.sqrt(dx**2 + dy**2)
                reward = -1.0 - edge_dist / (half * 2.0)

            # Fuel penalty
            reward -= 0.005 * float(np.linalg.norm(force)) / self.max_thrust

        return float(reward)

    def _check_terminal(self):
        x, y, vx, vy = self._state
        dist  = np.sqrt(x**2 + y**2)
        speed = np.sqrt(vx**2 + vy**2)
        info  = {"distance": dist, "speed": speed, "fuel": self._fuel, "steps": self._steps}

        # Successful dock
        if self.task == "docking" and dist < self.dock_radius and speed < self.dock_speed:
            info["outcome"] = "docked"
            return True, False, info

        # Crash (high-speed collision within dock radius)
        if self.task == "docking" and dist < self.dock_radius and speed >= self.dock_speed:
            info["outcome"] = "crash"
            return True, False, info

        # Runaway — too far from target (task-specific thresholds)
        # station_keeping uses 4× box: tight enough to terminate fast,
        # generous enough that the agent can recover from small excursions
        runaway_limit = 5_000.0 if self.task == "docking" else self.station_box * 4.0
        if dist > runaway_limit:
            info["outcome"] = "runaway"
            return True, False, info

        # Timeout
        if self._steps >= self.max_steps:
            info["outcome"] = "timeout"
            return False, True, info

        return False, False, info

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self):
        import matplotlib.pyplot as plt

        traj = np.array(self._trajectory)

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))

        ax = self._ax
        ax.clear()

        ax.plot(traj[:, 1], traj[:, 0], "c-", lw=0.8, alpha=0.6, label="Trajectory")
        ax.plot(traj[-1, 1], traj[-1, 0], "o", color="cyan", ms=6, label="Chaser")
        ax.plot(0, 0, "x", color="gold", ms=10, mew=2, label="Target")

        if self.task == "station_keeping":
            b = self.station_box
            rect = plt.Rectangle((-b, -b), 2*b, 2*b,
                                  linewidth=1, edgecolor="lime",
                                  facecolor="lime", alpha=0.05)
            ax.add_patch(rect)

        ax.set_xlim(-1_200, 1_200)
        ax.set_ylim(-1_200, 1_200)
        ax.set_xlabel("Along-track y [m]")
        ax.set_ylabel("Radial x [m]")
        ax.set_title(f"Task: {self.task} | Step {self._steps} | Fuel: {self._fuel:.1f} kg")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_facecolor("#0a0a12")
        self._fig.patch.set_facecolor("#0a0a12")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        plt.pause(0.001)
        plt.draw()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
