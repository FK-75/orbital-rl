"""
OrbitalEnv — Gymnasium Environment for Orbital RL
==================================================
Supports two tasks and two dimensionalities:

  task="docking"         — navigate to dock with target at origin
  task="station_keeping" — hold position inside a defined orbital box

  mode="2d"  — in-plane only (x, y) — faster training
  mode="3d"  — full 3D including cross-track z-axis

3D adds the out-of-plane CW equation:  z̈ = -n²z + fz
This axis is DECOUPLED from x/y — it behaves as a simple harmonic
oscillator. The agent gets an extra observation (z, vz) and an extra
thrust axis (fz).

Observation spaces:
  2D docking         : [x, y, vx, vy, fuel]              — 5-dim
  3D docking         : [x, y, z, vx, vy, vz, fuel]       — 7-dim
  2D station_keeping : [x, y, vx, vy, fuel, vy_corr]     — 6-dim
  3D station_keeping : [x, y, z, vx, vy, vz, fuel, vy_corr, vz_corr] — 9-dim

Action spaces:
  2D : [fx, fy]       — 2-dim
  3D : [fx, fy, fz]   — 3-dim
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dynamics import mean_motion, propagate


_TASKS = ("docking", "station_keeping")
_MODES = ("2d", "3d")

_POS_SCALE  = 1_000.0
_VEL_SCALE  = 10.0
_FUEL_SCALE = 100.0


class OrbitalEnv(gym.Env):
    """
    Parameters
    ----------
    task : str
        "docking" or "station_keeping"
    mode : str
        "2d" (default) or "3d" — whether to simulate the cross-track axis
    altitude_km : float
        Orbital altitude [km] (default 400 = LEO)
    dt : float
        Simulation timestep [s] (default 1.0)
    max_steps : int
        Episode horizon (default 1 000)
    max_thrust : float
        Maximum specific force per axis [m/s²] (default 0.1)
    initial_fuel : float
        Starting fuel [kg] (default 100)
    fuel_per_thrust : float
        Fuel per unit thrust per timestep [kg/(m/s²·s)]
    dock_radius : float
        Success distance threshold [m] (default 1.0)
    dock_speed : float
        Success speed threshold [m/s] (default 0.5)
    station_box : float
        Box half-width [m] (default 50)
    render_mode : str or None
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "docking",
        mode: str = "2d",
        altitude_km: float = 400.0,
        dt: float = 1.0,
        max_steps: int = 1_000,
        max_thrust: float = 0.1,
        initial_fuel: float = 100.0,
        fuel_per_thrust: float = 0.05,
        dock_radius: float = 1.0,
        dock_speed: float = 0.5,
        station_box: float = 50.0,
        render_mode: str | None = None,
    ):
        super().__init__()

        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")

        self.task          = task
        self.mode          = mode
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
        self.is_3d         = (mode == "3d")

        # ── Observation space ────────────────────────────────────────────────
        if self.is_3d:
            if task == "station_keeping":
                # [x,y,z, vx,vy,vz, fuel, vy_corr, vz_corr]
                obs_low  = np.array([-5,-5,-5, -5,-5,-5, 0, -5,-5], dtype=np.float32)
                obs_high = np.array([ 5, 5, 5,  5, 5, 5, 1,  5, 5], dtype=np.float32)
            else:
                # [x,y,z, vx,vy,vz, fuel]
                obs_low  = np.array([-5,-5,-5, -5,-5,-5, 0], dtype=np.float32)
                obs_high = np.array([ 5, 5, 5,  5, 5, 5, 1], dtype=np.float32)
        else:
            if task == "station_keeping":
                obs_low  = np.array([-5,-5,-5,-5, 0,-5], dtype=np.float32)
                obs_high = np.array([ 5, 5, 5, 5, 1, 5], dtype=np.float32)
            else:
                obs_low  = np.array([-5,-5,-5,-5, 0], dtype=np.float32)
                obs_high = np.array([ 5, 5, 5, 5, 1], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # ── Action space ─────────────────────────────────────────────────────
        n_act = 3 if self.is_3d else 2
        self.action_space = spaces.Box(
            low  = np.full(n_act, -1.0, dtype=np.float32),
            high = np.full(n_act,  1.0, dtype=np.float32),
        )

        self._fig = None
        self._trajectory: list[np.ndarray] = []

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.task == "docking":
            pos = self.np_random.uniform(-1_000, 1_000, size=2)
            vel = self.np_random.uniform(   -2.0,   2.0, size=2)
            if self.is_3d:
                z   = self.np_random.uniform(-200, 200)
                vz  = self.np_random.uniform( -1.0,  1.0)
                self._state = np.array([pos[0], pos[1], z,
                                        vel[0], vel[1], vz], dtype=np.float64)
            else:
                self._state = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float64)

        else:  # station_keeping
            inner = self.station_box * 0.3
            pos = self.np_random.uniform(-inner, inner, size=2)
            vel = self.np_random.uniform(-0.1, 0.1, size=2)
            if self.is_3d:
                z   = self.np_random.uniform(-inner, inner)
                vz  = self.np_random.uniform(-0.05, 0.05)
                self._state = np.array([pos[0], pos[1], z,
                                        vel[0], vel[1], vz], dtype=np.float64)
            else:
                self._state = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float64)

        self._fuel  = float(self.initial_fuel)
        self._steps = 0
        self._trajectory = [self._state[:2].copy()]
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        if self.is_3d:
            force = action * self.max_thrust          # shape (3,)
        else:
            force = action * self.max_thrust          # shape (2,)

        thrust_magnitude = float(np.linalg.norm(force))
        fuel_used        = self.fuel_per_unit * thrust_magnitude * self.dt
        self._fuel       = max(0.0, self._fuel - fuel_used)

        if self._fuel <= 0.0:
            force = np.zeros_like(force)

        self._state = propagate(self._state, self.n, force, self.dt)
        self._steps += 1
        self._trajectory.append(self._state[:2].copy())

        obs                    = self._get_obs()
        reward                 = self._compute_reward(force)
        done, truncated, info  = self._check_terminal()

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, truncated, info

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        if self.is_3d:
            x, y, z, vx, vy, vz = self._state
            obs = [x/_POS_SCALE, y/_POS_SCALE, z/_POS_SCALE,
                   vx/_VEL_SCALE, vy/_VEL_SCALE, vz/_VEL_SCALE,
                   self._fuel/_FUEL_SCALE]
            if self.task == "station_keeping":
                obs.append(-2.0 * self.n * x / _VEL_SCALE)  # vy correction
                obs.append(0.0)                               # vz correction = 0 (z is harmonic)
        else:
            x, y, vx, vy = self._state
            obs = [x/_POS_SCALE, y/_POS_SCALE,
                   vx/_VEL_SCALE, vy/_VEL_SCALE,
                   self._fuel/_FUEL_SCALE]
            if self.task == "station_keeping":
                obs.append(-2.0 * self.n * x / _VEL_SCALE)

        return np.array(obs, dtype=np.float32)

    def _get_pos_vel(self):
        """Return (position_3d, velocity_3d) regardless of mode."""
        if self.is_3d:
            x, y, z, vx, vy, vz = self._state
        else:
            x, y, vx, vy = self._state
            z = vz = 0.0
        return np.array([x, y, z]), np.array([vx, vy, vz])

    def _compute_reward(self, force: np.ndarray) -> float:
        pos, vel = self._get_pos_vel()
        dist  = float(np.linalg.norm(pos))
        speed = float(np.linalg.norm(vel))

        if self.task == "docking":
            reward  = -dist / _POS_SCALE
            reward += 0.5 / (dist + 0.5)
            if dist < 50.0:
                reward += 0.3 * max(0.0, 1.0 - speed / self.dock_speed)
            if dist < self.dock_radius and speed < self.dock_speed:
                reward += 500.0
            reward -= 0.005 * float(np.linalg.norm(force)) / self.max_thrust

        else:  # station_keeping
            if self.is_3d:
                x, y, z = pos
                in_box = (abs(x) < self.station_box and
                          abs(y) < self.station_box and
                          abs(z) < self.station_box)
            else:
                x, y = pos[0], pos[1]
                in_box = abs(x) < self.station_box and abs(y) < self.station_box

            if in_box:
                reward = 1.0 + 0.5 * (1.0 - dist / self.station_box)
            else:
                dx = max(0.0, abs(pos[0]) - self.station_box)
                dy = max(0.0, abs(pos[1]) - self.station_box)
                dz = max(0.0, abs(pos[2]) - self.station_box) if self.is_3d else 0.0
                edge_dist = np.sqrt(dx**2 + dy**2 + dz**2)
                reward = -1.0 - edge_dist / (self.station_box * 2.0)

            reward -= 0.005 * float(np.linalg.norm(force)) / self.max_thrust

        return float(reward)

    def _check_terminal(self):
        pos, vel = self._get_pos_vel()
        dist  = float(np.linalg.norm(pos))
        speed = float(np.linalg.norm(vel))
        info  = {"distance": dist, "speed": speed,
                 "fuel": self._fuel, "steps": self._steps}

        if self.task == "docking" and dist < self.dock_radius and speed < self.dock_speed:
            info["outcome"] = "docked"
            return True, False, info

        if self.task == "docking" and dist < self.dock_radius and speed >= self.dock_speed:
            info["outcome"] = "crash"
            return True, False, info

        runaway_limit = 5_000.0 if self.task == "docking" else self.station_box * 4.0
        if dist > runaway_limit:
            info["outcome"] = "runaway"
            return True, False, info

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
            rect = plt.Rectangle((-b, -b), 2*b, 2*b, linewidth=1,
                                  edgecolor="lime", facecolor="lime", alpha=0.05)
            ax.add_patch(rect)

        ax.set_xlim(-1_200, 1_200)
        ax.set_ylim(-1_200, 1_200)
        ax.set_xlabel("Along-track y [m]")
        ax.set_ylabel("Radial x [m]")
        mode_str = "3D" if self.is_3d else "2D"
        ax.set_title(f"{mode_str} {self.task} | Step {self._steps} | Fuel: {self._fuel:.1f} kg")
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
