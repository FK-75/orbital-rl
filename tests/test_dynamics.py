"""
tests/test_dynamics.py — Unit tests for CW propagator and environment
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dynamics import mean_motion, cw_derivatives, rk4_step, propagate


class TestMeanMotion:
    def test_leo_value(self):
        """LEO mean motion should be ~0.00113 rad/s."""
        n = mean_motion(400.0)
        assert 0.001 < n < 0.0015, f"Got n={n}"

    def test_higher_orbit_slower(self):
        """Higher orbit → lower mean motion."""
        assert mean_motion(800.0) < mean_motion(400.0)

    def test_geo_approx(self):
        """GEO orbit (~35 786 km) should have n ≈ 7.27e-5 rad/s."""
        n = mean_motion(35_786.0)
        assert abs(n - 7.27e-5) < 1e-6


class TestDerivatives:
    def setup_method(self):
        self.n = mean_motion(400.0)

    def test_zero_at_origin(self):
        """At origin with zero velocity and no force, all derivatives = 0."""
        derivs = cw_derivatives(np.zeros(6), self.n, np.zeros(3))
        np.testing.assert_allclose(derivs, np.zeros(6), atol=1e-12)

    def test_coriolis_term(self):
        """Radial velocity vx > 0 → negative along-track acceleration (Coriolis)."""
        state  = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])   # vx=1 m/s
        derivs = cw_derivatives(state, self.n, np.zeros(3))
        assert derivs[4] < 0  # ay = -2n*vx

    def test_gravity_gradient(self):
        """Positive radial offset → positive radial acceleration (gravity gradient)."""
        state  = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        derivs = cw_derivatives(state, self.n, np.zeros(3))
        assert derivs[3] > 0  # ax = 3n²x

    def test_thrust_adds_directly(self):
        """Applied force should appear directly in acceleration components."""
        fx, fy, fz = 0.05, -0.03, 0.02
        derivs = cw_derivatives(np.zeros(6), self.n, np.array([fx, fy, fz]))
        np.testing.assert_allclose(derivs[3], fx, atol=1e-12)
        np.testing.assert_allclose(derivs[4], fy, atol=1e-12)
        np.testing.assert_allclose(derivs[5], fz, atol=1e-12)


class TestPropagator:
    def setup_method(self):
        self.n = mean_motion(400.0)

    def test_cw_analytic_x_periodicity(self):
        """
        CW analytical solution for IC (x0, 0, 0, 0):
          x(T) = x0*(4 - 3*cos(2π)) = x0   (radial component is periodic)
          y(T) = 6*x0*(sin(2π) - 2π)        (secular drift — expected, not an error)

        We verify the x-component returns within 2 m after one full orbit.
        """
        x0 = 100.0
        s0 = np.array([x0, 0.0, 0.0, 0.0])
        s  = s0.copy()
        T  = int(2 * np.pi / self.n)
        for _ in range(T):
            s = propagate(s, self.n, np.zeros(2), dt=1.0)
        assert abs(s[0] - x0) < 2.0, f"x error: {abs(s[0]-x0):.2f} m"

    def test_thrust_changes_velocity(self):
        """Non-zero thrust must change velocity."""
        s0       = np.zeros(4)
        s_thrust = propagate(s0, self.n, np.array([0.1, 0.0]), dt=1.0)
        assert s_thrust[2] != 0.0

    def test_substep_convergence(self):
        """More substeps → more accurate (error monotonically decreasing)."""
        s0 = np.array([300.0, -200.0, 1.5, -0.5])
        f  = np.array([0.05, -0.03])
        s1  = propagate(s0, self.n, f, dt=5.0, substeps=1)
        s5  = propagate(s0, self.n, f, dt=5.0, substeps=5)
        s20 = propagate(s0, self.n, f, dt=5.0, substeps=20)
        assert np.linalg.norm(s5 - s20) < np.linalg.norm(s1 - s20)

    def test_zero_force_bounded(self):
        """Free drift should stay within a reasonable distance bound over 500 steps."""
        s0 = np.array([200.0, 100.0, 0.5, -0.3])
        s  = s0.copy()
        for _ in range(500):
            s = propagate(s, self.n, np.zeros(2), dt=1.0)
        assert np.linalg.norm(s[:2]) < 10_000.0


class TestOrbitalEnv:
    def test_reset_obs_in_bounds(self):
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"

    def test_step_obs_in_bounds(self):
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv()
        obs, _ = env.reset(seed=0)
        obs, reward, done, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)

    def test_fuel_decreases_with_thrust(self):
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv()
        env.reset(seed=0)
        fuel_before = env._fuel
        env.step(np.array([1.0, 1.0], dtype=np.float32))
        assert env._fuel < fuel_before

    def test_zero_thrust_no_fuel_consumed(self):
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv()
        env.reset(seed=0)
        fuel_before = env._fuel
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env._fuel == fuel_before

    def test_station_keeping_task(self):
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="station_keeping")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (6,), f"station_keeping obs should be 6-dim, got {obs.shape}"
        obs, reward, done, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)

    def test_docking_obs_is_5dim(self):
        """Docking task uses 5-dim observation."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="docking")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (5,), f"docking obs should be 5-dim, got {obs.shape}"

    def test_station_keeping_starts_inside_box(self):
        """Station-keeping should always start inside the box."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="station_keeping", station_box=50.0)
        for seed in range(50):
            env.reset(seed=seed)
            x, y = env._state[0], env._state[1]
            assert abs(x) < 50.0 and abs(y) < 50.0, \
                f"Start outside box at seed {seed}: x={x:.1f}, y={y:.1f}"

    def test_station_keeping_in_box_positive_reward(self):
        """Agent inside the box should receive positive reward."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="station_keeping", station_box=50.0)
        env.reset(seed=0)
        env._state = np.array([10.0, 10.0, 0.0, 0.0])
        _, reward, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert float(reward) > 0.0, f"In-box reward should be positive, got {float(reward):.3f}"

    def test_station_keeping_outside_box_negative_reward(self):
        """Agent outside the box should receive negative reward."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="station_keeping", station_box=50.0)
        env.reset(seed=0)
        env._state = np.array([200.0, 200.0, 0.0, 0.0])
        _, reward, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert float(reward) < 0.0, f"Out-of-box reward should be negative, got {float(reward):.3f}"

    def test_station_keeping_drift_hint_sign(self):
        """CW drift hint should be negative when x > 0 (drift is in -y direction)."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="station_keeping")
        env.reset(seed=0)
        env._state = np.array([20.0, 0.0, 0.0, 0.0])  # positive x offset
        obs = env._get_obs()
        # obs[5] = -2*n*x / VEL_SCALE — should be negative when x > 0
        assert obs[5] < 0.0, f"Drift hint should be negative for x>0, got {obs[5]:.4f}"

    def test_docking_success_condition(self):
        """Manually place agent at origin with near-zero velocity — should dock."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="docking", dock_radius=5.0, dock_speed=1.0)
        env.reset(seed=0)
        # Force state to be very close with low speed
        env._state = np.array([0.1, 0.1, 0.05, 0.05])
        env._steps = 0
        _, _, done, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert done
        assert info["outcome"] == "docked"

    def test_reward_improves_when_closer(self):
        """
        Agent closer to target should get a higher per-step reward than far away.
        With 1/(d+0.5) shaping this should hold at any pair of distances.
        """
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="docking")

        # Close: 50 m away, zero velocity
        env.reset(seed=0)
        env._state = np.array([50.0, 0.0, 0.0, 0.0])
        _, r_close, *_ = env.step(np.array([0.0, 0.0], dtype=np.float32))

        # Far: 900 m away, zero velocity
        env.reset(seed=0)
        env._state = np.array([900.0, 0.0, 0.0, 0.0])
        _, r_far, *_ = env.step(np.array([0.0, 0.0], dtype=np.float32))

        assert float(r_close) > float(r_far), \
            f"Close reward {float(r_close):.4f} should exceed far reward {float(r_far):.4f}"

    def test_terminal_dock_bonus(self):
        """Docking should produce a large positive reward spike."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="docking", dock_radius=5.0, dock_speed=1.0)
        env.reset(seed=0)
        env._state = np.array([0.1, 0.1, 0.05, 0.05])
        env._steps = 0
        _, reward, done, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert done
        assert info["outcome"] == "docked"
        assert float(reward) > 100.0, \
            f"Dock bonus should dominate reward, got {float(reward):.2f}"


class TestDynamics3D:
    """Tests for the 3D CW extension."""

    def setup_method(self):
        self.n = mean_motion(400.0)

    def test_z_axis_decoupled(self):
        """Out-of-plane motion should not affect in-plane dynamics."""
        from envs.dynamics import cw_derivatives
        # Same x,y,vx,vy — different z
        s_z0  = np.array([100.0, 50.0, 0.0, 1.0, 0.5, 0.0,  0.0,  0.0])
        s_z50 = np.array([100.0, 50.0, 0.0, 1.0, 0.5, 0.0, 50.0,  0.2])
        # Map to 6-dim for cw_derivatives
        s1 = np.array([s_z0[0], s_z0[1], s_z0[6],  s_z0[3],  s_z0[4],  s_z0[7]])
        s2 = np.array([s_z50[0], s_z50[1], s_z50[6], s_z50[3], s_z50[4], s_z50[7]])
        d1 = cw_derivatives(s1, self.n, np.zeros(3))
        d2 = cw_derivatives(s2, self.n, np.zeros(3))
        # In-plane derivatives (indices 0,1,3,4) must be identical
        np.testing.assert_allclose(d1[[0,1,3,4]], d2[[0,1,3,4]], atol=1e-12)

    def test_z_harmonic_oscillator(self):
        """z-axis without thrust should oscillate like -n²z."""
        from envs.dynamics import cw_derivatives
        z0 = 100.0
        state = np.array([0.0, 0.0, z0, 0.0, 0.0, 0.0])
        derivs = cw_derivatives(state, self.n, np.zeros(3))
        # az = -n²*z0
        expected_az = -self.n**2 * z0
        np.testing.assert_allclose(derivs[5], expected_az, rtol=1e-8)

    def test_3d_propagate_backward_compat(self):
        """2D propagate calls still work after 3D extension."""
        from envs.dynamics import propagate
        s2d = np.array([100.0, 50.0, 0.5, -0.3])
        f2d = np.array([0.05, -0.02])
        result = propagate(s2d, self.n, f2d, dt=1.0)
        assert result.shape == (4,)

    def test_3d_env_obs_shape(self):
        """3D env should return 7-dim obs for docking."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="docking", mode="3d")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (7,), f"Expected (7,), got {obs.shape}"

    def test_3d_env_action_shape(self):
        """3D env should have 3-dim action space."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="docking", mode="3d")
        assert env.action_space.shape == (3,)

    def test_3d_station_keeping_obs_shape(self):
        """3D station_keeping should return 9-dim obs."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="station_keeping", mode="3d")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (9,), f"Expected (9,), got {obs.shape}"

    def test_3d_step_runs_without_error(self):
        """3D env should step cleanly for 10 steps."""
        from envs.orbital_env import OrbitalEnv
        env = OrbitalEnv(task="docking", mode="3d")
        obs, _ = env.reset(seed=0)
        for _ in range(10):
            obs, r, done, trunc, info = env.step(env.action_space.sample())
            if done or trunc:
                break
        assert env.observation_space.contains(obs)
