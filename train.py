"""
train.py — Train a PPO agent on OrbitalEnv
==========================================
Usage:
    python train.py --task docking
    python train.py --task station_keeping
    python train.py --task docking --timesteps 2000000
    python train.py --task docking --resume models/docking/ppo_docking_final.zip

Checkpoints saved to models/{task}/ every 100k steps.
TensorBoard logs written to logs/{task}/
"""

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.orbital_env import OrbitalEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       default="docking",
                   choices=["docking", "station_keeping"])
    p.add_argument("--timesteps",  type=int, default=1_000_000)
    p.add_argument("--n_envs",     type=int, default=8)
    p.add_argument("--altitude",   type=float, default=400.0)
    p.add_argument("--device",     type=str,   default="auto")
    p.add_argument("--resume",     type=str,   default=None,
                   help="Path to a .zip model to resume training from")
    p.add_argument("--fresh",      action="store_true",
                   help="Delete existing logs and checkpoints before training")
    return p.parse_args()


def main():
    args = parse_args()

    model_dir = Path("models") / args.task
    log_dir   = Path("logs")   / args.task
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        import shutil
        print(f"  --fresh: clearing {log_dir} and checkpoints in {model_dir}")
        if log_dir.exists():
            shutil.rmtree(log_dir)
        # Remove checkpoint zips but keep best/
        for f in model_dir.glob("ppo_*_steps.zip"):
            f.unlink()
        log_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        return Monitor(OrbitalEnv(task=args.task, altitude_km=args.altitude))

    train_env = make_vec_env(make_env, n_envs=args.n_envs)
    eval_env  = make_vec_env(make_env, n_envs=1)

    checkpoint_cb = CheckpointCallback(
        save_freq   = 100_000 // args.n_envs,
        save_path   = str(model_dir),
        name_prefix = f"ppo_{args.task}",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(model_dir / "best"),
        log_path             = str(log_dir),
        eval_freq            = 50_000 // args.n_envs,
        n_eval_episodes      = 20,
        deterministic        = True,
        verbose              = 1,
    )

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env            = train_env,
            tensorboard_log= str(log_dir),
            device         = args.device,
            verbose        = 1,
        )
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 256,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.005,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            policy_kwargs   = dict(net_arch=[256, 256]),
            tensorboard_log = str(log_dir),
            device          = args.device,
            verbose         = 1,
        )

    base_env = train_env.envs[0].unwrapped
    print(f"\n{'='*60}")
    print(f"  Task         : {args.task}")
    print(f"  Altitude     : {args.altitude} km  (n = {base_env.n:.6f} rad/s)")
    print(f"  Device       : {model.device}")
    print(f"  Timesteps    : {args.timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Dock speed   : {base_env.dock_speed} m/s")
    print(f"  TensorBoard  : tensorboard --logdir {log_dir}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps = args.timesteps,
        callback        = [checkpoint_cb, eval_cb],
        progress_bar    = True,
        reset_num_timesteps = args.resume is None,
    )

    final_path = model_dir / f"ppo_{args.task}_final"
    model.save(str(final_path))
    print(f"\n✓ Saved → {final_path}.zip")
    print(f"  Run: python enjoy.py --task {args.task}\n")


if __name__ == "__main__":
    main()
