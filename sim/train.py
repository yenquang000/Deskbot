"""Staged PPO training for Deskbot.

  python train.py --stage 1 --steps 1500000 --out models/stage1     # Day 2 overnight
  python train.py --stage 2 --steps 2000000 --load models/stage1 --out models/stage2   # Day 4
  python train.py --stage 3 --steps 3000000 --load models/stage2 --out models/final    # Day 7

or everything back to back:  python train.py --curriculum

All stages share the same 26-number observation, so each stage continues from the
previous stage's weights. Best-on-eval checkpoints are kept in <out>_best/.
"""
import argparse
import os

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import DeskbotEnv


def train_stage(stage, steps, out, load=None, n_envs=8, seed=0, subproc=True):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vec_cls = SubprocVecEnv if subproc and n_envs > 1 else DummyVecEnv
    env = make_vec_env(lambda: DeskbotEnv(stage=stage), n_envs=n_envs, seed=seed, vec_env_cls=vec_cls)
    eval_env = Monitor(DeskbotEnv(stage=stage))

    if load:
        model = PPO.load(load, env=env, device="cpu")
        print(f"Stage {stage}: continuing from {load}")
    else:
        model = PPO(
            "MlpPolicy", env, seed=seed, verbose=1, device="cpu",   # small MLP: CPU is faster than GPU
            n_steps=1024, batch_size=512, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.01,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=nn.Tanh),
        )
    callbacks = [
        CheckpointCallback(save_freq=max(250_000 // n_envs, 1), save_path=f"{out}_ckpt", name_prefix="ppo"),
        EvalCallback(eval_env, n_eval_episodes=20, eval_freq=max(100_000 // n_envs, 1),
                     best_model_save_path=f"{out}_best", deterministic=True),
    ]
    print(f"Stage {stage}: training {steps:,} steps ...")
    model.learn(total_timesteps=steps, callback=callbacks, reset_num_timesteps=not load)
    model.save(out)
    print(f"Saved {out}.zip")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--out", default="models/stage1")
    ap.add_argument("--load", default=None)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-subproc", action="store_true", help="use if multiprocessing misbehaves")
    ap.add_argument("--curriculum", action="store_true", help="run stages 1-3 back to back")
    a = ap.parse_args()

    if a.curriculum:
        p = train_stage(1, 1_500_000, "models/stage1", None, a.envs, a.seed, not a.no_subproc)
        p = train_stage(2, 2_000_000, "models/stage2", p, a.envs, a.seed, not a.no_subproc)
        train_stage(3, 3_000_000, "models/final", p, a.envs, a.seed, not a.no_subproc)
    else:
        train_stage(a.stage, a.steps, a.out, a.load, a.envs, a.seed, not a.no_subproc)
