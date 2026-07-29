"""
rl_optuna.py

Optuna hyperparameter search for PPO/RL training on Baloo.
Matches the same evaluation protocol as MPC Optuna:
  - Same 250 fixed LHS boxes (seed=42) for evaluation
  - Same outcome metrics (success rate)
  - 6 hyperparameters tuned (same count as MPC)

Each trial:
  1. Trains PPO for 6M steps with the trial's hyperparams
  2. Evaluates on 250 fixed LHS boxes
  3. Reports success rate to Optuna

Usage:
    uv run python rl_optuna.py --trials 100
    uv run python rl_optuna.py --trials 100 --eval_boxes 250
"""

import os
import ast
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import qmc

# ── Paths ─────────────────────────────────────────────────────────────────
CTRL_TIMESTEP  = 0.05
TOTAL_TIMESTEPS = 6_000_000
NUM_ENVS        = 16
MAX_STEPS       = 1200   # matches Curtis 60s at 20Hz
LIFT_THRESHOLD  = 0.5    # box must rise 0.5m above initial z


# ── LHS box sampling (matches Curtis Table 9.1) ────────────────────────────

def load_or_generate_lhs_samples(N, seed=42):
    fname = f"{N}_lhs_samples.txt"
    if os.path.exists(fname):
        with open(fname) as f:
            return [ast.literal_eval(line.strip()) for line in f]
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    lhs = sampler.random(n=N)
    samples = []
    for i in range(N):
        x    = np.interp(lhs[i, 0], [0, 1], [0.2, 0.6])ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968
        y    = np.interp(lhs[i, 1], [0, 1], [0.2, 0.6])
        z    = np.interp(lhs[i, 2], [0, 1], [0.5, 1.25])
        mass = np.interp(lhs[i, 3], [0, 1], [0.5, 10.ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.9680])
        xpos = np.interp(lhs[i, 4], [0, 1], [-0.1, 0.1])
        rot  = np.interp(lhs[i, 5], [0, 1], [-np.pi / 3, np.pi / 3])
        samples.append((x, y, z, mass, xpos, rot))ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968
    with open(fname, "w") as f:
        for s in samples:
            f.write(f"{s}\n")ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968
    return samplesial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968

ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968
# ── Evaluate a trained policy on LHS boxes ─ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968iial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968al 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=iaial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968l 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.9680.968───────────────────────────────

def evaluate_policy(policy, boxes, n_eval_boxes, seed=42):
    """ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968
    Evaluate a trained PPO policy on fixed LHS boxes.
    Returns success rate.
    """ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968
    from baloo_gym.envs.baloo_v9 import Balial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968ooV9
    from baloo_gym.wrappers.three_part_rewarial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968d_ial iial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968> sial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968uccess=0.968al 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.96825: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=ialial 25ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.9680.968wrapper import ThreePartRewardWrapper
ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 enial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968t=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968
    # Same fixed indices as MPC Optuna (seed=42)
    fixed_idxs = np.random.default_rng(42)ial 25: lr=3.6e-04->3.6e-06 gamma=0.9959 ent=0.0023 gae=0.909 batch=1024 n_steps=4096 -> success=0.968.choice(len(boxes), size=n_eval_boxes, replace=False)

    successes = 0
    for i in fixed_idxs:
        x, y, z, mass, xpos, rot = boxes[i]
        try:
            env = BalooV9(
                render_mode=None,
                camera_name='frontcam',
                ctrl_timestep=CTRL_TIMESTEP,
                randomize_initial_height=False,
                randomize_object_size=False,
                randomize_object_mass=False,
                object_size=[x, y, z],
                object_mass=mass,
                object_xpos=xpos,
                object_zrotation=rot,
            )
            env = ThreePartRewardWrapper(env, reward_selection=['dont_drop'])
            obs, _ = env.reset()

            done = False
            step = 0
            success = False
            while not done and step < MAX_STEPS:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                success = info.get("is_success", False)
                step += 1
            env.close()
            if success:
                successes += 1
        except Exception as e:
            print(f"  Eval failed on box {i}: {e}")

    return successes / n_eval_boxes


# ── Train one PPO trial ────────────────────────────────────────────────────

def train_and_evaluate(trial_params, trial_number, n_eval_boxes, boxes):
    """
    Train PPO with given hyperparams for TOTAL_TIMESTEPS, then evaluate.
    Returns success rate.
    """
    import copy
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.utils import set_random_seed
    from baloo_gym.utils.helpers import build_env

    set_random_seed(42 + trial_number, using_cuda=True)  # different seed per trial but reproducible

    lr_initial = trial_params["lr_initial"]
    lr_final   = trial_params["lr_final"]
    gamma      = trial_params["gamma"]
    ent_coef   = trial_params["ent_coef"]
    gae_lambda = trial_params["gae_lambda"]
    batch_size = trial_params["batch_size"]
    n_steps    = trial_params["n_steps"]

    def linear_schedule(initial_value, final_value):
        def func(progress_remaining):
            return progress_remaining * (initial_value - final_value) + final_value
        return func

    config = {
        "total_timesteps": TOTAL_TIMESTEPS,
        "ctrl_timestep": CTRL_TIMESTEP,
        "env_name": "baloo_v9",
        "time_limit_sec": 60,
        "curriculum_selection": [],
        "reward_selection": ["copy_baseline"],
        "randomize_initial_height": False,
        "randomize_object_size": True,
        "randomize_object_mass": True,
        "randomize_object_quat": True,
        "randomize_object_pos": True,
    }

    save_dir = f"rl_optuna_trials/trial_{trial_number}"
    os.makedirs(save_dir, exist_ok=True)

    vec_env = make_vec_env(
        build_env,
        env_kwargs={"config": config, "baseline": False,
                    "render_mode": "rgb_array", "potential_based_reward": False,
                    "resolution": 0.5},
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=f"{save_dir}/monitor",
        monitor_kwargs={"info_keywords": ("is_success",)},
        seed=42,
    )

    policy_kwargs = dict(net_arch=[512, 512])
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        batch_size=batch_size,
        learning_rate=linear_schedule(lr_initial, lr_final),
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        verbose=0,
        device="auto",
    )

    print(f"\n  Training trial {trial_number} for {TOTAL_TIMESTEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    # save model
    model.save(f"{save_dir}/model")
    print(f"  Model saved → {save_dir}/model.zip")

    # evaluate
    print(f"  Evaluating on {n_eval_boxes} boxes...")
    success_rate = evaluate_policy(model, boxes, n_eval_boxes)
    print(f"  Success rate: {success_rate:.3f}")

    vec_env.close()
    return success_rate


# ── Optuna objective ───────────────────────────────────────────────────────

def run_optuna(n_trials=100, n_eval_boxes=250):
    import optuna
    from tqdm import tqdm
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    boxes = load_or_generate_lhs_samples(1000, seed=42)
    print(f"Using {n_eval_boxes} fixed eval boxes (seed=42, same as MPC Optuna)")

    pbar = tqdm(total=n_trials, desc="RL Optuna trials", unit="trial")

    def objective(trial):
        params = {
            "lr_initial":  trial.suggest_float("lr_initial",  1e-4, 1e-3, log=True),
            "lr_final":    trial.suggest_float("lr_final",    1e-6, 1e-4, log=True),
            "gamma":       trial.suggest_float("gamma",       0.98, 0.999),
            "ent_coef":    trial.suggest_float("ent_coef",    0.0,  0.01),
            "gae_lambda":  trial.suggest_float("gae_lambda",  0.9,  0.99),
            "batch_size":  trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "n_steps":     trial.suggest_categorical("n_steps",    [4096, 8192, 16384]),
        }

        try:
            rate = train_and_evaluate(params, trial.number, n_eval_boxes, boxes)
        except Exception as e:
            print(f"\nTrial {trial.number} failed: {e}")
            rate = 0.0

        trial.set_user_attr("params", params)

        try:
            best_so_far = study.best_value
        except ValueError:
            best_so_far = 0.0

        pbar.set_postfix({"success": f"{rate:.3f}", "best": f"{best_so_far:.3f}"})
        pbar.update(1)
        print(f"\nTrial {trial.number}: lr={params['lr_initial']:.1e}->{params['lr_final']:.1e} "
              f"gamma={params['gamma']:.4f} ent={params['ent_coef']:.4f} "
              f"gae={params['gae_lambda']:.3f} batch={params['batch_size']} "
              f"n_steps={params['n_steps']} -> success={rate:.3f}")
        return rate

    study = optuna.create_study(
        direction='maximize',
        study_name='rl_baloo',
        storage='sqlite:///rl_optuna.db',
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    pbar.close()

    print("\n=== Best hyperparameters ===")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  best success rate: {study.best_value:.3f}")

    with open("rl_optuna_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("Saved → rl_optuna_best_params.json")
    return study.best_params


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",     type=int, default=100)
    parser.add_argument("--eval_boxes", type=int, default=250)
    args = parser.parse_args()

    run_optuna(n_trials=args.trials, n_eval_boxes=args.eval_boxes)
