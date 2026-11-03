#take in policy from recent_model folder of a given experiment.and
# load it with sb3
# build an environment
# run the environment with the policy
# record the rewards, actions, and observations

from stable_baselines3 import PPO
import argparse
from baloo_gym.utils.helpers import build_env, record_rollout, make_movie
from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_gym.wrappers import ObjectPerturbationWrapper
import wandb
import os
import ast
from baloo_mujoco_sim.utils.baloo_mj_api import (set_box_size,
                                                 set_box_position)
import shutil
import numpy as np
from baloo_gym.utils.helpers import load_or_download_model

np.set_printoptions(precision=3, suppress=True)


def load_or_generate_lhs_samples(n_samples=1000, seed=42, filepath="1000_lhs_samples.txt"):
    candidate_paths = [
        filepath,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_data/perturbations", filepath),
        os.path.expanduser("~/baloo-ws/baloo-mujoco-sim/1000_lhs_samples.txt"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            print(f"Loading LHS samples from {p}")
            with open(p, "r") as f:
                return [ast.literal_eval(line.strip()) for line in f if line.strip()]

    print(f"Generating {n_samples} LHS samples (seed={seed})...")
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    raw = sampler.random(n=n_samples)
    x_s = np.interp(raw[:, 0], [0, 1], [0.2, 0.6])
    y_s = np.interp(raw[:, 1], [0, 1], [0.2, 0.6])
    z_s = np.interp(raw[:, 2], [0, 1], [0.5, 1.25])
    m_s = np.interp(raw[:, 3], [0, 1], [0.5, 10.0])
    xp_s = np.interp(raw[:, 4], [0, 1], [-0.1, 0.1])
    r_s = np.interp(raw[:, 5], [0, 1], [-np.pi / 3, np.pi / 3])
    combos = [(x, y, z, m, xp, r) for x, y, z, m, xp, r in zip(x_s, y_s, z_s, m_s, xp_s, r_s)]
    try:
        with open(filepath, "w") as f:
            for c in combos:
                f.write(f"{c}\n")
    except Exception:
        pass
    return combos


def extract_reward_trajectories(infos):
    reward_history = {
        "dont_drop": [],
        "dropped": [],
        "chest_proximity": [],
        "touch_ground": [],
        'copy_baseline': [],
        "tactile_nonzero": [],
        "minimize_torques": [],
        "upward_force": [],
        "success": [],
        "box_fell_over": [],
    }

    for info in infos:
        for k in reward_history.keys():
            reward_history[k].append(info["reward_terms"].get(k, 0))

    return reward_history


parser = argparse.ArgumentParser()
parser.add_argument('--runid', type=str, help="Wandb run id")
parser.add_argument('--num_rollouts',
                    type=int,
                    default=1,
                    help="Number of rollouts to average over")

parser.add_argument(
    '--resolution',
    type=int,
    default=1,
    help="Resolution multiplier of video (160x120) * resolution")

parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--num_detail',
                    type=int,
                    default=5,
                    help="Number of rollouts to save detailed video and plots for")
parser.add_argument('--perturb',
                    action='store_true',
                    default=False,
                    help="Apply external force perturbations to the object")
parser.add_argument('--perturbation_magnitude',
                    type=float,
                    default=100.0,
                    help="Magnitude of perturbation in Newtons (default: 100.0)")
parser.add_argument('--use_lhs',
                    action='store_true',
                    default=False,
                    help="Evaluate using fixed Latin Hypercube Sampled (LHS) boxes")
parser.add_argument('--lhs_file',
                    type=str,
                    default="1000_lhs_samples.txt",
                    help="Path to file containing LHS samples")

args = parser.parse_args()

#load the config from the wandb synced run.
if args.runid is None:
    is_baseline = True
    folder_name = "open_loop_hugger"
    config = {
        "total_timesteps":
        1000000,
        "ctrl_timestep":
        .05,
        "env_name":
        "baloo_v9",
        "time_limit_sec":
        60,
        "curriculum_selection": [],
        'reward_selection': [
            # 'dont_drop',
            'copy_baseline',
            # 'chest_proximity',
            # 'touch_ground',
            # 'tactile_nonzero',
            # 'upward_force',
            # 'minimize_torques',
        ],
        "randomize_initial_height":
        False,
        "randomize_object_size":
        True,
        "randomize_object_mass":
        True,
        "randomize_object_quat":
        True,
        "randomize_object_pos":
        True,
    }
    model = None
else:
    is_baseline = False
    try:
        run = wandb.Api().run(f"curtiscjohnson/ppo_baloo/{args.runid}")
        folder_name = f"{run.name}-{run.id}"
        run_config = run.config
    except Exception as e:
        print(f"Warning: W&B API fetch failed ({e}). Checking local configuration...")
        run = None
        local_exp = "./new_experiments"
        folder_name = None
        if os.path.exists(local_exp):
            for f in os.listdir(local_exp):
                if args.runid in f:
                    folder_name = f
                    break
        if folder_name is None:
            folder_name = f"run-{args.runid}"

        run_config = {
            "total_timesteps": 40000000,
            "ctrl_timestep": 0.05,
            "env_name": "baloo_v9",
            "time_limit_sec": 60,
            "curriculum_selection": [],
            "reward_selection": ["copy_baseline"],
            "gamma": 0.995,
        }
        config_yaml_path = f"{local_exp}/{folder_name}/config.yaml"
        if os.path.exists(config_yaml_path):
            import yaml
            with open(config_yaml_path, "r") as cyf:
                loaded_yaml = yaml.safe_load(cyf)
                for k, v in loaded_yaml.items():
                    if isinstance(v, dict) and "value" in v:
                        run_config[k] = v["value"]

    config = {
        "total_timesteps": run_config["total_timesteps"],
        "ctrl_timestep": run_config["ctrl_timestep"],
        "env_name": run_config["env_name"],
        "time_limit_sec": run_config["time_limit_sec"],
        # "time_limit_sec": 120,
        "curriculum_selection": run_config["curriculum_selection"],
        'reward_selection': run_config['reward_selection'],
        "randomize_initial_height": False,
        "randomize_object_size": True,
        "randomize_object_mass": True,
    }

    model_path = load_or_download_model(args, f"./new_experiments")
    model = PPO.load(model_path)

# Build results directory tag
tmp_name = args.model_name.split(".")[0]
tag_parts = []
if args.perturb:
    tag_parts.append(f"pert{int(args.perturbation_magnitude)}")
else:
    tag_parts.append("nopert")
if args.use_lhs:
    tag_parts.append("lhs")
tag_str = "_" + "_".join(tag_parts)

out_folder_name = f"{folder_name}-{tmp_name}{tag_str}"
out_dir = f"./evaluation_results/{out_folder_name}"
os.makedirs(out_dir, exist_ok=True)

if args.use_lhs:
    lhs_samples = load_or_generate_lhs_samples(args.num_rollouts, filepath=args.lhs_file)
    print(f"Using LHS samples ({len(lhs_samples)} available). Each rollout tests a fixed sampled box.")
    if args.perturb:
        print(f"Applying object perturbations: {args.perturbation_magnitude} N")
    env = None
else:
    env = build_env(config,
                    baseline=is_baseline,
                    render_mode="rgb_array",
                    resolution=args.resolution)
    if args.perturb:
        print(f"Applying object perturbations: {args.perturbation_magnitude} N")
        env = ObjectPerturbationWrapper(env)
        env.perturbation_magnitude = args.perturbation_magnitude

successes = []
returns = []
lengths = []

for j in range(args.num_rollouts):
    save_detail = j < args.num_detail

    if args.use_lhs:
        x, y, z, m, xp, r = lhs_samples[j % len(lhs_samples)]
        trial_config = config.copy()
        trial_config["randomize_object_size"] = False
        trial_config["randomize_object_mass"] = False
        trial_config["randomize_object_quat"] = False
        trial_config["randomize_object_pos"] = False
        current_env = build_env(
            trial_config,
            baseline=is_baseline,
            render_mode="rgb_array",
            resolution=args.resolution,
            object_size=[x, y, z],
            object_mass=m,
            object_xpos=xp,
            object_zrotation=r,
        )
        if args.perturb:
            current_env = ObjectPerturbationWrapper(current_env)
            current_env.perturbation_magnitude = args.perturbation_magnitude
    else:
        current_env = env

    if is_baseline:
        policy_model = OpenLoopHuggerPolicy(N=50)
        frames, rewards, actions, observations, infos = record_rollout(
            current_env, policy_model, deterministic=True, render=save_detail, return_dist=False)
        dists = None
    else:
        policy_model = model
        if save_detail:
            frames, rewards, actions, observations, infos, dists = record_rollout(
                current_env, policy_model, deterministic=True, render=True, return_dist=True)
        else:
            frames, rewards, actions, observations, infos = record_rollout(
                current_env, policy_model, deterministic=True, render=False, return_dist=False)
            dists = None

    # (current_env closed at end of iteration if use_lhs)

    is_succ = infos[-1].get("is_success", False) if "is_success" in infos[-1] else False
    successes.append(is_succ)
    lengths.append(len(rewards))

    # calculate discounted return
    rewards = np.array(rewards)
    discounted_sum = 0
    try:
        gamma = run_config.get("gamma", 0.99) if "run_config" in locals() and run_config else 0.99
    except:
        gamma = 0.99

    for i in range(len(rewards)):
        discounted_sum += (gamma**i) * rewards[i]
    returns.append(discounted_sum)

    print(
        f"[{j+1}/{args.num_rollouts}] return: {discounted_sum:.2f} | len: {len(rewards)} | success: {is_succ}"
    )

    if save_detail:
        reward_history = extract_reward_trajectories(infos)
        run_path = f"{out_dir}/rollout_{j}"
        os.makedirs(run_path, exist_ok=True)

        import matplotlib.pyplot as plt
        import numpy as np
        fig, axs = plt.subplots(10, 1, figsize=(10, 40), sharex=True)

        for i, k in enumerate(reward_history.keys()):
            axs[i].plot(reward_history[k], label=k)
            axs[i].legend()
            axs[i].minorticks_on()
            axs[i].grid(which='both')
            axs[i].set_ylabel("Rewards")

        axs[-1].set_xlabel("Timesteps")
        plt.savefig(run_path + f"/reward_terms.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        make_movie(frames,
                   run_path + f"/evaluation_rollout.mp4",
                   fps=1 / config["ctrl_timestep"])

        fig = plt.figure()
        plt.plot(rewards)
        plt.minorticks_on()
        plt.grid(which='both')
        plt.xlabel("Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(run_path + f"/rewards.png", dpi=300)
        plt.close(fig)

        fig, axs = plt.subplots(
            current_env.action_space.shape[0],
            1,
            sharex=True,
            figsize=(10, 30),
        )
        for i in range(current_env.action_space.shape[0]):
            axs[i].plot(np.array(actions)[:, i])
            axs[i].set_ylabel(f"a{i}")
            axs[i].minorticks_on()
            axs[i].grid(which='both')

        axs[-1].set_xlabel("Timesteps")
        plt.savefig(run_path + f"/actions.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # make histogram of actions
        fig, axs = plt.subplots(
            current_env.action_space.shape[0],
            1,
            figsize=(10, 30),
        )
        for i in range(current_env.action_space.shape[0]):
            axs[i].hist(np.array(actions)[:, i], bins=100)
            axs[i].set_ylabel(f"a{i}")
            axs[i].minorticks_on()
            axs[i].grid(which='both')

        axs[-1].set_xlabel("Action Value")
        plt.savefig(run_path + f"/actions_hist.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, axs = plt.subplots(current_env.observation_space.shape[0],
                                1,
                                sharex=True,
                                figsize=(10, 60))
        for i in range(current_env.observation_space.shape[0]):
            axs[i].plot(np.array(observations)[:, i])
            axs[i].set_ylabel(f"o{i}")
            axs[i].minorticks_on()
            axs[i].grid(which='both')

        axs[-1].set_xlabel("Timesteps")
        plt.savefig(run_path + f"/observations.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # plot the action distribution for each action
        if args.runid is not None and dists is not None:
            means = [dists[i][0] for i in range(len(dists))]
            stds = [dists[i][1] for i in range(len(dists))]

            means = np.array(means)
            stds = np.array(stds)

            fig, axs = plt.subplots(current_env.action_space.shape[0], 1, figsize=(10, 30))

            for i in range(current_env.action_space.shape[0]):
                axs[i].plot(means[:, i], label="mean")
                axs[i].plot(np.array(actions)[:, i], 'r--', label="action")
                axs[i].plot(np.ones(len(means)) * current_env.action_space.low[i], 'k--')
                axs[i].plot(np.ones(len(means)) * current_env.action_space.high[i], 'k--')
                axs[i].fill_between(
                    np.arange(len(means)),
                    means[:, i] - 3 * stds[:, i],
                    means[:, i] + 3 * stds[:, i],
                    alpha=0.5,
                    label="3 std",
                )
                axs[i].legend()
                axs[i].set_ylabel(f"a{i}")
                axs[i].minorticks_on()
                axs[i].grid(which='both')

            axs[-1].set_xlabel("Timesteps")
            plt.savefig(run_path + f"/action_dist.png",
                        dpi=300,
                        bbox_inches='tight')
            plt.close('all')

    if args.use_lhs:
        current_env.close()

print(f"\n{'='*55}")
print(f"=== Evaluation Complete ({len(successes)} rollouts) ===")
print(f"{'='*55}")
print(f"  Success rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{len(successes)})")
print(f"  Mean return:  {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
print(f"  Mean length:  {np.mean(lengths):.1f}")
print(f"{'='*55}\n")

import json
summary = {
    "num_rollouts": len(successes),
    "success_rate": float(np.mean(successes)),
    "mean_return": float(np.mean(returns)),
    "std_return": float(np.std(returns)),
    "mean_length": float(np.mean(lengths)),
    "perturb": args.perturb,
    "perturbation_magnitude": float(args.perturbation_magnitude) if args.perturb else 0.0,
    "use_lhs": args.use_lhs,
    "model_name": args.model_name,
    "runid": args.runid,
}
with open(f"{out_dir}/summary.json", "w") as sf:
    json.dump(summary, sf, indent=2)
print(f"Summary saved to {out_dir}/summary.json")
