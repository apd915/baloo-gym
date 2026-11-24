import importlib
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_box_position,
    get_box_vel,
    get_elevator_height,
    get_elevator_vel,
    get_joint_angles,
    get_joint_vel,
)

import cProfile

from stable_baselines3.common.env_checker import check_env
from baloo_gym.wrappers import ThreePartRewardWrapper, PotentialBasedRewardWrapper
from stable_baselines3.common.policies import obs_as_tensor

from gymnasium.wrappers import TimeLimit

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import wandb
from tqdm import tqdm
import os
from pathlib import Path
import time
import imageio.v2 as imageio
import os
import gc


def get_sensor_data(model, data):
    left_pos = []
    left_vel = []
    right_pos = []
    right_vel = []
    for i in range(3):
        left_pos.append(get_joint_angles(model, data, "left", i))
        left_vel.append(get_joint_vel(model, data, "left", i))
        right_pos.append(get_joint_angles(model, data, "right", i))
        right_vel.append(get_joint_vel(model, data, "right", i))

    object_pos = get_box_position(model, data)
    object_vel = get_box_vel(model, data)
    elevator_pos = get_elevator_height(model, data)
    elevator_vel = get_elevator_vel(model, data)

    return {
        "object_pos": object_pos,
        "object_vel": object_vel,
        "elevator_pos": elevator_pos,
        "elevator_vel": elevator_vel,
        "left_pos": np.hstack(left_pos),
        "right_pos": np.hstack(right_pos),
        "left_vel": np.hstack(left_vel),
        "right_vel": np.hstack(right_vel),
    }


def get_action_dist_params(model, obs):
    obs = model.policy.obs_to_tensor(obs)[0]
    dis = model.policy.get_distribution(obs)
    loc = dis.distribution.loc.detach().cpu().numpy()
    scale = dis.distribution.scale.detach().cpu().numpy()
    return loc.squeeze(), scale.squeeze()


def record_rollout(env,
                   policy,
                   render=True,
                   deterministic=True,
                   return_dist=True) -> tuple:
    obs, info = env.reset()
    # print(f"In rollout, obs shape: {obs.shape}")
    done = False

    frames = []
    rewards = []
    actions = []
    observations = []
    infos = []
    action_dist = []
    time_taken = 0

    profiler = cProfile.Profile()

    while not done:
        action, _states = policy.predict(obs, deterministic=deterministic)
        if return_dist:
            action_dist.append(get_action_dist_params(policy, obs))
        observations.append(obs)
        actions.append(action)
        if render:
            frames.append(env.render())

        profiler.enable()
        obs, reward, terminated, truncated, info = env.step(action)
        profiler.disable()

        done = terminated or truncated
        rewards.append(reward)
        infos.append(info)

    env.close()

    profiler.dump_stats(f"env_step.prof")
    if return_dist:
        return frames, rewards, actions, observations, infos, action_dist
    else:
        return frames, rewards, actions, observations, infos


def make_movie(frames: list, filename: str, fps=30):
    if "/" in filename:
        path = filename
    else:
        path = os.path.join("videos", filename)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec='libx264',
                            quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)

    # Cleanup
    del frames
    gc.collect()


def build_env(config: dict, baseline: bool, render_mode, **kwargs):
    """Builds a gym environment with the given configuration.

    Args:
        config (dict):
            total_timesteps (int): The total number of timesteps to train the model.
            ctrl_timestep (float): The duration of a single control timestep.
            env_name (str): The name of the environment to build.
            time_limit_sec (float): The time limit for each episode.
            time_aware_obs (bool): Whether to use time-aware observations.
    """

    name2class = {
        "baloo_v0": "BalooV0",
        "baloo_v1": "BalooV1",
        "baloo_v2": "BalooV2",
        "baloo_v3": "BalooV3",
        "baloo_v4": "BalooV4",
        "baloo_v5": "BalooV5",
        "baloo_v6": "BalooV6",
        "baloo_v7": "BalooV7",
        "baloo_v8": "BalooV8",
        "baloo_v9": "BalooV9",
    }

    EnvClass = getattr(
        importlib.import_module(f"baloo_gym.envs.{config['env_name']}"),
        name2class[config["env_name"]])

    resolution = kwargs.get("resolution", 1)

    env = EnvClass(
        render_mode=render_mode,
        camera_name="frontcam",
        ctrl_timestep=config["ctrl_timestep"],
        render_width=int(320 * resolution),
        render_height=int(240 * resolution),
        randomize_initial_height=config["randomize_initial_height"],
        randomize_object_size=config["randomize_object_size"],
        randomize_object_mass=config["randomize_object_mass"],
        object_size=kwargs.get("object_size", None),
        object_mass=kwargs.get("object_mass", None),
        randomize_object_quat=config.get("randomize_object_quat", False),
        randomize_object_pos=config.get("randomize_object_pos", False),
        object_xpos=kwargs.get("object_xpos", None),
        object_zrotation=kwargs.get("object_zrotation", None),
    )

    check_env(env)

    #emit truncation signal at the end of the episode.
    env = TimeLimit(env,
                    max_episode_steps=int(
                        config["time_limit_sec"] / config["ctrl_timestep"], ))

    env = ThreePartRewardWrapper(env, config["reward_selection"])

    if kwargs.get("potential_based_reward", False):
        print("using potential based reward")
        env = PotentialBasedRewardWrapper(
            env, reward_selection=config["reward_selection"])

    # env = CurriculumEnv(env, config["curriculum_selection"])

    if baseline:
        print("Using open-loop baseline policy.")

    #needs monitor after time limit since that emits the done signals.
    if kwargs.get("monitor", False) == True:
        env = Monitor(env, info_keywords=("is_success", ))
    return env


def parallelize_env(args, config, run_folder, save_freq):
    #automatically wraps each environment in a monitor
    vec_env = make_vec_env(
        build_env,
        env_kwargs={
            "config": config,
            "baseline": False,
            "render_mode": "rgb_array",
        },
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=f"new_experiments/{run_folder}/monitor",
        monitor_kwargs={
            'info_keywords': ('is_success', ),
        },
    )

    vec_env = VecVideoRecorder(
        vec_env,
        f"new_experiments/{run_folder}/videos",
        record_video_trigger=lambda x: x % save_freq == 0,
        video_length=30 / config["ctrl_timestep"],
        name_prefix=run_folder,
    )

    return vec_env


def load_or_download_model(args, local_experiment_folder):
    #this needs to download everythign in the new_experiments folder from wandb
    run_path = None
    try:
        #try loading model from local experiments folder
        #find folder containing runid in /home/curtis/baloo/baloo-gym/new_experiments
        for f in os.listdir(local_experiment_folder):
            if args.runid in f:
                run_path = f"{local_experiment_folder}/{f}/"
                print(f"Found run locally in {run_path}")
                break

        if run_path is None:
            raise FileNotFoundError(
                f"{args.runid} not found in {local_experiment_folder}.")
    except:
        #try downloading model from wandb
        print(
            f"{args.runid} not found in {local_experiment_folder}. Trying download from wandb instead..."
        )

        try:
            run = wandb.Api().run(f"curtiscjohnson/ppo_baloo/{args.runid}")
            print(f"Found run on wandb: curtiscjohnson/ppo_baloo/{args.runid}")
        except Exception as e:
            print(f"Error downloading from wandb: {e}")
            raise FileNotFoundError(
                f"{args.runid} not found in {local_experiment_folder} or on wandb."
            )

        print(f"Downloading run files to {local_experiment_folder}...")
        for file in tqdm(run.files()):
            #download everything in the new_experiments folder
            if file.name.startswith("new_experiments/"):
                file.download(root=os.path.dirname(local_experiment_folder),
                              replace=True)
                run_name = Path(file.name).parts[1]

        run_path = os.path.join(local_experiment_folder, run_name)

    #find model_name in the run_path recursively
    model_path = None
    for root, dirs, files in os.walk(run_path):
        for file in files:
            if file.endswith(".zip") and (args.model_name in file):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Model not found in {run_path}. Please specify the model name.")

    print(f"Loading model from {model_path}")
    return model_path
