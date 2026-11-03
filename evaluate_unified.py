#!/usr/bin/env python3
"""
Unified Baloo Controller Evaluation Framework
=============================================
Centralized, standardized benchmarking suite for the Baloo robot platform.
Evaluates four distinct control paradigms:
  1. Open-Loop (OL): Deterministic feedforward trajectory (N=50)
  2. PID: 6-Joint independent slip-correction PID hugger
  3. RL: Trained Proximal Policy Optimization (PPO) neural policy
  4. MPPI: Model Predictive Path Integral controller with parallel rollouts

Standardized Benchmark Protocol:
  - Exact Latin Hypercube Sampled (LHS) boxes (seed=42, 1000 samples)
  - Evaluates under identical disturbance conditions:
      * Condition 1: 0 N (no external perturbation)
      * Condition 2: 100 N downward perturbation on the object
  - Identical physical termination criteria:
      * SUCCESS: Box rises >= 0.5 m above initial resting height
      * TIP:     Box rotates > 80 degrees relative to world z-axis
      * SLIP:    Timeout (max_steps elapsed) without lift or tip
  - Unified hardware metrics:
      * Peak chamber pressure per joint across all trials
      * 99th percentile pressure per joint
      * Discounted returns and episode lengths
      * Video and trajectory plots for detail trials
"""

import os
os.environ["MUJOCO_GL"] = "egl"
import faulthandler
faulthandler.enable()
import sys
import argparse
import ast
import json
import time
import shutil
import io
import contextlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.stats import qmc
import mujoco

# ---------------------------------------------------------------------------
# Workspace Paths & Plugin Setup
# ---------------------------------------------------------------------------
WS_ROOT = Path("/home/radlab/baloo-ws")
GYM_DIR = WS_ROOT / "baloo-gym"
SIM_DIR = WS_ROOT / "baloo-mujoco-sim"

# Add package paths
for p in [str(GYM_DIR / "src"), str(SIM_DIR / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import baloo_mujoco_sim
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_box_position,
    get_box_quat,
    get_joint_pressures,
    set_elevator_cmd,
    set_joint_pressure_commands,
    set_box_size,
    set_box_mass,
)
from baloo_gym.envs.baloo_v9 import BalooV9
from baloo_gym.wrappers.object_perturbation_wrapper import ObjectPerturbationWrapper
from baloo_gym.wrappers.three_part_reward_wrapper import ThreePartRewardWrapper
from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy as BaseOpenLoopPolicy
from baloo_gym.utils.helpers import make_movie

# Suppress PID console spam
import baloo_gym.policies.PIDHugger as pid_hugger_module
pid_hugger_module.OpenLoopHuggerPolicy.print_pressures = lambda self, *args, **kwargs: None
pid_hugger_module.OpenLoopHuggerPolicy.save_logs = lambda self, *args, **kwargs: None


def setup_mujoco_plugins():
    """Ensure all required custom MuJoCo plugins are registered."""
    plugin_dirs = [
        SIM_DIR / "plugin/build/joint_angle_estimator/libjoint_angle_estimator.so",
        SIM_DIR / "plugin/build/ruckig_actuator/libRuckigActuator.so",
        SIM_DIR / "plugin/build/motion_profile_servo/libMotionProfileServo.so",
    ]
    for p in plugin_dirs:
        if p.exists():
            try:
                mujoco.mj_loadPluginLibrary(str(p))
            except mujoco.FatalError as e:
                if "already registered" not in str(e):
                    print(f"Warning loading plugin {p.name}: {e}")
            except Exception:
                pass


setup_mujoco_plugins()

# ---------------------------------------------------------------------------
# Global Constants & Hardware Limits
# ---------------------------------------------------------------------------
CTRL_TIMESTEP = 0.05
MAX_STEPS_DEFAULT = 1200
BOX_RISE_DELTA_THRESHOLD = 0.5  # metres
TIP_ANGLE_COS = np.cos(np.radians(80.0))  # cos(80 deg)

JOINT_LABELS = [
    "Left J0", "Left J1", "Left J2",
    "Right J0", "Right J1", "Right J2",
]
JOINT_SLUGS = [
    "left_j0", "left_j1", "left_j2",
    "right_j0", "right_j1", "right_j2",
]
JOINT_COLOURS = [
    "#4C72B0", "#DD8452", "#55A868",
    "#C44E52", "#8172B2", "#937860",
]
OUTCOME_COLOURS = {
    "success": "#2ca02c",
    "tip":     "#d62728",
    "slip":    "#9467bd",
}
OUTCOME_LABELS = {
    "success": "Success",
    "tip":     "Tipped",
    "slip":    "Slip (timeout)",
}

# Hardware maximum pressure ceilings (kPa)
HARDWARE_PRESSURE_CAPS = {
    "left_j0":  300.0,
    "left_j1":  300.0,
    "left_j2":  300.0,
    "right_j0": 300.0,
    "right_j1": 300.0,
    "right_j2": 300.0,
}

PRESSURE_CAPS = dict(HARDWARE_PRESSURE_CAPS)


def cap_action(a: np.ndarray, pressure_caps: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Clips normalized action to enforce hardware or percentile pressure caps."""
    caps = pressure_caps or PRESSURE_CAPS
    a = np.clip(a.copy(), -1, 1)
    for j_idx, slug in enumerate(JOINT_SLUGS):
        p_max = caps.get(slug, 300.0)
        max_delta = max(0.0, (p_max - 150.0) / 150.0)
        idx = 1 + j_idx * 2
        a[idx] = np.clip(a[idx], -max_delta, max_delta)
        a[idx + 1] = np.clip(a[idx + 1], -max_delta, max_delta)
    return a


def get_step_pressures(model, data) -> np.ndarray:
    """Return (6, 4) array: 6 joints, 4 chambers each."""
    return np.stack([
        get_joint_pressures(model, data, "left",  0),
        get_joint_pressures(model, data, "left",  1),
        get_joint_pressures(model, data, "left",  2),
        get_joint_pressures(model, data, "right", 0),
        get_joint_pressures(model, data, "right", 1),
        get_joint_pressures(model, data, "right", 2),
    ])


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling (LHS) Generator / Loader
# ---------------------------------------------------------------------------
def load_or_generate_lhs_samples(n_samples: int = 1000, seed: int = 42,
                                 filepath: str = "1000_lhs_samples.txt") -> List[Tuple]:
    """Loads 1000 LHS samples from disk or deterministically generates them."""
    candidate_paths = [
        Path(filepath),
        GYM_DIR / filepath,
        SIM_DIR / filepath,
        GYM_DIR / "paper_data/perturbations" / filepath,
    ]
    for p in candidate_paths:
        if p.exists():
            print(f"Loading {n_samples} LHS samples from {p}")
            with open(p, "r") as f:
                samples = [ast.literal_eval(line.strip()) for line in f if line.strip()]
                return samples[:n_samples]

    print(f"Generating {n_samples} LHS samples (seed={seed})...")
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    raw = sampler.random(n=n_samples)
    x_s = np.interp(raw[:, 0], [0, 1], [0.2, 0.6])
    y_s = np.interp(raw[:, 1], [0, 1], [0.2, 0.6])
    z_s = np.interp(raw[:, 2], [0, 1], [0.5, 1.25])
    m_s = np.interp(raw[:, 3], [0, 1], [0.5, 10.0])
    xp_s = np.interp(raw[:, 4], [0, 1], [-0.1, 0.1])
    r_s = np.interp(raw[:, 5], [0, 1], [-np.pi / 3, np.pi / 3])
    combos = [(float(x), float(y), float(z), float(m), float(xp), float(r))
              for x, y, z, m, xp, r in zip(x_s, y_s, z_s, m_s, xp_s, r_s)]

    try:
        out_p = GYM_DIR / filepath
        with open(out_p, "w") as f:
            for c in combos:
                f.write(f"{c}\n")
        print(f"Saved generated samples to {out_p}")
    except Exception as e:
        print(f"Could not persist LHS file: {e}")

    return combos[:n_samples]


# ---------------------------------------------------------------------------
# Polymorphic Controller Adapters
# ---------------------------------------------------------------------------
class BaseControllerAdapter:
    """Base interface for all evaluated controllers."""
    def __init__(self, name: str):
        self.name = name

    def reset(self, env: BalooV9, obs: np.ndarray, initial_z: float, box_params: tuple):
        pass

    def predict(self, env: BalooV9, obs: np.ndarray, step_idx: int) -> np.ndarray:
        raise NotImplementedError

    def cleanup(self):
        pass


class OpenLoopAdapter(BaseControllerAdapter):
    """Open-Loop feedforward controller."""
    def __init__(self, n_steps: int = 50):
        super().__init__("open_loop")
        self.n_steps = n_steps
        self.policy = None

    def reset(self, env: BalooV9, obs: np.ndarray, initial_z: float, box_params: tuple):
        self.policy = BaseOpenLoopPolicy(N=self.n_steps)

    def predict(self, env: BalooV9, obs: np.ndarray, step_idx: int) -> np.ndarray:
        action, _ = self.policy.predict(obs)
        return action


# 6-Joint independently tuned PID parameters from EvaluatePID_1000WithHistogramsAnd6PIDControllers.py
PER_JOINT_PID_PARAMS = {
    'left_j0': {
        'kp': 113.170, 'ki': 19.274, 'kd': 8.996,
        'threshold': 0.0607, 'correction_max': 149.3,
    },
    'left_j1': {
        'kp': 171.111, 'ki': 3.250, 'kd': 1.329,
        'threshold': 0.0900, 'correction_max': 149.9,
    },
    'left_j2': {
        'kp': 119.818, 'ki': 14.956, 'kd': 6.892,
        'threshold': 0.0886, 'correction_max': 13.0,
    },
    'right_j0': {
        'kp': 33.059, 'ki': 5.563, 'kd': 1.370,
        'threshold': 0.0728, 'correction_max': 149.4,
    },
    'right_j1': {
        'kp': 58.827, 'ki': 8.351, 'kd': 4.333,
        'threshold': 0.0938, 'correction_max': 126.2,
    },
    'right_j2': {
        'kp': 51.718, 'ki': 1.421, 'kd': 8.811,
        'threshold': 0.0291, 'correction_max': 37.1,
    },
}


class PIDAdapter(BaseControllerAdapter):
    """6-Joint Independent Slip-Correction PID Controller using tuned per-joint parameters and pressure caps."""
    def __init__(self, n_steps: int = 50, per_joint_pid_params: Optional[dict] = None, pressure_caps: Optional[dict] = None):
        super().__init__("pid")
        self.n_steps = n_steps
        import copy
        self.per_joint_pid_params = copy.deepcopy(per_joint_pid_params or PER_JOINT_PID_PARAMS)
        self.pressure_caps = pressure_caps
        if self.pressure_caps is not None:
            for slug, cap_val in self.pressure_caps.items():
                if slug in self.per_joint_pid_params:
                    # In SlipCorrectionPID, extra correction is added to antagonist (base 150)
                    self.per_joint_pid_params[slug]['correction_max'] = max(0.0, float(cap_val) - 150.0)
        self.policy = None

    def reset(self, env: BalooV9, obs: np.ndarray, initial_z: float, box_params: tuple):
        self.policy = pid_hugger_module.OpenLoopHuggerPolicy(
            N=self.n_steps,
            per_joint_pid_params=self.per_joint_pid_params
        )
        self.policy.restart()

    def predict(self, env: BalooV9, obs: np.ndarray, step_idx: int) -> np.ndarray:
        box_body_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        exact_box_z = env.unwrapped.data.xipos[box_body_id][2]
        with contextlib.redirect_stdout(io.StringIO()):
            action, _ = self.policy.predict(obs, exact_box_z=exact_box_z)
        if self.pressure_caps is not None:
            action = cap_action(action, self.pressure_caps)
        return action


class RLAdapter(BaseControllerAdapter):
    """Trained PPO Reinforcement Learning Policy."""
    def __init__(self, model_path: str):
        super().__init__("rl")
        from stable_baselines3 import PPO
        print(f"Loading RL checkpoint from: {model_path}")
        self.model = PPO.load(model_path)

    def reset(self, env: BalooV9, obs: np.ndarray, initial_z: float, box_params: tuple):
        pass

    def predict(self, env: BalooV9, obs: np.ndarray, step_idx: int) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return action


# MPPI Rollout Worker Globals
_worker_model = None

def _mppi_worker_init(xml_path: str, object_size: tuple, object_mass: float):
    global _worker_model
    try:
        spec = mujoco.MjSpec.from_file(xml_path)
    except TypeError:
        spec = mujoco.MjSpec()
        spec.from_file(xml_path)
    if object_size is not None:
        set_box_size(spec, *object_size)
    if object_mass is not None:
        set_box_mass(spec, object_mass)
    _worker_model = spec.compile()

def _mppi_rollout_worker(args):
    global _worker_model
    data_state, action_seq, nominal_seq, initial_z, w_guide, w_height, caps = args
    data = mujoco.MjData(_worker_model)
    mujoco.mj_setState(_worker_model, data, data_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    total_cost = 0.0
    for t in range(len(action_seq)):
        # Apply action with caps enforced
        a = cap_action(action_seq[t], caps)
        pressure_deltas = a[1:] * 150.0
        set_elevator_cmd(_worker_model, data, (a[0] + 1) / 2 * (-900.0))
        for side_idx, side in enumerate(['left', 'right']):
            for j in range(3):
                idx = side_idx * 6 + j * 2
                set_joint_pressure_commands(_worker_model, data, side, j, np.array([
                    150.0 + pressure_deltas[idx],
                    150.0 - pressure_deltas[idx],
                    150.0 + pressure_deltas[idx + 1],
                    150.0 - pressure_deltas[idx + 1],
                ]))
        mujoco.mj_step(_worker_model, data)

        # Guide cost (Curtis Eq 9.6)
        diff = a - nominal_seq[t]
        total_cost += -w_guide * np.exp(-0.5 * np.dot(diff, diff))

    # Task reward at terminal state
    box_z = get_box_position(_worker_model, data)[2]
    quat = get_box_quat(_worker_model, data)
    rot = R.from_quat(np.roll(quat, -1))
    box_z_axis = rot.apply([0, 0, 1])
    tipped = np.degrees(np.arccos(np.clip(np.dot(box_z_axis, [0, 0, 1]), -1, 1))) > 80.0
    lifted = box_z > (initial_z + 0.5)

    r = w_height * (box_z - initial_z)
    if tipped:
        r += 0.0  # R_TIP
    if lifted:
        r += 0.0  # R_LIFT
    total_cost -= r
    return total_cost


class MPPIAdapter(BaseControllerAdapter):
    """Model Predictive Path Integral (MPPI) Controller with Parallel Rollouts and Pressure Caps."""
    def __init__(self, n_samples: int = 32, horizon: int = 8, sigma: float = 0.15,
                 w_guide: float = 0.10, lambda_temp: float = 1.0, n_workers: int = 16,
                 pressure_caps: Optional[dict] = None):
        super().__init__("mppi")
        self.n_samples = n_samples
        self.horizon = horizon
        self.sigma = sigma
        self.w_guide = w_guide
        self.w_height = 0.0
        self.lambda_temp = lambda_temp
        self.n_workers = n_workers
        self.pressure_caps = pressure_caps
        self.xml_path = str(baloo_mujoco_sim.XML_PATH)

        self.ol_policy = None
        self.nominal_actions = None
        self.pool = None
        self.initial_z = 0.0
        self.rng = np.random.default_rng(42)

    def _compute_nominal_actions(self, object_size, object_mass, object_xpos, object_zrot, max_steps):
        nom_env = BalooV9(
            render_mode=None, camera_name='frontcam', ctrl_timestep=CTRL_TIMESTEP,
            randomize_initial_height=False, randomize_object_size=False,
            randomize_object_mass=False, object_size=list(object_size),
            object_mass=object_mass, object_xpos=object_xpos, object_zrotation=object_zrot
        )
        nom_obs, _ = nom_env.reset()
        policy = BaseOpenLoopPolicy(N=50)
        actions = []
        for _ in range(max_steps):
            a, _ = policy.predict(nom_obs)
            actions.append(a.copy())
            nom_obs, _, t, tr, _ = nom_env.step(a)
        nom_env.close()
        return np.array(actions)

    def reset(self, env: BalooV9, obs: np.ndarray, initial_z: float, box_params: tuple):
        self.cleanup()
        x, y, z, m, xp, r = box_params
        self.initial_z = initial_z
        self.ol_policy = BaseOpenLoopPolicy(N=50)
        self.nominal_actions = self._compute_nominal_actions((x, y, z), m, xp, r, MAX_STEPS_DEFAULT)

        from multiprocessing import Pool
        self.pool = Pool(
            self.n_workers,
            initializer=_mppi_worker_init,
            initargs=(self.xml_path, (x, y, z), m)
        )

    def predict(self, env: BalooV9, obs: np.ndarray, step_idx: int) -> np.ndarray:
        # Grasping phase: follow Open-Loop policy until LIFT state
        if self.ol_policy.state != "LIFT":
            a, _ = self.ol_policy.predict(obs)
            return cap_action(a, self.pressure_caps)

        # MPPI state sampling & rollout phase
        model = env.unwrapped.model
        data = env.unwrapped.data
        state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data_state = np.zeros(state_size)
        mujoco.mj_getState(model, data, data_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        t_end = min(step_idx + self.horizon, len(self.nominal_actions))
        nom_window = self.nominal_actions[step_idx:t_end]

        perturbations = self.rng.normal(0, self.sigma, size=(self.n_samples, len(nom_window), 13))
        candidates = nom_window[None] + perturbations

        args = [
            (data_state, candidates[i], nom_window, self.initial_z, self.w_guide, self.w_height, self.pressure_caps)
            for i in range(self.n_samples)
        ]
        costs = np.array(self.pool.map(_mppi_rollout_worker, args))

        weights = np.exp(-(costs - costs.min()) / self.lambda_temp)
        weights /= weights.sum()
        best_action = np.sum(weights[:, None] * candidates[:, 0, :], axis=0)
        return cap_action(best_action, self.pressure_caps)

    def cleanup(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None


# ---------------------------------------------------------------------------
# Single Trial Execution
# ---------------------------------------------------------------------------
def run_single_trial(
    adapter: BaseControllerAdapter,
    box_params: tuple,
    perturb: bool = False,
    perturbation_magnitude: float = 100.0,
    render: bool = False,
    max_steps: int = MAX_STEPS_DEFAULT,
    gamma: float = 0.99,
) -> Dict[str, Any]:
    """Runs a single episode for the given controller adapter under defined conditions."""
    x, y, z, mass, xpos, rot = box_params

    env = BalooV9(
        render_mode="rgb_array" if render else None,
        camera_name="frontcam",
        ctrl_timestep=CTRL_TIMESTEP,
        render_width=320,
        render_height=240,
        randomize_initial_height=False,
        randomize_object_size=False,
        randomize_object_mass=False,
        object_size=[x, y, z],
        object_mass=mass,
        object_xpos=xpos,
        object_zrotation=rot,
    )
    env = ThreePartRewardWrapper(env, reward_selection=['copy_baseline'])

    if perturb:
        env = ObjectPerturbationWrapper(env)
        env.perturbation_magnitude = perturbation_magnitude

    obs, _ = env.reset()
    model = env.unwrapped.model
    data = env.unwrapped.data
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    initial_z = float(data.xipos[box_body_id][2])

    adapter.reset(env, obs, initial_z, box_params)

    frames = []
    rewards = []
    actions = []
    observations = []
    pressures_detail = []
    joint_max = np.full(6, -np.inf)
    outcome = "slip"
    step_idx = 0

    try:
        while step_idx < max_steps:
            if render:
                frames.append(env.render())

            action = adapter.predict(env, obs, step_idx)
            actions.append(action.copy())
            observations.append(obs.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            # Record Chamber Pressures
            step_p = get_step_pressures(model, data)  # (6, 4)
            joint_max = np.maximum(joint_max, step_p.max(axis=1))
            if render:
                pressures_detail.append(step_p.copy())

            # Evaluate Standardized Termination Criteria
            current_z = float(data.xipos[box_body_id][2])
            rot_mat = data.xmat[box_body_id].reshape(3, 3)

            if rot_mat[2, 2] < TIP_ANGLE_COS:
                outcome = "tip"
                break
            elif current_z >= (initial_z + BOX_RISE_DELTA_THRESHOLD):
                outcome = "success"
                break
            elif terminated or truncated:
                outcome = "slip"
                break

            step_idx += 1

    finally:
        adapter.cleanup()
        env.close()
        del env
        import gc
        gc.collect()

    # Calculate Discounted Return
    discounted_return = float(sum((gamma ** i) * r for i, r in enumerate(rewards)))

    return {
        "outcome": outcome,
        "success": outcome == "success",
        "tip": outcome == "tip",
        "slip": outcome == "slip",
        "episode_len": len(rewards),
        "discounted_return": discounted_return,
        "joint_max": joint_max,
        "xsize": x,
        "ysize": y,
        "zsize": z,
        "mass_kg": mass,
        "xpos": xpos,
        "rotation_deg": float(np.degrees(rot)),
        "frames": frames if render else None,
        "actions": np.array(actions) if render else None,
        "observations": np.array(observations) if render else None,
        "rewards": np.array(rewards) if render else None,
        "pressures_detail": np.array(pressures_detail) if render else None,
    }


# ---------------------------------------------------------------------------
# Detailed Rollout Plotting & Media Generation
# ---------------------------------------------------------------------------
def save_trial_detail(trial_data: Dict[str, Any], trial_idx: int, out_dir: Path):
    """Generates MP4 video and detailed sensor/pressure plots for a rollout."""
    rollout_dir = out_dir / f"rollout_{trial_idx:04d}_{trial_data['outcome']}"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    # 1. MP4 Video
    if trial_data["frames"]:
        video_path = str(rollout_dir / "rollout.mp4")
        make_movie(trial_data["frames"], video_path, fps=int(1.0 / CTRL_TIMESTEP))

    # 2. Rewards trajectory
    if trial_data["rewards"] is not None and len(trial_data["rewards"]) > 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(trial_data["rewards"], color="#1f77b4", lw=1.5)
        ax.set_title(f"Trial {trial_idx} Rewards (Outcome: {trial_data['outcome'].upper()})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(rollout_dir / "rewards.png", dpi=150)
        plt.close(fig)

    # 3. Actions trajectory
    if trial_data["actions"] is not None and len(trial_data["actions"]) > 0:
        actions = trial_data["actions"]
        fig, axs = plt.subplots(4, 4, figsize=(14, 10), sharex=True)
        axs = axs.ravel()
        labels = ["Elevator"] + [f"J{j//2}_{'Ag' if j%2==0 else 'Ant'}_{'L' if j<6 else 'R'}" for j in range(12)]
        for i in range(13):
            axs[i].plot(actions[:, i], color="#2ca02c", lw=1.2)
            axs[i].set_title(labels[i], fontsize=9)
            axs[i].set_ylim(-1.05, 1.05)
            axs[i].grid(True, alpha=0.3)
        for i in range(13, 16):
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(rollout_dir / "actions.png", dpi=150)
        plt.close(fig)

    # 4. Joint Pressures Trajectory
    if trial_data["pressures_detail"] is not None and len(trial_data["pressures_detail"]) > 0:
        p_data = trial_data["pressures_detail"]  # (T, 6, 4)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
        for j, (ax, label) in enumerate(zip(axes.ravel(), JOINT_LABELS)):
            for chamber in range(4):
                ax.plot(p_data[:, j, chamber], label=f"Chamber {chamber+1}", alpha=0.8, lw=1.0)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Pressure (kPa)")
            ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        plt.savefig(rollout_dir / "pressures_trajectory.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Publication Quality Summary Histograms
# ---------------------------------------------------------------------------
def plot_summary_histograms(
    max_pressures: np.ndarray,
    outcomes: List[str],
    controller_name: str,
    condition_name: str,
    filename: Path,
    n_bins: int = 30
):
    """Generates 2x3 grid of max joint pressures stacked by outcome."""
    outcomes_arr = np.asarray(outcomes)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Peak Chamber Pressure per Joint — {controller_name.upper()} ({condition_name})",
        fontsize=13, fontweight="bold"
    )

    for j, (ax, label, colour) in enumerate(zip(axes.flat, JOINT_LABELS, JOINT_COLOURS)):
        data = max_pressures[:, j]
        outcome_keys = ["success", "tip", "slip"]
        _, bin_edges = np.histogram(data, bins=n_bins)
        bottoms = np.zeros(n_bins)

        for key in outcome_keys:
            mask = outcomes_arr == key
            if not mask.any():
                continue
            counts, _ = np.histogram(data[mask], bins=bin_edges)
            ax.bar(
                bin_edges[:-1], counts, width=np.diff(bin_edges),
                bottom=bottoms, align="edge",
                color=OUTCOME_COLOURS[key], alpha=0.85,
                label=f"{OUTCOME_LABELS[key]} (n={mask.sum()})",
            )
            bottoms += counts

        p99 = np.percentile(data, 99)
        ax.axvline(data.mean(), color="black", linestyle="--", lw=1.3, label=f"Mean: {data.mean():.1f} kPa")
        ax.axvline(p99, color="#d62728", linestyle=":", lw=1.5, label=f"p99: {p99:.1f} kPa")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Max Pressure (kPa)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.tick_params(labelsize=7)

        n_succ = (outcomes_arr == "success").sum()
        ax.text(
            0.02, 0.95,
            f"N={len(data)} | {n_succ/len(data)*100:.1f}% success\np99={p99:.1f} kPa",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )

    for row, arm in enumerate(["LEFT ARM", "RIGHT ARM"]):
        fig.text(0.005, 0.75 - row * 0.5, arm, va="center", ha="left",
                 fontsize=9, fontweight="bold", rotation=90, color="#333333")

    legend_patches = [
        Patch(facecolor=OUTCOME_COLOURS[k], alpha=0.85, label=OUTCOME_LABELS[k])
        for k in ["success", "tip", "slip"]
    ]
    fig.legend(
        handles=legend_patches, loc="lower center", ncol=3,
        fontsize=9, framealpha=0.8, title="Outcome colour key",
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary histogram saved -> {filename}")


# ---------------------------------------------------------------------------
# Controller Benchmark Runner
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# RL Pressure Percentile Extraction & Limit Loading
# ---------------------------------------------------------------------------
def extract_rl_pressure_percentiles(out_base: Path, output_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Extracts 90th, 95th, and 99th percentile peak chamber pressures per joint from RL evaluations.
    Saves to JSON and YAML files for consumption by PID and MPPI.
    """
    out_json = output_file or (out_base / "pressure_limits.json")
    out_yaml = out_json.with_suffix(".yaml")

    nopert_csv = out_base / "rl_nopert" / "trials.csv"
    pert100_csv = out_base / "rl_pert100" / "trials.csv"

    dfs = {}
    if nopert_csv.exists():
        try:
            dfs["nopert"] = pd.read_csv(nopert_csv)
        except Exception:
            pass
    if pert100_csv.exists():
        try:
            dfs["pert100"] = pd.read_csv(pert100_csv)
        except Exception:
            pass

    if not dfs:
        return load_pressure_limits(None, out_base)

    by_condition = {}
    for cond, df in dfs.items():
        by_condition[cond] = {}
        for p in [90, 95, 99]:
            by_condition[cond][f"p{p}"] = {
                slug: round(float(np.percentile(df[f"max_p_{slug}"], p)), 1)
                for slug in JOINT_SLUGS
                if f"max_p_{slug}" in df.columns
            }

    all_df = pd.concat(list(dfs.values()), ignore_index=True)
    combined = {}
    for p in [90, 95, 99]:
        combined[f"p{p}"] = {
            slug: round(float(np.percentile(all_df[f"max_p_{slug}"], p)), 1)
            for slug in JOINT_SLUGS
            if f"max_p_{slug}" in all_df.columns
        }

    data = {
        "source": "rl",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "percentiles": combined,
        "by_condition": by_condition,
    }

    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    try:
        import yaml
        with open(out_yaml, "w") as yf:
            yaml.dump(data, yf, default_flow_style=False, sort_keys=False)
    except Exception:
        pass

    print(f"\n📊 Extracted RL Pressure Percentiles -> {out_json} & {out_yaml}")
    for p in ["p90", "p95", "p99"]:
        print(f"  [{p.upper()}] {combined[p]}")

    return data


def load_pressure_limits(file_path: Optional[str] = None, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Loads pressure limits from json or yaml, extracts from existing RL runs, or returns empirical defaults."""
    target_path = None
    if file_path:
        target_path = Path(file_path)
    elif base_dir:
        cand_json = base_dir / "pressure_limits.json"
        cand_yaml = base_dir / "pressure_limits.yaml"
        if cand_json.exists():
            target_path = cand_json
        elif cand_yaml.exists():
            target_path = cand_yaml

    if target_path and target_path.exists():
        with open(target_path, "r") as f:
            if target_path.suffix in [".yaml", ".yml"]:
                import yaml
                return yaml.safe_load(f)
            else:
                return json.load(f)

    if base_dir:
        cand_rl = base_dir / "rl_nopert" / "trials.csv"
        if cand_rl.exists():
            return extract_rl_pressure_percentiles(base_dir)

    # Empirical fallback limits derived from Curtis's RL paper model
    return {
        "source": "curtis_rl_empirical_default",
        "percentiles": {
            "p90": {"left_j0": 300.0, "left_j1": 300.0, "left_j2": 162.1, "right_j0": 300.0, "right_j1": 276.5, "right_j2": 160.7},
            "p95": {"left_j0": 300.0, "left_j1": 300.0, "left_j2": 166.8, "right_j0": 300.0, "right_j1": 279.5, "right_j2": 161.3},
            "p99": {"left_j0": 300.0, "left_j1": 300.0, "left_j2": 170.5, "right_j0": 300.0, "right_j1": 282.2, "right_j2": 162.9},
        },
        "by_condition": {
            "nopert": {
                "p90": {"left_j0": 300.0, "left_j1": 300.0, "left_j2": 161.8, "right_j0": 300.0, "right_j1": 271.9, "right_j2": 160.9},
                "p95": {"left_j0": 300.0, "left_j1": 300.0, "left_j2": 162.4, "right_j0": 300.0, "right_j1": 278.6, "right_j2": 161.7},
                "p99": {"left_j0": 300.0, "left_j1": 300.0, "left_j2": 169.2, "right_j0": 300.0, "right_j1": 282.5, "right_j2": 162.9},
            },
            "pert100": {
                "p90": {"left_j0": 300.0, "left_j1": 300.0, "left_j2": 162.8, "right_j0": 300.0, "right_j1": 278.2, "right_j2": 160.5},
                "p95": {"left_j0": 300.0, "left_j1": 300.0, "left_j2": 168.8, "right_j0": 300.0, "right_j1": 279.7, "right_j2": 161.2},
                "p99": {"left_j0": 300.0, "left_j1": 300.0, "left_j2": 171.0, "right_j0": 300.0, "right_j1": 282.1, "right_j2": 162.5},
            }
        }
    }


def get_caps_for_task(limits_data: Dict[str, Any], percentile: Optional[str], condition: str) -> Optional[Dict[str, float]]:
    """Resolves per-joint pressure caps for a specific percentile and condition."""
    if not percentile:
        return None
    p_key = percentile.lower()
    by_cond = limits_data.get("by_condition", {})
    if condition in by_cond and p_key in by_cond[condition]:
        return by_cond[condition][p_key]
    percentiles = limits_data.get("percentiles", {})
    if p_key in percentiles:
        return percentiles[p_key]
    return None


# ---------------------------------------------------------------------------
# Controller Benchmark Runner
# ---------------------------------------------------------------------------
def evaluate_controller(
    controller_name: str,
    adapter: BaseControllerAdapter,
    condition: str,
    lhs_samples: List[Tuple],
    out_base_dir: Path,
    num_trials: int = 1000,
    num_detail: int = 5,
    max_steps: int = MAX_STEPS_DEFAULT,
    percentile: Optional[str] = None,
    pressure_caps: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Runs full batch evaluation for a specific controller under a given condition and percentile limit."""
    perturb = condition == "pert100"
    cond_label = "100N Perturbations" if perturb else "No Perturbations (0N)"
    cond_tag = "pert100" if perturb else "nopert"

    limit_tag = percentile.lower() if percentile else None
    if limit_tag:
        folder_name = f"{controller_name}_{limit_tag}_{cond_tag}"
        limit_display = limit_tag.upper()
    else:
        folder_name = f"{controller_name}_{cond_tag}"
        limit_display = "N/A"

    out_dir = out_base_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_dir / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r") as sf:
                existing = json.load(sf)
            if existing.get("num_trials", 0) >= num_trials:
                print(f"\n  ⚡ [{folder_name.upper()}] already completed ({existing['num_trials']} trials found in {summary_file}). Skipping!")
                return existing
        except Exception:
            pass

    limit_hdr = f" | Limit: {limit_display}" if limit_display != "N/A" else ""
    print(f"\n{'='*75}")
    print(f"  Benchmarking: {controller_name.upper()}{limit_hdr} | Condition: {cond_label}")
    print(f"  Total Trials: {num_trials} | Detail Trials (Videos/Plots): {num_detail}")
    print(f"{'='*75}")

    records = []
    all_joint_max = []
    outcomes = []
    returns = []
    lengths = []

    pbar_desc = f"{folder_name}"
    pbar = tqdm(total=num_trials, desc=pbar_desc, unit="trial")
    for i in range(num_trials):
        box_params = lhs_samples[i % len(lhs_samples)]
        is_detail = i < num_detail

        t0 = time.time()
        res = run_single_trial(
            adapter=adapter,
            box_params=box_params,
            perturb=perturb,
            perturbation_magnitude=100.0,
            render=is_detail,
            max_steps=max_steps,
        )
        elapsed = time.time() - t0

        all_joint_max.append(res["joint_max"])
        outcomes.append(res["outcome"])
        returns.append(res["discounted_return"])
        lengths.append(res["episode_len"])

        rec = {
            "trial_idx": i,
            "controller": controller_name,
            "limit": limit_display,
            "condition": cond_tag,
            "outcome": res["outcome"],
            "success": res["success"],
            "tip": res["tip"],
            "slip": res["slip"],
            "episode_len": res["episode_len"],
            "discounted_return": res["discounted_return"],
            "mass_kg": res["mass_kg"],
            "xsize": res["xsize"],
            "ysize": res["ysize"],
            "zsize": res["zsize"],
            "xpos": res["xpos"],
            "rotation_deg": res["rotation_deg"],
            "elapsed_sec": round(elapsed, 2),
        }
        for j_idx, slug in enumerate(JOINT_SLUGS):
            rec[f"max_p_{slug}"] = float(res["joint_max"][j_idx])
        records.append(rec)

        if is_detail:
            save_trial_detail(res, i, out_dir)

        pbar.set_postfix({
            "succ": f"{sum(r['success'] for r in records)/(i+1)*100:.1f}%",
            "outcome": res["outcome"]
        })
        pbar.update(1)

    pbar.close()

    # Convert to array & DataFrame
    all_joint_max = np.array(all_joint_max)
    df_trials = pd.DataFrame(records)
    csv_path = out_dir / "trials.csv"
    df_trials.to_csv(csv_path, index=False)

    # Compute Statistics
    n_total = len(records)
    n_succ = sum(r["success"] for r in records)
    n_tip = sum(r["tip"] for r in records)
    n_slip = sum(r["slip"] for r in records)

    succ_rate = (n_succ / n_total) * 100.0
    tip_rate = (n_tip / n_total) * 100.0
    slip_rate = (n_slip / n_total) * 100.0

    p90_pressures = {slug: float(np.percentile(all_joint_max[:, j], 90))
                     for j, slug in enumerate(JOINT_SLUGS)}
    p95_pressures = {slug: float(np.percentile(all_joint_max[:, j], 95))
                     for j, slug in enumerate(JOINT_SLUGS)}
    p99_pressures = {slug: float(np.percentile(all_joint_max[:, j], 99))
                     for j, slug in enumerate(JOINT_SLUGS)}

    # Plot summary histogram
    hist_file = out_dir / f"pressure_histograms_{folder_name}.png"
    plot_summary_histograms(
        max_pressures=all_joint_max,
        outcomes=outcomes,
        controller_name=controller_name,
        condition_name=f"{cond_label} ({limit_display})",
        filename=hist_file,
    )

    summary_data = {
        "controller": controller_name,
        "limit": limit_display,
        "condition": cond_tag,
        "num_trials": n_total,
        "success_rate_pct": round(succ_rate, 2),
        "tip_rate_pct": round(tip_rate, 2),
        "slip_rate_pct": round(slip_rate, 2),
        "mean_return": round(float(np.mean(returns)), 2),
        "std_return": round(float(np.std(returns)), 2),
        "mean_episode_len": round(float(np.mean(lengths)), 1),
        "applied_pressure_caps": pressure_caps,
        "p90_pressures_kpa": p90_pressures,
        "p95_pressures_kpa": p95_pressures,
        "p99_pressures_kpa": p99_pressures,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Print summary
    print(f"\n{'='*55}")
    print(f"=== {controller_name.upper()} [{limit_display}] ({cond_tag}) Results ({n_total} Trials) ===")
    print(f"{'='*55}")
    print(f"  Success Rate : {succ_rate:.2f}% ({n_succ}/{n_total})")
    print(f"  Tip Rate     : {tip_rate:.2f}% ({n_tip}/{n_total})")
    print(f"  Slip Rate    : {slip_rate:.2f}% ({n_slip}/{n_total})")
    print(f"  Mean Return  : {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"  Mean Length  : {np.mean(lengths):.1f} steps")
    print(f"  p99 Pressures (kPa):")
    for slug, p in p99_pressures.items():
        print(f"    {slug:10}: {p:.1f} kPa")
    print(f"{'='*55}\n")

    return summary_data


# ---------------------------------------------------------------------------
# Cross-Controller Comparison Table & Aggregator
# ---------------------------------------------------------------------------
def generate_comparison_report(out_dir: Path):
    """
    Compiles all evaluated controllers into publication-ready CSV and Markdown tables.
    Always scans out_dir for all summary.json files so that no controller is ever overwritten or lost.
    """
    all_summaries = []
    for p in sorted(out_dir.glob("*/summary.json")):
        try:
            with open(p, "r") as sf:
                s = json.load(sf)
            if "controller" in s and "condition" in s:
                all_summaries.append(s)
        except Exception:
            pass

    if not all_summaries:
        return

    # Deterministic publication order: OPEN_LOOP -> RL -> PID -> MPPI, limit N/A -> p90 -> p95 -> p99, condition nopert -> pert100
    ctrl_order = {"open_loop": 0, "rl": 1, "pid": 2, "mppi": 3}
    limit_order = {"n/a": 0, "none": 0, "hardware": 0, "p90": 1, "p95": 2, "p99": 3}
    cond_order = {"nopert": 0, "pert100": 1}

    all_summaries.sort(key=lambda s: (
        ctrl_order.get(s.get("controller", "").lower(), 99),
        limit_order.get(str(s.get("limit", s.get("percentile", "n/a"))).lower(), 99),
        cond_order.get(s.get("condition", "").lower(), 99)
    ))

    table_rows = []
    for s in all_summaries:
        limit_val = s.get("limit", s.get("percentile", "N/A"))
        limit_str = limit_val.upper() if limit_val else "N/A"
        row = {
            "Controller": s["controller"].upper(),
            "Limit": limit_str,
            "Condition": s["condition"],
            "Trials": s["num_trials"],
            "Success Rate (%)": f"{s['success_rate_pct']:.1f}%",
            "Tip Rate (%)": f"{s['tip_rate_pct']:.1f}%",
            "Slip Rate (%)": f"{s['slip_rate_pct']:.1f}%",
            "Mean Return": f"{s['mean_return']:.2f}",
            "Mean Steps": f"{s['mean_episode_len']:.1f}",
            "p99 L_J0": f"{s['p99_pressures_kpa']['left_j0']:.1f}",
            "p99 L_J1": f"{s['p99_pressures_kpa']['left_j1']:.1f}",
            "p99 L_J2": f"{s['p99_pressures_kpa']['left_j2']:.1f}",
            "p99 R_J0": f"{s['p99_pressures_kpa']['right_j0']:.1f}",
            "p99 R_J1": f"{s['p99_pressures_kpa']['right_j1']:.1f}",
            "p99 R_J2": f"{s['p99_pressures_kpa']['right_j2']:.1f}",
        }
        table_rows.append(row)

    df_comp = pd.DataFrame(table_rows)
    csv_file = out_dir / "unified_comparison.csv"
    df_comp.to_csv(csv_file, index=False)

    md_table = df_comp.to_markdown(index=False)
    md_file = out_dir / "unified_comparison.md"
    with open(md_file, "w") as f:
        f.write("# Baloo Controllers Unified Benchmark Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(md_table + "\n")

    print("\n" + "="*85)
    print("🏆 UNIFIED CONTROLLER BENCHMARK COMPARISON TABLE")
    print("="*85)
    print(md_table)
    print("="*85)
    print(f"Report saved to: {md_file} & {csv_file}\n")


# ---------------------------------------------------------------------------
# Main CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Baloo Controller Evaluation Framework")
    parser.add_argument(
        "--controller",
        choices=["open_loop", "pid", "rl", "mppi", "all"],
        default="all",
        help="Controller to benchmark (or 'all' for complete evaluation suite)"
    )
    parser.add_argument(
        "--condition",
        choices=["nopert", "pert100", "both"],
        default="both",
        help="Perturbation conditions to evaluate"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1000,
        help="Number of trials per condition (default: 1000)"
    )
    parser.add_argument(
        "--num_detail",
        type=int,
        default=5,
        help="Number of trials with recorded video and trajectory plots (default: 5)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS_DEFAULT,
        help=f"Timeout limit in control steps (default: {MAX_STEPS_DEFAULT})"
    )
    parser.add_argument(
        "--lhs_file",
        type=str,
        default="1000_lhs_samples.txt",
        help="LHS samples filename or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results/unified",
        help="Root output directory for results"
    )
    parser.add_argument(
        "--rl_model_path",
        type=str,
        default=str(GYM_DIR / "new_experiments/lucky-silence-739-k1b5sgl8/recent_model/model.zip"),
        help="Path to trained RL model (.zip)"
    )
    parser.add_argument(
        "--mppi_workers",
        type=int,
        default=16,
        help="Worker count for MPPI parallel rollouts (default: 16)"
    )
    parser.add_argument(
        "--mppi_samples",
        type=int,
        default=32,
        help="Sample count per MPPI step (default: 32)"
    )
    parser.add_argument(
        "--mppi_horizon",
        type=int,
        default=8,
        help="Horizon depth for MPPI (default: 8)"
    )
    parser.add_argument(
        "--percentile",
        type=str,
        default="all",
        choices=["p90", "p95", "p99", "all"],
        help="Actuation limit percentile for PID/MPPI (default: all -> evaluates p90, p95, p99)"
    )
    parser.add_argument(
        "--pressure_limits_file",
        type=str,
        default="",
        help="Path to custom JSON or YAML file containing pressure percentile limits"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help="Scan --output_dir for existing summary.json files and regenerate unified comparison table"
    )
    parser.add_argument(
        "--subtask",
        action="store_true",
        default=False,
        help="Internal flag: indicates running within an isolated worker subprocess"
    )

    args = parser.parse_args()

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # If --merge is requested, aggregate existing results immediately and print full table
    if args.merge:
        generate_comparison_report(out_base)
        return

    # 1. Load LHS Samples
    lhs_samples = load_or_generate_lhs_samples(args.num_trials, filepath=args.lhs_file)
    print(f"Loaded {len(lhs_samples)} deterministic LHS boxes for evaluation.")

    # 2. Resolve Conditions
    conditions = ["nopert", "pert100"] if args.condition == "both" else [args.condition]

    # 3. Resolve Controllers & Tasks
    controllers_to_run = ["rl", "open_loop", "pid", "mppi"] if args.controller == "all" else [args.controller]

    tasks = []
    # RL always evaluated FIRST so we can extract its 90th, 95th, 99th percentiles
    if "rl" in controllers_to_run:
        for cond in conditions:
            tasks.append({"controller": "rl", "percentile": None, "condition": cond})

    # Open Loop (reference baseline)
    if "open_loop" in controllers_to_run:
        for cond in conditions:
            tasks.append({"controller": "open_loop", "percentile": None, "condition": cond})

    # PID (p90, p95, p99 with and without perturbations -> 6 evaluations)
    if "pid" in controllers_to_run:
        p_list = ["p90", "p95", "p99"] if args.percentile == "all" else [args.percentile]
        for p in p_list:
            for cond in conditions:
                tasks.append({"controller": "pid", "percentile": p, "condition": cond})

    # MPPI (p90, p95, p99 with and without perturbations -> 6 evaluations)
    if "mppi" in controllers_to_run:
        p_list = ["p90", "p95", "p99"] if args.percentile == "all" else [args.percentile]
        for p in p_list:
            for cond in conditions:
                tasks.append({"controller": "mppi", "percentile": p, "condition": cond})

    is_multi_task = (len(tasks) > 1) and not args.subtask
    if is_multi_task:
        import subprocess

        limits_file = Path(args.pressure_limits_file) if args.pressure_limits_file else (out_base / "pressure_limits.json")
        if not limits_file.exists():
            load_pressure_limits(str(limits_file) if args.pressure_limits_file else None, out_base)

        for task in tasks:
            ctrl = task["controller"]
            cond = task["condition"]
            p = task["percentile"]
            folder_name = f"{ctrl}_{cond}" if p is None else f"{ctrl}_{p}_{cond}"
            target_summary = out_base / folder_name / "summary.json"

            if target_summary.exists():
                try:
                    with open(target_summary, "r") as sf:
                        ex = json.load(sf)
                    if ex.get("num_trials", 0) >= args.num_trials:
                        print(f"\n⚡ [{folder_name.upper()}] already completed ({ex['num_trials']} trials). Skipping!")
                        continue
                except Exception:
                    pass

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--controller", ctrl,
                "--condition", cond,
                "--num_trials", str(args.num_trials),
                "--num_detail", str(args.num_detail),
                "--max_steps", str(args.max_steps),
                "--lhs_file", str(args.lhs_file),
                "--output_dir", str(args.output_dir),
                "--rl_model_path", str(args.rl_model_path),
                "--mppi_workers", str(args.mppi_workers),
                "--mppi_samples", str(args.mppi_samples),
                "--mppi_horizon", str(args.mppi_horizon),
                "--pressure_limits_file", str(limits_file),
                "--subtask",
            ]
            if p is not None:
                cmd.extend(["--percentile", p])

            limit_str = f" [{p.upper()}]" if p else ""
            print(f"\n🚀 Launching isolated evaluation process for {ctrl.upper()}{limit_str} [{cond}]...")
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"⚠️ Process for {ctrl.upper()}{limit_str} [{cond}] returned code {ret.returncode}")

            # If RL just completed, immediately extract/update pressure_limits.json and .yaml
            if ctrl == "rl":
                extract_rl_pressure_percentiles(out_base, limits_file)

            # Incrementally update comparison table and print cumulative stats
            generate_comparison_report(out_base)

        print("\n" + "="*85)
        print("🎉 ALL REQUESTED BENCHMARKS COMPLETED SUCCESSFULLY")
        print("="*85)
        generate_comparison_report(out_base)
        return

    # 4. Single-task execution (isolated child process or direct single run)
    limits_file = Path(args.pressure_limits_file) if args.pressure_limits_file else (out_base / "pressure_limits.json")
    limits_data = load_pressure_limits(str(limits_file) if args.pressure_limits_file else None, out_base)

    target_ctrl = controllers_to_run[0]
    target_cond = conditions[0]
    target_p = None if target_ctrl in ["open_loop", "rl"] else ("p99" if args.percentile == "all" else args.percentile)
    caps = get_caps_for_task(limits_data, target_p, target_cond)

    if target_ctrl == "open_loop":
        adapter = OpenLoopAdapter(n_steps=50)
    elif target_ctrl == "rl":
        if not os.path.exists(args.rl_model_path):
            print(f"Error: RL model not found at {args.rl_model_path}")
            sys.exit(1)
        adapter = RLAdapter(args.rl_model_path)
    elif target_ctrl == "pid":
        adapter = PIDAdapter(n_steps=50, pressure_caps=caps)
    elif target_ctrl == "mppi":
        adapter = MPPIAdapter(
            n_samples=args.mppi_samples,
            horizon=args.mppi_horizon,
            n_workers=args.mppi_workers,
            pressure_caps=caps,
        )

    summary = evaluate_controller(
        controller_name=target_ctrl,
        adapter=adapter,
        condition=target_cond,
        lhs_samples=lhs_samples,
        out_base_dir=out_base,
        num_trials=args.num_trials,
        num_detail=args.num_detail,
        max_steps=args.max_steps,
        percentile=target_p,
        pressure_caps=caps,
    )

    if target_ctrl == "rl":
        extract_rl_pressure_percentiles(out_base, limits_file)

    if not args.subtask:
        generate_comparison_report(out_base)


if __name__ == "__main__":
    main()
