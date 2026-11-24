import numpy as np
import csv
import os
from baloo_gym.utils.observation_spaces import StateObservationObjectOnly
from stable_baselines3.common.policies import BasePolicy


# =============================================================================
# Slip Correction PID
# =============================================================================
# Monitors relative velocity between elevator and box during LIFT. If the
# box drops (v_box < v_elevator), it outputs a positive squeeze_correction
# value which is added to the joint antagonist pressures to tighten the grip.
#
# Tuning guide:
#   kp  — how aggressively to respond to slip velocity (damping).
#   ki  — responds to accumulated slip distance.
#   kd  — responds to relative acceleration.
#   threshold — deadband in m/s. Corrections only fire outside this range.
#   correction_max — hard cap on extra delta-pressure (kPa).
# =============================================================================

class SlipCorrectionPID:

    def __init__(
        self,
        kp=80.0,
        ki=2.0,
        kd=1.5,
        dt=0.05,
        threshold=0.01,
        correction_min=0.0,
        correction_max=50.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.threshold = threshold
        self.correction_min = correction_min
        self.correction_max = correction_max

        self._integral = 0.0
        self._prev_error = 0.0
        self.active = False

    def arm(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self.active = True

    def update(self, v_slip):
        if not self.active:
            return 0.0

        error = v_slip

        if abs(error) < self.threshold:
            return 0.0
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        u = (self.kp * error
             + self.ki * self._integral
             + self.kd * derivative)

        return float(np.clip(u, self.correction_min, self.correction_max))

    def disarm(self):
        self.active = False
        self._integral = 0.0
        self._prev_error = 0.0


# =============================================================================
# Squeeze direction reference
# =============================================================================
# All joints follow the same [-1, +1] squeeze pattern:
#   agonist   channel: multiply correction by -1 (decrease pressure)
#   antagonist channel: multiply correction by +1 (increase pressure)
#
# J0: actions[1,2]   left  |  actions[7,8]   right
# J1: actions[3,4]   left  |  actions[9,10]  right
# J2: actions[5,6]   left  |  actions[11,12] right
# =============================================================================

SQUEEZE_DIR = np.array([-1.0, +1.0])   # same pattern for all joints, both arms

CHAMBER_MAP = {
    "left_j0_agonist":    1,
    "left_j0_antagonist": 2,
    "left_j1_agonist":    3,
    "left_j1_antagonist": 4,
    "left_j2_agonist":    5,
    "left_j2_antagonist": 6,
    "right_j0_agonist":   7,
    "right_j0_antagonist":8,
    "right_j1_agonist":   9,
    "right_j1_antagonist":10,
    "right_j2_agonist":   11,
    "right_j2_antagonist":12,
}


# =============================================================================
# Per-joint PID config helper
# =============================================================================
# per_joint_pid_params is a dict keyed by joint name, each value is a dict
# of SlipCorrectionPID kwargs (kp, ki, kd, threshold, correction_max).
#
# Example:
#   per_joint_pid_params = {
#       'left_j0':  {'kp': 105.1, 'ki': 12.8, 'kd': 4.9, 'threshold': 0.09,
#                    'correction_max': 299.3},
#       'left_j1':  {'kp': 90.0,  'ki': 5.0,  'kd': 2.0, 'threshold': 0.05,
#                    'correction_max': 299.9},
#       ...
#   }
#
# Any joint not present in the dict falls back to SlipCorrectionPID defaults.
# =============================================================================

JOINT_NAMES = [
    'left_j0', 'left_j1', 'left_j2',
    'right_j0', 'right_j1', 'right_j2',
]

# Fixed 99th-percentile correction_max per joint (kPa).
# Optuna never tunes these — they are safety ceilings derived from data.
DEFAULT_CORRECTION_MAX = {
    'left_j0':  149.3,
    'left_j1':  149.9,
    'left_j2':   13.0,
    'right_j0': 149.4,
    'right_j1': 126.2,
    'right_j2':  37.1,
}


def _make_pid(joint_name, per_joint_params):
    """
    Build a SlipCorrectionPID for the given joint.
    Uses per_joint_params[joint_name] if present, otherwise defaults.
    correction_max is always taken from DEFAULT_CORRECTION_MAX (never
    overridden by Optuna).
    """
    params = dict(per_joint_params.get(joint_name, {}))
    # Enforce default correction_max only if not explicitly provided
    params.setdefault('correction_max', DEFAULT_CORRECTION_MAX[joint_name])
    params.setdefault('dt', 0.05)
    return SlipCorrectionPID(**params)


class OpenLoopHuggerPolicy(BasePolicy):

    def __init__(self, N, slip_pid_params=None, per_joint_pid_params=None):
        """
        Parameters
        ----------
        N : int
            Number of trajectory timesteps.
        slip_pid_params : dict, optional
            DEPRECATED — kept for backward compatibility. If provided and
            per_joint_pid_params is None, the same params are applied to
            all six joints (old behaviour).
        per_joint_pid_params : dict of dicts, optional
            Keys are joint names ('left_j0', 'left_j1', 'left_j2',
            'right_j0', 'right_j1', 'right_j2').
            Values are dicts of SlipCorrectionPID kwargs.
            correction_max is always overridden by DEFAULT_CORRECTION_MAX.
        """
        super().__init__(observation_space=None, action_space=None)

        # ── Time-Delay Gate for Slip Monitoring ─────────────────────────────
        self.lift_time = 0.0
        self.min_height = -.9
        self.step_along_trajectory = 0
        self.state = "APPROACH"
        self.prev_actions = np.zeros(13)

        self.N = N
        self.actions_lb = np.asarray([-900] + [-150] * 12)
        self.actions_ub = np.asarray([0]    + [150]  * 12)
        self.avg_pressure = 150

        # ── Open-loop trajectories (unchanged) ──────────────────────────────
        self.left_lift_j0_delta_traj = np.linspace(
            np.array([0, 0]), np.array([85, 0]), N)
        self.left_grab_j0_delta_traj = np.linspace(
            np.array([85, 0]), np.array([0, 150]), N)

        self.left_lift_j1_delta_traj = np.linspace(
            np.array([0, 0]), np.array([45, 45]), N)
        self.left_grab_j1_delta_traj = np.linspace(
            np.array([45, 45]), np.array([-100, 150]), N)

        self.right_lift_j0_delta_traj = np.linspace(
            np.array([0, 0]), np.array([-20, 80]), N)
        self.right_grab_j0_delta_traj = np.linspace(
            np.array([-20, 80]), np.array([150, -50]), N)

        self.right_lift_j1_delta_traj = np.linspace(
            np.array([0, 0]), np.array([50, 50]), N)
        self.right_grab_j1_delta_traj = np.linspace(
            np.array([50, 50]), np.array([45, -105]), N)

        # ── Build 6 independent PID controllers ─────────────────────────────
        # Resolve which param source to use
        if per_joint_pid_params is not None:
            joint_params = per_joint_pid_params
        elif slip_pid_params is not None:
            # Backward-compat: broadcast the single dict to all 6 joints
            joint_params = {name: dict(slip_pid_params) for name in JOINT_NAMES}
        else:
            joint_params = {}

        self.pid_left_j0  = _make_pid('left_j0',  joint_params)
        self.pid_left_j1  = _make_pid('left_j1',  joint_params)
        self.pid_left_j2  = _make_pid('left_j2',  joint_params)
        self.pid_right_j0 = _make_pid('right_j0', joint_params)
        self.pid_right_j1 = _make_pid('right_j1', joint_params)
        self.pid_right_j2 = _make_pid('right_j2', joint_params)

        # Convenience list for arm/disarm/restart loops
        self._all_pids = [
            self.pid_left_j0,  self.pid_left_j1,  self.pid_left_j2,
            self.pid_right_j0, self.pid_right_j1, self.pid_right_j2,
        ]

        self.top_time = 0.0
        self.logs = []
        self.current_time = 0.0

        # Track whether the PID controllers were actually armed,
        # separately from whether they produced a nonzero correction.
        self.pid_was_armed = False
        self.pid_was_active = False

    # -------------------------------------------------------------------------

    def print_pressures(self, current_actions, base_pressure=150.0):
        print("\n--- Current Arm Pressures (kPa) ---")
        for chamber_name, index in CHAMBER_MAP.items():
            delta_pressure = current_actions[index]
            total_pressure = base_pressure + delta_pressure
            print(f"{chamber_name:22}: {total_pressure:6.2f} kPa  (Delta: {delta_pressure:+.2f})")
        print("-----------------------------------\n")

    def _predict(self, observation, deterministic=False):
        return None

    def normalize_actions(self, commands):
        return 2 * (commands - self.actions_lb) / (self.actions_ub - self.actions_lb) - 1

    # -------------------------------------------------------------------------

    def predict(self, obs, deterministic=True, exact_box_z=None):
        actions = self.prev_actions.copy()
        self.pid_was_active = False

        if len(obs) > 1:
            mujoco_observation = StateObservationObjectOnly.from_standardized_array(obs)
            elevator_height = mujoco_observation.elevator_pos
            v_slip = mujoco_observation.elevator_vel - mujoco_observation.box_twist[2]
        else:
            elevator_height = obs
            v_slip = 0.0

        if exact_box_z is not None:
            box_z = exact_box_z
        elif len(obs) > 1:
            box_z = mujoco_observation.chest2box[2]
        else:
            box_z = None

        # ── APPROACH ─────────────────────────────────────────────────────────
        if self.state == "APPROACH":
            for pid in self._all_pids:
                pid.disarm()

            if self.step_along_trajectory < self.N:
                actions[0] = -900
                actions[1] = self.left_lift_j0_delta_traj[self.step_along_trajectory][0]
                actions[2] = self.left_lift_j0_delta_traj[self.step_along_trajectory][1]
                actions[3] = self.left_lift_j1_delta_traj[self.step_along_trajectory][0]
                actions[4] = self.left_lift_j1_delta_traj[self.step_along_trajectory][1]
                actions[5] = 0
                actions[6] = 0
                actions[7]  = self.right_lift_j0_delta_traj[self.step_along_trajectory][0]
                actions[8]  = self.right_lift_j0_delta_traj[self.step_along_trajectory][1]
                actions[9]  = self.right_lift_j1_delta_traj[self.step_along_trajectory][0]
                actions[10] = self.right_lift_j1_delta_traj[self.step_along_trajectory][1]
                actions[11] = 0
                actions[12] = 0

                self.prev_actions = actions.copy()
                self.step_along_trajectory += 1

            if (np.isclose(elevator_height, -.900, atol=.1)
                    and self.step_along_trajectory == self.N):
                self.state = "GRASP"
                self.step_along_trajectory = 0

        # ── GRASP ─────────────────────────────────────────────────────────────
        elif self.state == "GRASP":
            print(
                f"[POLICY] GRASP | step={self.step_along_trajectory}/{self.N} "
                f"| elevator_height={elevator_height:.4f} "
                f"| box_z={box_z}"
            )

            actions[0] = -900
            actions[1] = self.left_grab_j0_delta_traj[self.step_along_trajectory][0]
            actions[2] = self.left_grab_j0_delta_traj[self.step_along_trajectory][1]
            actions[3] = self.left_grab_j1_delta_traj[self.step_along_trajectory][0]
            actions[4] = self.left_grab_j1_delta_traj[self.step_along_trajectory][1]
            actions[5] = 0
            actions[6] = 0
            actions[7]  = self.right_grab_j0_delta_traj[self.step_along_trajectory][0]
            actions[8]  = self.right_grab_j0_delta_traj[self.step_along_trajectory][1]
            actions[9]  = self.right_grab_j1_delta_traj[self.step_along_trajectory][0]
            actions[10] = self.right_grab_j1_delta_traj[self.step_along_trajectory][1]
            actions[11] = 0
            actions[12] = 0

            self.prev_actions = actions.copy()
            self.step_along_trajectory += 1

            if self.step_along_trajectory == self.N:
                print(
                    f"[POLICY] >>> ARMING PID <<< "
                    f"| box_z={box_z} "
                    f"| step={self.step_along_trajectory}"
                )

                self.state = "LIFT"
                self.step_along_trajectory = 0

                if box_z is not None:
                    for pid in self._all_pids:
                        pid.arm()

                    self.pid_was_armed = True

                    print("[POLICY] >>> PID ARMED <<<")
                else:
                    print("[POLICY] !!! PID NOT ARMED: box_z is None !!!")

        # ── LIFT ──────────────────────────────────────────────────────────────
        elif self.state == "LIFT":
            print(
                f"[POLICY] >>> ENTERED LIFT <<< "
                f"| box_z={box_z} "
                f"| lift_time={self.lift_time:.3f}"
            )

            actions[0] = 0
            self.lift_time += self.pid_left_j0.dt
            acceleration_window = 0.35

            if box_z is not None and self.lift_time >= acceleration_window:
                # Each joint gets its own independent squeeze correction
                sq_left_j0  = self.pid_left_j0.update(v_slip)
                sq_left_j1  = self.pid_left_j1.update(v_slip)
                sq_left_j2  = self.pid_left_j2.update(v_slip)
                sq_right_j0 = self.pid_right_j0.update(v_slip)
                sq_right_j1 = self.pid_right_j1.update(v_slip)
                sq_right_j2 = self.pid_right_j2.update(v_slip)

                print(
                    f"[PID] v_slip={v_slip:+.4f} m/s | "
                    f"corrections="
                    f"LJ0={sq_left_j0:.2f}, "
                    f"LJ1={sq_left_j1:.2f}, "
                    f"LJ2={sq_left_j2:.2f}, "
                    f"RJ0={sq_right_j0:.2f}, "
                    f"RJ1={sq_right_j1:.2f}, "
                    f"RJ2={sq_right_j2:.2f}"
                )
            else:
                sq_left_j0 = sq_left_j1 = sq_left_j2 = 0.0
                sq_right_j0 = sq_right_j1 = sq_right_j2 = 0.0

            # ── J0 squeeze (NEW — was open-loop only before) ──────────────
            if sq_left_j0 > 0:
                actions[1] = np.clip(actions[1] + sq_left_j0 * SQUEEZE_DIR[0], -150, 150)
                actions[2] = np.clip(actions[2] + sq_left_j0 * SQUEEZE_DIR[1], -150, 150)
            if sq_right_j0 > 0:
                actions[7] = np.clip(actions[7] + sq_right_j0 * SQUEEZE_DIR[0], -150, 150)
                actions[8] = np.clip(actions[8] + sq_right_j0 * SQUEEZE_DIR[1], -150, 150)

            # ── J1 squeeze ────────────────────────────────────────────────
            if sq_left_j1 > 0:
                actions[3] = np.clip(actions[3] + sq_left_j1 * SQUEEZE_DIR[0], -150, 150)
                actions[4] = np.clip(actions[4] + sq_left_j1 * SQUEEZE_DIR[1], -150, 150)
            if sq_right_j1 > 0:
                actions[9]  = np.clip(actions[9]  + sq_right_j1 * SQUEEZE_DIR[0], -150, 150)
                actions[10] = np.clip(actions[10] + sq_right_j1 * SQUEEZE_DIR[1], -150, 150)

            # ── J2 squeeze ────────────────────────────────────────────────
            if sq_left_j2 > 0:
                actions[5] = np.clip(actions[5] + sq_left_j2 * SQUEEZE_DIR[0], -150, 150)
                actions[6] = np.clip(actions[6] + sq_left_j2 * SQUEEZE_DIR[1], -150, 150)
            if sq_right_j2 > 0:
                actions[11] = np.clip(actions[11] + sq_right_j2 * SQUEEZE_DIR[0], -150, 150)
                actions[12] = np.clip(actions[12] + sq_right_j2 * SQUEEZE_DIR[1], -150, 150)

            self.prev_actions = actions.copy()

            any_squeeze = any(s > 0 for s in [
                sq_left_j0, sq_left_j1, sq_left_j2,
                sq_right_j0, sq_right_j1, sq_right_j2,
            ])
            if any_squeeze:
                self.pid_was_active = True

        self.print_pressures(self.prev_actions)

        # ── Log ───────────────────────────────────────────────────────────────
        self.current_time += self.pid_left_j0.dt
        log_entry = {'time': self.current_time, 'v_slip': v_slip}
        for chamber_name, index in CHAMBER_MAP.items():
            log_entry[chamber_name] = actions[index]
        self.logs.append(log_entry)

        norm_actions = self.normalize_actions(actions)
        return norm_actions, None

    # -------------------------------------------------------------------------

    def save_logs(self, filename="baloo_log.csv"):
        if not self.logs:
            return
        keys = self.logs[0].keys()
        with open(filename, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.logs)
        print(f"Log saved to {filename}")

    def restart(self):
        self.save_logs()
        self.step_along_trajectory = 0
        self.state = "APPROACH"
        self.prev_actions = np.zeros(13)

        self.pid_was_armed = False
        self.pid_was_active = False


        for pid in self._all_pids:
            pid.disarm()
        self.lift_time = 0.0
        self.top_time = 0.0
        self.current_time = 0.0
        self.logs = []