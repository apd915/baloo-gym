import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import detect_box_on_ground, apply_wrench_to_body, clear_wrenches
import numpy as np
import mujoco

import numpy as np
from scipy.spatial.transform import Rotation as R
import threading


def rotation_matrix_from_vector_to_vector(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0.999999:
            return np.eye(3)  # Same direction
        else:
            # 180 deg rotation: pick any orthogonal axis
            axis = np.array([1, 0, 0]) if abs(v1[0]) < 0.99 else np.array(
                [0, 1, 0])
            orthogonal = np.cross(v1, axis)
            orthogonal /= np.linalg.norm(orthogonal)
            return R.from_rotvec(np.pi * orthogonal).as_matrix()
    else:
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        return R.from_rotvec(angle * axis).as_matrix()


class ObjectPerturbationWrapper(gym.Wrapper):
    """
    A wrapper for the environment that applies perturbations to objects.
    """

    def __init__(self, env):
        super().__init__(env)
        self.perturbation_duration: int = int(1 /
                                              self.unwrapped.control_timestep)
        self.perturbation_magnitude: float = 100.0
        self.step_counter: int = 0
        self.arrow_initialized: bool = False
        self.force_direction = np.array([0, 0, -1])
        self.perturbation_enabled = False

    def step(self, action):
        """
        Take a step in the environment and apply perturbations.
        """
        force = np.zeros(3)
        cycle_length = 2 * self.perturbation_duration  #1s ON, 1s OFF

        # Apply perturbations on the object once it is off the ground
        if detect_box_on_ground(self.unwrapped.model, self.unwrapped.data):
            #reset everything
            self.step_counter = 0
            self.force_direction = np.array([0, 0, -1])
            self.perturbation_enabled = False

        else:
            #on or off every self.perturbation_duration steps
            self.perturbation_enabled = (
                self.step_counter % cycle_length) >= self.perturbation_duration

            if self.perturbation_enabled:
                # #check if we just enable the perturbation to update direction
                # if self.step_counter % cycle_length == self.perturbation_duration:
                #     self.force_direction = np.random.uniform(-1, 1, 3)
                #     self.force_direction /= np.linalg.norm(
                #         self.force_direction)
                force = self.force_direction * self.perturbation_magnitude

            self.step_counter += 1

        clear_wrenches(self.unwrapped.model, self.unwrapped.data)
        apply_wrench_to_body(self.unwrapped.model,
                             self.unwrapped.data,
                             "box",
                             force=force,
                             torque=np.zeros(3))

        # self._update_force_arrow(self.perturbation_enabled)

        # print(f"perturbation enabled: {self.perturbation_enabled}")
        # print(f"applied force: {force}")
        # print(self.unwrapped.data.qfrc_applied)

        obs, reward, terminated, truncated, info = self.env.step(action)
        # clear_wrenches(self.unwrapped.model, self.unwrapped.data)

        info['perturbation_enabled'] = self.perturbation_enabled
        return obs, reward, terminated, truncated, info

    def _update_force_arrow(self, force_active):
        box_pos = self.unwrapped.data.body('box').xipos
        arrow_pos = box_pos + np.array([0, 0, 1], dtype=np.float64)

        v1 = np.array([0, 0, 1], dtype=np.float64)
        R = rotation_matrix_from_vector_to_vector(v1, self.force_direction)
        R = R.astype(np.float64).flatten()

        rgba = np.array([1.0, 0.0, 0.0, float(force_active)], dtype=np.float64)

        viewer = self.unwrapped.mujoco_renderer.viewer
        if viewer is None:
            return  # Or handle error

        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[0.05, 0.05, 0.5],
            pos=arrow_pos,
            mat=R,
            rgba=rgba,
        )

    def render(self):
        return super().render()

    # def render(self):
    #     ret = super().render()

    #     if not self.arrow_initialized:
    #         #this will add the marker the next time the scene is rendered.
    #         # everything in the marker list will be re-rendered each time.
    #         self.unwrapped.mujoco_renderer.viewer.add_marker(
    #             type=mujoco.mjtGeom.mjGEOM_ARROW,
    #             size=[0.05, 0.05, 0.5],
    #             pos=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    #             mat=np.eye(3, dtype=np.float64),
    #             rgba=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    #         )

    #         self.arrow_initialized = True

    #     return ret

    def close(self):
        try:
            if hasattr(self.unwrapped, "mujoco_renderer"):
                viewer = getattr(self.unwrapped.mujoco_renderer, "viewer",
                                 None)
                if viewer is not None:
                    viewer.close()
                    self.unwrapped.mujoco_renderer.viewer = None
        except Exception as e:
            print(f"Error closing viewer: {e}")

        super().close()
