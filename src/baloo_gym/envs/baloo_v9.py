from baloo_gym.envs.baloo_base import BalooBase
from gymnasium import spaces
import numpy as np
from baloo_gym.utils.observation_spaces import StateObservationObjectOnly
from baloo_gym.utils.helpers import get_sensor_data
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_joint_pressures,
    get_elevator_activation,
    get_elevator_cmd,
    get_contact_forces_on_body,
    get_box_position,
    get_chest_position,
    get_box_quat,
    get_box_vel,
    get_box_angvel,
)
from baloo_gym.utils.action_spaces import NormalizedDifferentialPressure
import mujoco


class BalooV9(BalooBase):

    def __init__(
        self,
        render_mode=None,
        camera_name='frontcam',
        ctrl_timestep=0.01,
        render_width=320,
        render_height=240,
        desired_box_pos=None,
        randomize_initial_height=False,
        randomize_object_size=False,
        randomize_object_mass=False,
        object_size=None,
        object_mass=None,
        randomize_object_quat=False,
        randomize_object_pos=False,
        object_xpos=None,
        object_zrotation=None,
    ):
        super().__init__(
            render_mode=render_mode,
            camera_name=camera_name,
            ctrl_timestep=ctrl_timestep,
            render_width=render_width,
            render_height=render_height,
            randomize_initial_height=randomize_initial_height,
            randomize_object_size=randomize_object_size,
            randomize_object_mass=randomize_object_mass,
            object_size=object_size,
            object_mass=object_mass,
            randomize_object_quat=randomize_object_quat,
            randomize_object_pos=randomize_object_pos,
            object_xpos=object_xpos,
            object_zrotation=object_zrotation,
        )

        self.action_space = spaces.Box(
            -1,
            1,
            shape=NormalizedDifferentialPressure.shape,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            -1, 1, shape=StateObservationObjectOnly.shape, dtype=np.float32)

        self.rise_time = 3
        self.lpf = LowPassFilter(rise_time=self.rise_time,
                                 sampling_period=ctrl_timestep)
        self.p_filt_cmd = np.zeros(self.len_command - 1)

    def get_observation_from_mujoco(self):
        #implement something that includes previous actions as well as state observations
        sensor_data = get_sensor_data(self.model, self.data)

        sensor_data["left_j0_pressures"] = get_joint_pressures(
            self.model, self.data, 'left', 0)
        sensor_data["left_j1_pressures"] = get_joint_pressures(
            self.model, self.data, 'left', 1)
        sensor_data["left_j2_pressures"] = get_joint_pressures(
            self.model, self.data, 'left', 2)

        sensor_data["right_j0_pressures"] = get_joint_pressures(
            self.model, self.data, 'right', 0)
        sensor_data["right_j1_pressures"] = get_joint_pressures(
            self.model, self.data, 'right', 1)
        sensor_data["right_j2_pressures"] = get_joint_pressures(
            self.model, self.data, 'right', 2)

        box_xpos = get_box_position(self.unwrapped.model, self.unwrapped.data)
        box_quat = get_box_quat(self.unwrapped.model, self.unwrapped.data)
        sensor_data["box_pose"] = np.hstack((box_xpos, box_quat))
        del sensor_data["object_pos"]

        box_vel = get_box_vel(self.unwrapped.model, self.unwrapped.data)
        box_angvel = get_box_angvel(self.unwrapped.model, self.unwrapped.data)
        sensor_data["box_twist"] = np.hstack((box_vel, box_angvel))
        del sensor_data["object_vel"]

        sensor_data["left_j0_p_filt_cmd"] = self.p_filt_cmd[:4]
        sensor_data["left_j1_p_filt_cmd"] = self.p_filt_cmd[4:8]
        sensor_data["left_j2_p_filt_cmd"] = self.p_filt_cmd[8:12]
        sensor_data["right_j0_p_filt_cmd"] = self.p_filt_cmd[12:16]
        sensor_data["right_j1_p_filt_cmd"] = self.p_filt_cmd[16:20]
        sensor_data["right_j2_p_filt_cmd"] = self.p_filt_cmd[20:24]

        chest_xpos = get_chest_position(self.unwrapped.model,
                                        self.unwrapped.data)

        box_xpos = get_box_position(self.unwrapped.model, self.unwrapped.data)

        sensor_data["chest2box"] = box_xpos - chest_xpos

        box_size = self.unwrapped.model.geom("box").size
        sensor_data["box_size"] = 2 * np.array(
            box_size)  #2* because mujoco reports half sizes

        rawObs = StateObservationObjectOnly(**sensor_data)

        tmp = rawObs.normalize_and_center().astype(
            self.observation_space.dtype)

        return tmp

    def map_action_to_commands(self, action):
        '''
        actions are continous and normalized between -1 and 1
        '''

        #filter ONLY pressure actions, since elevator is filtered already in sim
        pressure_actions = action[1:]
        filtered_pressure_actions = self.lpf.apply_filter(pressure_actions)

        #reconstruct filtered actions
        filtered_actions = np.zeros_like(action)
        filtered_actions[0] = action[0]
        filtered_actions[1:] = filtered_pressure_actions

        unnorm_filtered_actions = NormalizedDifferentialPressure(
            filtered_actions).unnormalize()

        #make this more concise later with a loop
        filt_commands = np.zeros(self.len_command)
        filt_commands[0] = unnorm_filtered_actions.elevator_height_cmd
        filt_commands[
            1] = 150 + unnorm_filtered_actions.left_j0_delta_pressure[0]
        filt_commands[
            2] = 150 - unnorm_filtered_actions.left_j0_delta_pressure[0]

        filt_commands[
            3] = 150 + unnorm_filtered_actions.left_j0_delta_pressure[1]
        filt_commands[
            4] = 150 - unnorm_filtered_actions.left_j0_delta_pressure[1]

        filt_commands[
            5] = 150 + unnorm_filtered_actions.left_j1_delta_pressure[0]
        filt_commands[
            6] = 150 - unnorm_filtered_actions.left_j1_delta_pressure[0]

        filt_commands[
            7] = 150 + unnorm_filtered_actions.left_j1_delta_pressure[1]
        filt_commands[
            8] = 150 - unnorm_filtered_actions.left_j1_delta_pressure[1]

        filt_commands[
            9] = 150 + unnorm_filtered_actions.left_j2_delta_pressure[0]
        filt_commands[
            10] = 150 - unnorm_filtered_actions.left_j2_delta_pressure[0]

        filt_commands[
            11] = 150 + unnorm_filtered_actions.left_j2_delta_pressure[1]
        filt_commands[
            12] = 150 - unnorm_filtered_actions.left_j2_delta_pressure[1]

        filt_commands[
            13] = 150 + unnorm_filtered_actions.right_j0_delta_pressure[0]
        filt_commands[
            14] = 150 - unnorm_filtered_actions.right_j0_delta_pressure[0]

        filt_commands[
            15] = 150 + unnorm_filtered_actions.right_j0_delta_pressure[1]
        filt_commands[
            16] = 150 - unnorm_filtered_actions.right_j0_delta_pressure[1]

        filt_commands[
            17] = 150 + unnorm_filtered_actions.right_j1_delta_pressure[0]
        filt_commands[
            18] = 150 - unnorm_filtered_actions.right_j1_delta_pressure[0]

        filt_commands[
            19] = 150 + unnorm_filtered_actions.right_j1_delta_pressure[1]
        filt_commands[
            20] = 150 - unnorm_filtered_actions.right_j1_delta_pressure[1]

        filt_commands[
            21] = 150 + unnorm_filtered_actions.right_j2_delta_pressure[0]
        filt_commands[
            22] = 150 - unnorm_filtered_actions.right_j2_delta_pressure[0]

        filt_commands[
            23] = 150 + unnorm_filtered_actions.right_j2_delta_pressure[1]
        filt_commands[
            24] = 150 - unnorm_filtered_actions.right_j2_delta_pressure[1]

        self.p_filt_cmd = filt_commands[1:]

        return filt_commands

    def calculate_reward(self) -> float:
        return 0

    def reset(self, seed=None, options=None):
        #reset stateful info in this class
        self.lpf = LowPassFilter(rise_time=self.rise_time,
                                 sampling_period=self.control_timestep)
        self.p_filt_cmd = np.zeros(self.len_command - 1)

        return super().reset(seed=seed, options=options)


class LowPassFilter:

    def __init__(self, rise_time, sampling_period):
        """
        Initialize the real-time low-pass filter based on rise time.

        Parameters:
        - rise_time (float): The rise time (seconds) of the low-pass filter.
        - sampling_period (float): The sampling period (seconds) between each update.
        """
        # Calculate cutoff frequency (f_c) from the rise time (T_r)
        tau = rise_time / 2.2  # Time constant

        # Calculate alpha based on cutoff frequency and sampling period
        self.alpha = sampling_period / (tau + sampling_period)

        # Initialize the previous output (y[n-1]) to 0
        self.previous_output = 0

    def apply_filter(self, new_input):
        """
        Apply the filter to the new input and return the filtered output.

        Parameters:
        - new_input (float): The new data (sensor input, etc.)

        Returns:
        - float: The filtered output.
        """
        # Update the filtered output using the previous output and the current input
        filtered_output = (
            1 - self.alpha) * self.previous_output + self.alpha * new_input

        # Store the current output for the next iteration
        self.previous_output = filtered_output

        return filtered_output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Define the rise time and sampling rate
    rise_time = 3  # seconds
    sampling_period = 0.05  # seconds (100 Hz sampling rate)

    # Initialize the real-time low-pass filter with rise time
    lpf = LowPassFilter(rise_time, sampling_period)

    # Create a step input signal (1 for all time steps after t=0)
    num_steps = 500
    step_input = np.ones(num_steps)

    # Initialize an array to store the filtered output
    filtered_output = np.zeros(num_steps)

    # Apply the low-pass filter to the step input
    for i in range(num_steps):
        filtered_output[i] = lpf.apply_filter(step_input[i])

    # Plot the step input and the filtered output (step response)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(num_steps) * sampling_period,
             step_input,
             label='Step Input',
             linestyle='--')
    plt.plot(np.arange(num_steps) * sampling_period,
             filtered_output,
             label='Filtered Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Step Response of Real-Time Low-Pass Filter')
    plt.legend()
    plt.grid(True)
    plt.show()
