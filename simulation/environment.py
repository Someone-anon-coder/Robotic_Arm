# simulation/environment.py

import pybullet as p
import pybullet_data
import time
import configs.main_config as cfg
from simulation.sensor_handler import SensorHandler
# NEW: Import gymnasium and numpy
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Make the class inherit from gym.Env
class RoboticArmEnv(gym.Env):
    """
    The main environment class for the robotic arm simulation.
    This class now conforms to the Gymnasium API.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.physics_client = self._connect_to_simulation()

        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.config.SIMULATION_TIME_STEP)
        
        # --- Load Models and Get Joint Info ---
        p.loadURDF("plane.urdf")
        self.robot_id = self._load_model("robot", self.config.ROBOT_URDF_PATH)
        self.glove_id = self._load_model("glove", self.config.GLOVE_URDF_PATH, is_ghost=True)

        # Create mappings from joint names to their pybullet indices
        self.robot_joint_map = self._get_joint_map(self.robot_id)
        self.glove_joint_map = self._get_joint_map(self.glove_id)

        self.sensor_handler = SensorHandler(glove_model_id=self.glove_id, config=self.config)
        
        # --- Define Action and Observation Spaces ---
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

        self._setup_camera()
        print("--- Gymnasium Environment Initialized ---")

    # --- Core Gym Methods: step, reset, close ---
    def step(self, action):
        """Applies an action, steps the simulation, and returns the results."""
        self._apply_action(action)
        p.stepSimulation()
        
        observation = self._get_observation()
        reward = self._compute_reward(observation)
        terminated = False  # In this phase, the simulation doesn't end on its own
        truncated = False
        info = {} # Placeholder for auxiliary diagnostic info
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        # For now, we just reset the robot to a zero-pose.
        # Later, we can randomize the start state.
        for joint_name in self.config.CONTROLLED_JOINTS:
            robot_joint_index = self.robot_joint_map[joint_name]
            glove_joint_index = self.glove_joint_map[joint_name]
            p.resetJointState(self.robot_id, robot_joint_index, 0)
            p.resetJointState(self.glove_id, glove_joint_index, 0)

        observation = self._get_observation()
        info = {}
        return observation, info

    def close(self):
        """Disconnects from the simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            print("--- Simulation Disconnected ---")

    # --- Helper Methods ---

    def _apply_action(self, action):
        """Applies the given action to the robotic arm motors."""
        for i, joint_name in enumerate(self.config.CONTROLLED_JOINTS):
            joint_index = self.robot_joint_map[joint_name]
            target_position = action[i]
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                # Forces can be tuned for better performance
                force=500,
                maxVelocity=10.0
            )

    def _get_observation(self):
        """Gathers all sensor data and robot states into a single observation dict."""
        flex_data, imu_data = self.sensor_handler.get_all_sensor_readings(visualize_flex=False)
        
        # For now, we just use the sensor data. Later we will add robot joint states.
        observation = {
            "flex_sensors": np.array(list(flex_data.values()), dtype=np.float32),
            "imu_actual": np.concatenate([
                imu_data["IMU_a"]["linear_velocity"],
                imu_data["IMU_a"]["angular_velocity"],
                imu_data["IMU_a"]["orientation_quaternion"]
            ]).astype(np.float32)
        }
        return observation
        
    def _compute_reward(self, observation):
        """Calculates the reward based on how well the robot mimics the glove."""
        robot_joint_states = self.get_model_joint_states(self.robot_id)
        glove_joint_states = self.get_model_joint_states(self.glove_id)
        
        # Calculate the difference (error) between the joint positions
        error = np.array(robot_joint_states) - np.array(glove_joint_states)
        # Use Mean Squared Error, a common reward metric for imitation
        mse = np.mean(np.square(error))
        
        # Reward is the negative of the error (we want to minimize error)
        reward = -mse
        return reward

    def _define_action_space(self):
        """Defines the action space based on the robot's joint limits."""
        num_joints = len(self.config.CONTROLLED_JOINTS)
        low = []
        high = []
        for joint_name in self.config.CONTROLLED_JOINTS:
            joint_index = self.robot_joint_map[joint_name]
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            low.append(joint_info[8])  # Lower limit
            high.append(joint_info[9]) # Upper limit
        
        return spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def _define_observation_space(self):
        """Defines the observation space for the RL agent."""
        # This will evolve, but for now it's the sensor data
        return spaces.Dict({
            "flex_sensors": spaces.Box(low=0, high=1, shape=(len(self.config.FLEX_SENSOR_MAPPING),), dtype=np.float32),
            "imu_actual": spaces.Box(low=-np.inf, high=np.inf, shape=(3+3+4,), dtype=np.float32) # lin_vel, ang_vel, quat
        })

    def get_model_joint_states(self, model_id):
        """Returns the current position of all controlled joints for a given model."""
        joint_states = []
        for joint_name in self.config.CONTROLLED_JOINTS:
            joint_index = self.robot_joint_map[joint_name] if model_id == self.robot_id else self.glove_joint_map[joint_name]
            state = p.getJointState(model_id, joint_index)
            joint_states.append(state[0]) # state[0] is the position
        return joint_states

    def _connect_to_simulation(self):
        return p.connect(p.GUI)

    def _load_model(self, name, urdf_path, is_ghost=False):
        """Generic model loader."""
        print(f"Loading {name} model from: {urdf_path}")
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        model_id = p.loadURDF(str(urdf_path), start_pos, start_orientation, useFixedBase=True)
        if is_ghost:
            self._set_transparency(model_id, alpha=0.4)
        return model_id

    def _get_joint_map(self, model_id):
        """Creates a mapping from joint names to their PyBullet index for a model."""
        joint_map = {}
        for i in range(p.getNumJoints(model_id)):
            info = p.getJointInfo(model_id, i)
            name = info[1].decode('UTF-8')
            joint_map[name] = i
        return joint_map

    def _set_transparency(self, model_id, alpha=0.5):
        for i in range(-1, p.getNumJoints(model_id)):
            p.changeVisualShape(model_id, i, rgbaColor=list(p.getVisualShapeData(model_id, i)[0][7])[:3] + [alpha])

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(1.2, 50, -30, [0, 0, 0.5])