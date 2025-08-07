# simulation/environment.py

import pybullet as p
import pybullet_data
import time
import configs.main_config as cfg
# Import the new SensorHandler class
from simulation.sensor_handler import SensorHandler 

class RoboticArmEnv:
    """
    The main environment class for the robotic arm simulation.
    Handles the connection to PyBullet, loading assets, and running the simulation.
    """
    def __init__(self, config):
        self.config = config
        self.physics_client = self._connect_to_simulation()

        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.config.SIMULATION_TIME_STEP)
        
        p.loadURDF("plane.urdf")
        self.robot_id = self._load_robot_model()
        self.glove_id = self._load_glove_model()
        
        # --- NEW: Initialize the Sensor Handler ---
        # It needs to know about the glove model to read data from it.
        self.sensor_handler = SensorHandler(glove_model_id=self.glove_id, config=self.config)
        # --- END NEW ---

        self._setup_camera()
        print("--- Simulation Environment Initialized ---")

    def _connect_to_simulation(self):
        print("Connecting to PyBullet...")
        client = p.connect(p.GUI)
        return client

    def _load_robot_model(self):
        print(f"Loading robot from: {self.config.ROBOT_URDF_PATH}")
        # Using the improved start position from your changes
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        robot_id = p.loadURDF(
            str(self.config.ROBOT_URDF_PATH),
            start_pos,
            start_orientation,
            useFixedBase=True
        )
        return robot_id

    def _load_glove_model(self):
        print(f"Loading ghost glove from: {self.config.GLOVE_URDF_PATH}")
        # Using the improved start position from your changes
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        glove_id = p.loadURDF(
            str(self.config.GLOVE_URDF_PATH),
            start_pos,
            start_orientation,
            useFixedBase=True
        )
        
        self._set_transparency(glove_id, alpha=0.4)
        print("Ghost glove model loaded and set to be semi-transparent.")
        return glove_id

    def _set_transparency(self, model_id, alpha=0.5):
        for i in range(p.getNumJoints(model_id) + 1):
            visual_shape_data = p.getVisualShapeData(model_id, i)
            if visual_shape_data:
                for shape in visual_shape_data:
                    rgba = list(shape[7])
                    rgba[3] = alpha
                    p.changeVisualShape(model_id, i, rgbaColor=rgba)

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=35,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0.7]
        )

    # --- NEW: Method to get all sensor readings at once ---
    def get_all_sensor_readings(self, visualize_flex=True):
        """
        Convenience method to get all sensor data from the handler.
        
        Args:
            visualize_flex (bool): If True, flex sensor debug lines are drawn.
            
        Returns:
            A tuple containing (flex_sensor_data, imu_data)
        """
        flex_data = self.sensor_handler.get_flex_sensor_values(visualize=visualize_flex)
        imu_data = self.sensor_handler.get_imu_values()
        return flex_data, imu_data
    # --- END NEW ---

    def run_visualization(self):
        print("\nStarting visualization. Close the window to exit.")
        try:
            while True:
                p.stepSimulation()
                time.sleep(self.config.SIMULATION_TIME_STEP)
        except p.error as e:
            print(f"PyBullet window closed.")

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            print("--- Simulation Disconnected ---")