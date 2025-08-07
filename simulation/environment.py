# simulation/environment.py

import pybullet as p
import pybullet_data
import time
import configs.main_config as cfg

class RoboticArmEnv:
    """
    The main environment class for the robotic arm simulation.
    Handles the connection to PyBullet, loading assets, and running the simulation.
    """
    def __init__(self, config):
        """
        Initializes the simulation environment.

        Args:
            config: A configuration module or object with necessary parameters.
        """
        self.config = config
        self.physics_client = self._connect_to_simulation()

        # Set up the simulation environment
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.config.SIMULATION_TIME_STEP)
        
        # Load the ground plane and models
        p.loadURDF("plane.urdf")
        self.robot_id = self._load_robot_model()
        self.glove_id = self._load_glove_model()
        
        # Set a good camera angle for viewing
        self._setup_camera()
        print("--- Simulation Environment Initialized ---")

    def _connect_to_simulation(self):
        """Establishes a connection to the PyBullet physics server."""
        print("Connecting to PyBullet...")
        # p.GUI for graphical visualization, p.DIRECT for running without a window
        client = p.connect(p.GUI)
        return client

    def _load_robot_model(self):
        """Loads the primary robotic arm model into the simulation."""
        print(f"Loading robot from: {self.config.ROBOT_URDF_PATH}")
        # Starting position and orientation for the main robot
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        robot_id = p.loadURDF(
            str(self.config.ROBOT_URDF_PATH),
            start_pos,
            start_orientation,
            useFixedBase=True # The arm is fixed to its base
        )
        return robot_id

    def _load_glove_model(self):
        """
        Loads the 'ghost' glove model and makes it semi-transparent for visualization.
        This model will serve as the target pose for the robot to imitate.
        """
        print(f"Loading ghost glove from: {self.config.GLOVE_URDF_PATH}")
        # We place it to the side for easy comparison in this initial step
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        glove_id = p.loadURDF(
            str(self.config.GLOVE_URDF_PATH),
            start_pos,
            start_orientation,
            useFixedBase=True
        )
        
        # Make the glove model semi-transparent (ghost-like)
        self._set_transparency(glove_id, alpha=0.4)
        print("Ghost glove model loaded and set to be semi-transparent.")
        return glove_id

    def _set_transparency(self, model_id, alpha=0.5):
        """
        Sets the transparency for all links of a given model.
        
        Args:
            model_id (int): The unique ID of the model in PyBullet.
            alpha (float): The transparency level (0.0=fully transparent, 1.0=fully opaque).
        """
        # Iterate through all links of the model, including the base link (-1)
        for i in range(p.getNumJoints(model_id) + 1):
            visual_shape_data = p.getVisualShapeData(model_id, i)
            if visual_shape_data:
                # The first element in visual_shape_data is a list of shapes
                for shape in visual_shape_data:
                    # The RGBA color is the 7th element in the shape data tuple
                    rgba = list(shape[7])
                    rgba[3] = alpha # Set the alpha channel
                    p.changeVisualShape(model_id, i, rgbaColor=rgba)

    def _setup_camera(self):
        """Sets a default camera view for the simulation."""
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=35,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0.7]
        )

    def run_visualization(self):
        """
        Runs a simple loop to keep the simulation window open for visualization.
        In the future, this will be replaced by the main training loop.
        """
        print("\nStarting visualization. Close the window to exit.")
        try:
            while True:
                p.stepSimulation()
                time.sleep(self.config.SIMULATION_TIME_STEP)
        except p.error as e:
            # This handles the case where the user closes the window
            print(f"PyBullet window closed.")

    def close(self):
        """Disconnects from the PyBullet simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            print("--- Simulation Disconnected ---")