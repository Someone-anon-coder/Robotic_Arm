# Will contain the main Gymnasium environment class (RoboticArmEnv) and visualizer.
import pybullet as p
import pybullet_data
import time
from types import ModuleType

class ArmVisualizer:
    """A class to visualize the robotic arm and glove in PyBullet."""

    def __init__(self, config: ModuleType):
        """
        Initializes the PyBullet simulation environment.

        Args:
            config: A configuration module with simulation parameters.
        """
        self.client_id = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath('eeg_robotic_arm_project/urdf')
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set a nice camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )

        # Load ground plane
        p.loadURDF("plane.urdf")

        # Load the Robotic Arm
        self.robot_id = p.loadURDF(
            fileName=config.ROBOT_URDF_PATH,
            basePosition=config.ROBOT_START_POS,
            baseOrientation=config.START_ORIENTATION,
            useFixedBase=True
        )

        # Load the Ghost Glove
        self.glove_id = p.loadURDF(
            fileName=config.GLOVE_URDF_PATH,
            basePosition=config.GLOVE_START_POS,
            baseOrientation=config.START_ORIENTATION,
            useFixedBase=True
        )

        # Make the Glove Transparent
        num_joints = p.getNumJoints(self.glove_id)
        for i in range(-1, num_joints):  # -1 for the base
            visual_shape_data = p.getVisualShapeData(self.glove_id, i)
            if visual_shape_data:
                # The visual_shape_data is a list of tuples, one for each visual shape
                for shape in visual_shape_data:
                    # shape[7] is the rgbaColor
                    rgba_color = list(shape[7])
                    rgba_color[3] = 0.4  # Set alpha to 0.4 for transparency
                    p.changeVisualShape(
                        self.glove_id,
                        i,
                        rgbaColor=rgba_color,
                        shapeIndex=shape[0] # specify which shape to change
                    )

    def run(self) -> None:
        """
        Runs the simulation loop.
        """
        try:
            while True:
                p.stepSimulation()
                time.sleep(1./240.)
        except p.error as e:
            # This exception is thrown when the user closes the GUI window.
            print(f"PyBullet GUI closed. {e}")


    def close(self) -> None:
        """
        Closes the PyBullet connection.
        """
        p.disconnect(self.client_id)
