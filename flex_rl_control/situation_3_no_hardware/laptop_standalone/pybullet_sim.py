# File: flex_rl_control/situation_3_no_hardware/laptop_standalone/pybullet_sim.py

import pybullet as p
import pybullet_data
import os

class PyBulletSim:
    """
    Manages the PyBullet Physics simulation.
    Loads the Ghost Arm (Target) and the Robotic Arm (Agent).
    """
    def __init__(self):
        # Connect to GUI
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Disable gravity so arms don't drop down while testing
        p.setGravity(0, 0, 0) 

        # Paths to URDFs (Assumes script is executed from the project root)
        glove_urdf_path = "urdf/glove.urdf"
        arm_urdf_path = "urdf/robotic_arm.urdf"

        if not os.path.exists(glove_urdf_path):
            raise FileNotFoundError(f"Missing URDF: {glove_urdf_path}")
        if not os.path.exists(arm_urdf_path):
            raise FileNotFoundError(f"Missing URDF: {arm_urdf_path}")

        # 1. Load Ghost (Glove) - Positioned slightly to the left
        self.ghost_id = p.loadURDF(glove_urdf_path, basePosition=[-0.15, 0, 0], useFixedBase=True)
        # Make the Ghost Arm transparent greenish
        self._set_transparent(self.ghost_id,[0.2, 0.8, 0.2, 0.4])

        # 2. Load Agent (Robotic Arm) - Positioned slightly to the right
        self.agent_id = p.loadURDF(arm_urdf_path, basePosition=[0.15, 0, 0], useFixedBase=True)

        # Dictionary mapping joint names to their pybullet indices
        self.ghost_joints = self._get_joint_mapping(self.ghost_id)
        self.agent_joints = self._get_joint_mapping(self.agent_id)

    def _get_joint_mapping(self, body_id):
        """ Creates a dictionary mapping joint names to their indices. """
        mapping = {}
        for i in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, i)
            name = info[1].decode('utf-8')
            mapping[name] = i
        return mapping

    def _set_transparent(self, body_id, rgba):
        """ Changes the visual shape to be semi-transparent. """
        for i in range(-1, p.getNumJoints(body_id)):
            p.changeVisualShape(body_id, i, rgbaColor=rgba)

    def _apply_finger_angles(self, body_id, joint_map, prefix, angle):
        """
        Helper function: Takes a single finger angle and applies it to the MCP, PIP, and DIP joints
        so the entire finger curls smoothly together in the 27-DOF URDF.
        """
        joints =[f"{prefix}_mcp_flex_joint", f"{prefix}_pip_joint", f"{prefix}_dip_joint"]
        if prefix == "thumb":
            joints =[f"{prefix}_mcp_flex_joint", f"{prefix}_ip_joint"]

        for j_name in joints:
            if j_name in joint_map:
                p.setJointMotorControl2(
                    bodyIndex=body_id,
                    jointIndex=joint_map[j_name],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=1.5 # Simulated motor torque
                )

    def update_ghost(self, sensor_data):
        """ Updates the transparent Ghost arm based on raw user sensor data. """
        # Convert flex 0-1023 directly to radians for the visual ghost
        thumb_ang = (sensor_data["flex_thumb"] / 1023.0) * 1.57
        index_ang = (sensor_data["flex_index"] / 1023.0) * 1.57
        middle_ang = (sensor_data["flex_middle"] / 1023.0) * 1.57
        ring_ang = (sensor_data["flex_ring"] / 1023.0) * 1.57
        pinky_ang = (sensor_data["flex_pinky"] / 1023.0) * 1.57
        wrist_ang = sensor_data["imu_euler"][1]

        self._apply_finger_angles(self.ghost_id, self.ghost_joints, "thumb", thumb_ang)
        self._apply_finger_angles(self.ghost_id, self.ghost_joints, "index", index_ang)
        self._apply_finger_angles(self.ghost_id, self.ghost_joints, "middle", middle_ang)
        self._apply_finger_angles(self.ghost_id, self.ghost_joints, "ring", ring_ang)
        self._apply_finger_angles(self.ghost_id, self.ghost_joints, "pinky", pinky_ang)

        if "wrist_flex_joint" in self.ghost_joints:
            p.setJointMotorControl2(self.ghost_id, self.ghost_joints["wrist_flex_joint"], 
                                    p.POSITION_CONTROL, targetPosition=wrist_ang, force=2.0)

    def update_agent(self, motor_commands):
        """ Updates the solid Robotic Arm based on RL Agent commands. """
        self._apply_finger_angles(self.agent_id, self.agent_joints, "thumb", motor_commands["thumb_angle"])
        self._apply_finger_angles(self.agent_id, self.agent_joints, "index", motor_commands["index_angle"])
        self._apply_finger_angles(self.agent_id, self.agent_joints, "middle", motor_commands["middle_angle"])
        self._apply_finger_angles(self.agent_id, self.agent_joints, "ring", motor_commands["ring_angle"])
        self._apply_finger_angles(self.agent_id, self.agent_joints, "pinky", motor_commands["pinky_angle"])

        if "wrist_flex_joint" in self.agent_joints:
            p.setJointMotorControl2(self.agent_id, self.agent_joints["wrist_flex_joint"], 
                                    p.POSITION_CONTROL, targetPosition=motor_commands["wrist_angle"], force=2.0)

    def step(self):
        """ Advances the physics engine by one step. """
        p.stepSimulation()