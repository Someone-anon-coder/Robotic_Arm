# File: flex_rl_control/situation_1_full_hardware/laptop_server/pybullet_sim.py

import pybullet as p
import pybullet_data
import os

class PyBulletSim:
    """
    Physics Simulation Visualizer for Situation 1.
    Renders both the target pose (Ghost) and RL pose (Agent).
    """
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0) 

        # URDF paths relative to project root
        g_path, a_path = "urdf/glove.urdf", "urdf/robotic_arm.urdf"
        
        self.ghost_id = p.loadURDF(g_path, basePosition=[-0.15, 0, 0], useFixedBase=True)
        self.agent_id = p.loadURDF(a_path, basePosition=[0.15, 0, 0], useFixedBase=True)

        for i in range(-1, p.getNumJoints(self.ghost_id)):
            p.changeVisualShape(self.ghost_id, i, rgbaColor=[0.2, 0.8, 0.2, 0.4])

        self.ghost_joints = self._get_joint_mapping(self.ghost_id)
        self.agent_joints = self._get_joint_mapping(self.agent_id)

    def _get_joint_mapping(self, body_id):
        return {p.getJointInfo(body_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(body_id))}

    def _apply_curl(self, body_id, joint_map, prefix, angle):
        joints = [f"{prefix}_mcp_flex_joint", f"{prefix}_pip_joint", f"{prefix}_dip_joint"]
        if prefix == "thumb": joints = [f"{prefix}_mcp_flex_joint", f"{prefix}_ip_joint"]
        for j in joints:
            if j in joint_map:
                p.setJointMotorControl2(body_id, joint_map[j], p.POSITION_CONTROL, targetPosition=angle)

    def update_ghost(self, sensor_data):
        for f in ["thumb", "index", "middle", "ring", "pinky"]:
            ang = (sensor_data.get(f"flex_{f}", 0) / 1023.0) * 1.57
            self._apply_curl(self.ghost_id, self.ghost_joints, f, ang)
        
        wrist = sensor_data.get("imu_euler", [0,0,0])[1]
        if "wrist_flex_joint" in self.ghost_joints:
            p.setJointMotorControl2(self.ghost_id, self.ghost_joints["wrist_flex_joint"], p.POSITION_CONTROL, targetPosition=wrist)

    def update_agent(self, motor_commands):
        for f in ["thumb", "index", "middle", "ring", "pinky"]:
            self._apply_curl(self.agent_id, self.agent_joints, f, motor_commands[f"{f}_angle"])
        
        if "wrist_flex_joint" in self.agent_joints:
            p.setJointMotorControl2(self.agent_id, self.agent_joints["wrist_flex_joint"], p.POSITION_CONTROL, targetPosition=motor_commands["wrist_angle"])

    def step(self):
        p.stepSimulation()