import pybullet as p
import pybullet_data
import yaml
import os

class BiomimeticArmEnv:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 1. Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.config['gravity'])
        p.setRealTimeSimulation(0)
        p.setTimeStep(self.config['time_step'])

        # 2. Load Plane
        p.loadURDF("plane.urdf")

        # 3. Load Models
        # Get the absolute path to the urdf files
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ghost_urdf_path = os.path.join(project_root, self.config['urdf_paths']['ghost'])
        robot_urdf_path = os.path.join(project_root, self.config['urdf_paths']['robot'])

        self.ghost_id = p.loadURDF(ghost_urdf_path,
                                   self.config['ghost_start_pos'],
                                   useFixedBase=True)
        self.robot_id = p.loadURDF(robot_urdf_path,
                                   self.config['robot_start_pos'],
                                   useFixedBase=True)

        # 4. Setup Mappings
        self.joint_map_ghost = {}
        self.joint_map_robot = {}
        self._map_joints(self.ghost_id, self.joint_map_ghost)
        self._map_joints(self.robot_id, self.joint_map_robot)

        self.sim_step_count = 0
        self.debug_text_id = -1

    def _map_joints(self, body_id, target_dict):
        num_joints = p.getNumJoints(body_id)
        for i in range(num_joints):
            info = p.getJointInfo(body_id, i)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                target_dict[joint_name] = i

    def reset(self):
        # For now, just reset the joint states to their initial positions (0)
        for joint_index in self.joint_map_robot.values():
            p.resetJointState(self.robot_id, joint_index, 0)
        for joint_index in self.joint_map_ghost.values():
            p.resetJointState(self.ghost_id, joint_index, 0)
        self.sim_step_count = 0
        # Reset the debug text as well
        if self.debug_text_id != -1:
            p.removeUserDebugItem(self.debug_text_id)
        self.debug_text_id = -1


    def step(self, joint_targets: dict[str, float]):
        # Update Ghost Arm
        for joint_name, target_angle in joint_targets.items():
            if joint_name in self.joint_map_ghost:
                p.resetJointState(self.ghost_id, self.joint_map_ghost[joint_name], target_angle)

        # Update Robotic Arm
        pid_params = self.config['motor_pid']
        for joint_name, target_angle in joint_targets.items():
            if joint_name in self.joint_map_robot:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=self.joint_map_robot[joint_name],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=pid_params['force'],
                    positionGain=pid_params['position_gain'],
                    velocityGain=pid_params['velocity_gain']
                )

        # Visual Feedback
        if self.debug_text_id != -1:
            p.removeUserDebugItem(self.debug_text_id)

        # Get robot's base position to display text above it
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        text_pos = [robot_pos[0], robot_pos[1], robot_pos[2] + 0.5] # 0.5m above the base

        self.debug_text_id = p.addUserDebugText(
            f"Sim Step: {self.sim_step_count}",
            text_pos,
            textColorRGB=[1, 0, 0],
            textSize=1.5
        )

        p.stepSimulation()
        self.sim_step_count += 1
