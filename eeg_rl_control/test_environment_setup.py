import pybullet as p
import pybullet_data
import time
import math
import os

# 1. Initialization
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])

planeId = p.loadURDF("plane.urdf")

# 2. Model Loading
agent_start_pos = [0, 0, 0]
ghost_start_pos = [0, 0.7, 0]
agent_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
ghost_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Construct absolute paths to URDF files
script_dir = os.path.dirname(os.path.realpath(__file__))
robotic_arm_path = os.path.join(script_dir, "/urdf/robotic_arm.urdf")
glove_path = os.path.join(script_dir, "/urdf/glove.urdf")

robotic_arm_id = p.loadURDF(robotic_arm_path, agent_start_pos, agent_start_orientation, useFixedBase=True)
glove_id = p.loadURDF(glove_path, ghost_start_pos, ghost_start_orientation, useFixedBase=True)


# 3. Joint Introspection
def get_joint_map(body_id):
    joint_map = {}
    num_joints = p.getNumJoints(body_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(body_id, i)
        joint_name = joint_info[1].decode('UTF-8')
        joint_map[joint_name] = i
    return num_joints, joint_map

agent_num_joints, agent_joint_map = get_joint_map(robotic_arm_id)
print(f"Agent Arm (ID: {robotic_arm_id}) has {agent_num_joints} joints.")
print(f"Joint map: {agent_joint_map}")


ghost_num_joints, ghost_joint_map = get_joint_map(glove_id)
print(f"Ghost Arm (ID: {glove_id}) has {ghost_num_joints} joints.")
print(f"Joint map: {ghost_joint_map}")


# 4. Control Verification Loop
try:
    while True:
        # Control the Ghost Arm (Kinematic)
        ghost_wrist_joint_name = 'wrist_flex_joint'
        if ghost_wrist_joint_name in ghost_joint_map:
            ghost_wrist_joint_index = ghost_joint_map[ghost_wrist_joint_name]
            angle = 0.5 * math.sin(time.time() * 3) # Oscillate between -0.5 and 0.5
            p.resetJointState(glove_id, ghost_wrist_joint_index, targetValue=angle)

        # Control the Agent Arm (Dynamic)
        agent_elbow_joint_name = 'elbow_flex_joint'
        if agent_elbow_joint_name in agent_joint_map:
            agent_elbow_joint_index = agent_joint_map[agent_elbow_joint_name]
            p.setJointMotorControl2(
                bodyUniqueId=robotic_arm_id,
                jointIndex=agent_elbow_joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=1.0,
                force=100
            )

        p.stepSimulation()
        time.sleep(1./240.)

except p.error as e:
    print(f"PyBullet error: {e}")
except KeyboardInterrupt:
    print("Simulation stopped by user.")
finally:
    p.disconnect()
