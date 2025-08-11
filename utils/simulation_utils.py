# eeg_robotic_arm_rl/utils/simulation_utils.py

import pybullet as p
import pybullet_data

def setup_simulation(config, connect_mode=p.GUI):
    """
    Initializes the PyBullet simulation environment.
    """
    physics_client = p.connect(connect_mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.setGravity(*config.GRAVITY)
    p.setTimeStep(config.SIM_TIME_STEP)
    
    p.loadURDF("plane.urdf")
    
    # Load the primary robotic arm
    robot_id = p.loadURDF(
        config.ROBOT_URDF_PATH,
        config.ROBOT_START_POS,
        config.ROBOT_START_ORN,
        useFixedBase=True
    )
    
    # Load the "ghost" glove arm
    glove_id = p.loadURDF(
        config.GLOVE_URDF_PATH,
        config.GLOVE_START_POS,
        config.GLOVE_START_ORN,
        useFixedBase=True
    )
    
    # Make the glove semi-transparent for clear visualization
    for joint_index in range(p.getNumJoints(glove_id)):
        p.changeVisualShape(glove_id, joint_index, rgbaColor=[0.5, 0.5, 0.5, 0.5])

    return physics_client, robot_id, glove_id

def print_joint_info(robot_id, robot_name="Robot"):
    """
    Prints the name, index, and other details of each joint in the model.
    """
    num_joints = p.getNumJoints(robot_id)
    print(f"--- Joint Information for: {robot_name} (ID: {robot_id}) ---")
    print(f"Total number of joints: {num_joints}")
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_index = info[0]
        joint_name = info[1].decode('utf-8')
        joint_type = info[2]
        
        print(f"  Index: {joint_index:<3} | Name: {joint_name:<25} | Type: {joint_type}")
    print("-" * 50)

def get_joint_mappings(robot_id):
    """
    Creates mappings from joint name to index and vice-versa.
    Filters out fixed joints.
    """
    name_to_index = {}
    index_to_name = {}
    num_joints = p.getNumJoints(robot_id)
    
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        joint_type = info[2]
        
        # We only care about controllable (revolute) joints
        if joint_type == p.JOINT_REVOLUTE:
            name_to_index[joint_name] = i
            index_to_name[i] = joint_name
            
    return name_to_index, index_to_name