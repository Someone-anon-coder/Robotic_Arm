import pybullet as p
import pybullet_data
import time
import os

class SimEnvironment:
    def __init__(self):
        # 1. Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0) # We will step manually for control

        # 2. Load Plane
        p.loadURDF("plane.urdf")

        # 3. Define Paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "urdf")
        
        # 4. Load Models
        # The Ghost (Input) - Green/Transparent tint ideally, but we use position 
        start_pos_ghost = [0, 0, 0]
        self.ghost_id = p.loadURDF(os.path.join(urdf_path, "glove.urdf"), 
                                   start_pos_ghost, 
                                   useFixedBase=True)

        # The Robot (Output) - Offset by 0.5 meters in Y
        start_pos_robot = [0, 0, 0]
        self.robot_id = p.loadURDF(os.path.join(urdf_path, "robotic_arm.urdf"), 
                                   start_pos_robot, 
                                   useFixedBase=True)

        # 5. Setup Mappings and GUI
        self.joint_map_ghost = {} # {joint_name: joint_index}
        self.joint_map_robot = {} # {joint_name: joint_index}
        self.sliders = {}         # {joint_name: slider_id}

        self._map_joints(self.ghost_id, self.joint_map_ghost)
        self._map_joints(self.robot_id, self.joint_map_robot)
        
        # Create visual labels
        # p.addUserDebugText("GHOST (Input)", [0, 0, 0.8], [0, 0, 0], textSize=1.5)
        # p.addUserDebugText("ROBOT (Output)", [0, 0.5, 0.8], [0, 0, 0], textSize=1.5)

        # Create Sliders for the Ghost Arm
        self._create_sliders()

    def _map_joints(self, body_id, target_dict):
        """
        Iterates through a URDF and maps joint names to indices.
        Only includes movable joints (Revolute/Prismatic).
        """
        num_joints = p.getNumJoints(body_id)
        for i in range(num_joints):
            info = p.getJointInfo(body_id, i)
            # info[1] is name (bytes), info[2] is type
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]

            # We only care about controllable joints (Revolute or Prismatic)
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                target_dict[joint_name] = i
                # Enable the motor for the robot (force=0 lets us control it later)
                p.setJointMotorControl2(body_id, i, controlMode=p.VELOCITY_CONTROL, force=0)

    def _create_sliders(self):
        """
        Creates GUI sliders for every joint found in the Ghost Arm.
        """
        print("Creating Sliders for joints...")
        for name, index in self.joint_map_ghost.items():
            # Get limits from URDF
            info = p.getJointInfo(self.ghost_id, index)
            lower_limit = info[8]
            upper_limit = info[9]
            
            # Default position is usually 0, but let's check if limits allow it
            start_pos = 0
            if start_pos < lower_limit: start_pos = lower_limit
            if start_pos > upper_limit: start_pos = upper_limit

            # Add slider
            slider_id = p.addUserDebugParameter(name, lower_limit, upper_limit, start_pos)
            self.sliders[name] = slider_id

    def step(self):
        """
        Main Control Loop
        1. Read Sliders
        2. Update Ghost (Kinematic / Instant)
        3. Calculate Robot Action (Physics / PID)
        """
        
        for joint_name, slider_id in self.sliders.items():
            # 1. Read Target Value from GUI (Simulating Sensor Input)
            target_angle = p.readUserDebugParameter(slider_id)

            # 2. Update Ghost Arm (Instantaneous visual target)
            ghost_idx = self.joint_map_ghost[joint_name]
            p.resetJointState(self.ghost_id, ghost_idx, target_angle)

            # 3. Update Robotic Arm (Simulated Motor Control)
            # We check if the robot actually has this joint (Robotic arm might differ slightly from glove)
            if joint_name in self.joint_map_robot:
                robot_idx = self.joint_map_robot[joint_name]
                
                # This is the internal PID calculation provided by PyBullet.
                # positionControl simulates a servo motor trying to reach an angle.
                # forces and gains prevent it from "snapping" instantly or overshooting physics.
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=robot_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=100,      # Max motor force (Simulates torque limit)
                    positionGain=0.3, # P-term: Lower = smoother/slower, Higher = Snappier
                    velocityGain=1    # D-term: Damping
                )

        # Advance physics simulation
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    sim = SimEnvironment()
    
    print("\n--- Simulation Started ---")
    print("1. Use the sidebar sliders to control the 'Ghost Arm'.")
    print("2. The 'Robotic Arm' will physically attempt to follow.")
    print("3. Note: The Robot Arm has mass/inertia, so it won't move instantly like the Ghost.")
    
    while True:
        sim.step()
