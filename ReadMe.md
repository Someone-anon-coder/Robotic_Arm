# EEG-Controlled Remote Robotic Arm (Phase 1: Glove Control)

## 1. Project Overview

This project aims to develop an advanced control system for a high-degree-of-freedom (27-DOF) robotic arm. The ultimate vision is to control the arm using EEG (Electroencephalography) signals to interpret user intent. **Phase 1** establishes the core control system using a sensor-equipped glove as the primary input device.

The central challenge is to create a system that not only mimics the user's hand gestures but intelligently infers the position of the entire arm (shoulder and elbow) from the motion and orientation of the hand. The system must provide smooth, intuitive, and real-time control.

To solve this, we are developing a **Hierarchical Reinforcement Learning (HRL)** model. This approach breaks down the complex task of controlling 27 joints into a manageable hierarchy of decision-making agents, enabling efficient learning and real-time performance. The entire system is first developed and trained in a realistic physics simulation (PyBullet) before deployment on physical hardware.

---

## 2. Key Features

- **27-DOF Robotic Arm:** Kinematically detailed model from shoulder to fingertip, enabling human-like movements.
- **Hierarchical Reinforcement Learning (HRL):** Multi-layered control architecture dividing the complex control problem into simpler sub-problems.
- **Soft Actor-Critic (SAC):** Core RL algorithm for sample efficiency and stability in continuous control tasks.
- **Intelligent Arm Pose Inference:** RL challenge to map hand/wrist sensor data (IMU, Flex Sensors) to shoulder and elbow joint configurations.
- **Simulation-First Development:** Uses PyBullet for rapid prototyping, safe training, and debugging.
- **Modular and Extensible Codebase:** Scalable project structure for future EEG integration.

---

## 3. System Architecture

The project is divided into three core components:

### 3.1. User Input: Sensor Glove

The glove provides ground-truth data for the robotic arm to follow. In simulation, this is represented by a `glove.urdf` model.

**Sensors:**
- **14 Flex Sensors:** Simulated by calculating Euclidean distance between tracker links and flexed anchor points.
    - 4 for finger flexion/extension (index, middle, ring, pinky)
    - 4 for finger adduction/abduction
    - 3 for thumb (flexion/extension, adduction/abduction, opposition/reposition)
    - 2 for wrist (flexion/extension, adduction/abduction)
    - 1 for palm bending (pinky phalanx to thumb)
- **2 IMU Sensors:**
    - **IMU_a (Actual):** On the back of the hand, provides velocity and orientation.
    - **IMU_r (Reference):** Fixed at robot base, provides a stable reference frame.

### 3.2. Robotic Arm: Learning Agent

Physical manifestation of the control system, represented by `robotic_arm.urdf`.

**Degrees of Freedom (DOF):**
- Shoulder (2 DOF): Pan, Tilt
- Elbow (2 DOF): Flexion, Rotation
- Wrist (2 DOF): Flexion/Extension, Adduction/Abduction
- Palm (1 DOF): Metacarpal block flexion
- Fingers (16 DOF): 4 DOF per finger
- Thumb (4 DOF): Opposition/reposition, flexion joints

### 3.3. Control System: Hierarchical RL Brain

Three-tier hierarchy:

- **Tier 1: High-Level Coordinator (`CoordinatorAgent`)**
    - **Purpose:** Sets strategic goals for the arm.
    - **Observation:** IMU data, summarized hand state.
    - **Action:** Outputs target wrist position and orientation.
    - **Timescale:** Acts infrequently (e.g., every 15-20 simulation steps).

- **Tier 2: Mid-Level Controllers (`ArmController`, `HandController`)**
    - **ArmController:** Solves inverse kinematics for upper arm.
        - **Observation:** Shoulder/elbow joint angles, wrist subgoal.
        - **Action:** Target joint angles for shoulder/elbow.
    - **HandController:** Translates finger sensor data into wrist/hand pose.
        - **Observation:** Flex sensor values, wrist/hand joint states.
        - **Action:** Target joint angles for wrist, palm, fingers.

- **Tier 3: Low-Level Executor (SAC Policy)**
    - **Purpose:** Executes commands from Tier 2 smoothly and efficiently.
    - **Observation:** Target joint angles, current joint angles/velocities.
    - **Action:** Final torque/force commands for all motors.
    - **Reward:** Minimize error, penalize jerky movements and excessive energy use.

---

## 4. Hardware & Software Stack

**Target Hardware:**
- Compute: NVIDIA Jetson Orin
- Glove Interface: Arduino or similar microcontroller
- Sensors: Flex sensors, IMU (e.g., MPU-6050)
- Actuators: 27 servo motors

**Simulation Software:**
- Physics Engine: PyBullet
- RL Environment: Gymnasium (formerly OpenAI Gym)
- Deep Learning: PyTorch or TensorFlow
- Programming Language: Python 3.x

---

## 5. Simulation Environment (`RoboticArmEnv`)

Training occurs within a custom Gymnasium environment.

- **Training Paradigm:** Two models—visible robotic arm and semi-transparent "ghost" glove. The robotic arm is rewarded for matching the ghost arm's pose.
- **Reward Function:**
    - **Pose Matching Reward:** High reward for minimizing distance between robotic and ghost arm links.
    - **Action Smoothness Penalty:** Penalizes jerky motion.
    - **Energy Penalty:** Penalizes high motor torques.
    - **Joint Limit Penalty:** Large penalty for exceeding joint limits.

---

## 6. Project Directory Structure

```
eeg_robotic_arm_project/
├── README.md                # This file
├── urdf/
│   ├── robotic_arm.urdf     # URDF for the learning agent
│   └── glove.urdf           # URDF for the target "ghost" arm and sensor simulation
├── simulation/
│   ├── environment.py       # Main Gymnasium environment class (RoboticArmEnv)
│   └── sensor_handler.py    # Simulates IMU and Flex Sensor data from the glove model
├── rl_agents/
│   ├── sac.py               # Soft Actor-Critic algorithm
│   ├── hierarchy.py         # HRL architecture (Coordinator, Controllers)
│   └── policies.py          # Actor and Critic network definitions
├── configs/
│   └── main_config.py       # Centralized configuration (hyperparameters, joint names, etc.)
├── utils/
│   └── helpers.py           # Utility functions (quaternion math, reward shaping, etc.)
└── train.py                 # Main script to initialize environment and start training loop
```

---

## 7. How to Run

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

### Visualization

A script will be added to load a pre-trained model and visualize its performance in the PyBullet GUI.

---

## 8. Future Work

- **Phase 2: EEG Integration:** Develop a signal processing pipeline to translate raw EEG data into high-level commands for the CoordinatorAgent. The HRL model from Phase 1 will serve as the "spinal cord" for execution.
- **Sim2Real Transfer:** Fine-tune simulation-trained models on the physical robotic arm to bridge the "reality gap."
- **Task-Oriented Learning:** Train the arm on specific tasks (e.g., picking, placing, manipulating objects) using task-completion as the primary reward signal.

---
