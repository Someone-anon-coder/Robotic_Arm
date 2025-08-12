# EEG Controlled Remote Robotic Arm: RL Simulation

## 1. Project Overview

This project focuses on developing a sophisticated Reinforcement Learning (RL) system to control a high-degree-of-freedom (27-DOF) robotic arm. The primary goal is to enable the robotic arm to intuitively and smoothly imitate the movements of a human user wearing a sensor-equipped glove.

The simulation phase is designed to train one or more RL agents within a physics-based environment (PyBullet) without requiring the physical hardware. A key challenge this project addresses is the **inference of upper-arm motion** (shoulder and elbow), as the control system only receives direct sensor input from the user's hand and wrist via an Inertial Measurement Unit (IMU) and flex sensors.

---

## 2. Core Simulation Concept: The Agent and the "Ghost"

To effectively train the agent, we employ a dual-model paradigm:

*   **The Agent (`robotic_arm.urdf`)**: This is the active, physics-enabled model of the robotic arm. The RL agent learns to apply forces/torques to its 27 joints to control its movement and achieve a target pose.

*   **The Target or "Ghost" (`glove.urdf`)**: This is a kinematic, non-colliding visual representation of the desired pose. In the simulation, we programmatically set the joint angles of this "ghost" arm to serve as the goal. The agent observes the state of the ghost to understand what it needs to imitate.

This "ghosting" setup provides a clear, continuous stream of `(current_state, goal_state)` data, which is essential for the RL training loop.

---

## 3. System Architecture

### 3.1. Observation Space (The Agent's "Senses")

The agent's decision-making is based on a comprehensive set of inputs:

*   **Simulated Flex Sensor Data**: We simulate 14 flex sensors by calculating the real-time Euclidean distance between `tracker` links on the ghost's fingers/wrist and their corresponding `flexed` base links. This provides a realistic, continuous value that mirrors how a physical flex sensor would behave.
*   **Simulated IMU Data**:
    *   Linear and angular velocities from the "ghost" hand's IMU (`IMU_A`).
    *   Quaternion orientation from the "ghost" hand's IMU.
    *   Reference data from a static base IMU (`IMU_R`).
*   **Historical Motion Context**: To give the agent a sense of dynamics, the observation includes the last 10 frames of the ghost IMU's linear velocities (`dx, dy, dz`), providing a short-term memory of the target's trajectory.
*   **Proprioception**: The agent's own state, including the current joint positions and velocities for all 27 motors of the `robotic_arm.urdf`.

### 3.2. Action Space (The Agent's "Muscles")

*   **Type**: Continuous.
*   **Dimensions**: A vector of 27 values, where each value corresponds to the target velocity or position for one of the robotic arm's servo motors.
*   **Algorithm**: We will use **Soft Actor-Critic (SAC)**, a state-of-the-art off-policy algorithm ideal for high-dimensional continuous control tasks due to its sample efficiency and stability.

### 3.3. Reward Function (The Agent's "Motivation")

The reward signal is carefully engineered to shape the desired behavior:

1.  **Hierarchical Importance**: Rewards are weighted to prioritize learning the more difficult, inferred movements first: `Shoulder > Elbow > Wrist > Palm > Thumb > Fingers`.
2.  **Goal Achievement**: A large, one-time reward is given for matching the ghost's pose within a tight tolerance, preventing jittery behavior around the goal.
3.  **Path Guidance**: A continuous, dense reward is provided for reducing the distance between the agent's and the ghost's key points (e.g., fingertips), guiding the agent toward the correct pose.
4.  **Efficiency Penalty**: A small negative reward is applied at each timestep to encourage the agent to reach the goal quickly.
5.  **Deviation Penalty**: A significant penalty is applied if the agent moves *away* from the target, discouraging unproductive exploration.

### 3.4. Architectural Strategy: Monolithic to Hierarchical

*   **Phase 1: Monolithic SAC**: We will start by training a single SAC agent that takes the full observation and outputs the full 27-DOF action. This is crucial for debugging the environment and establishing a performance baseline.
*   **Phase 2: Hierarchical SAC (H-SAC)**: To tackle the high dimensionality, we will evolve the architecture into a hierarchical model.
    *   **High-Level "Manager" Agent**: Observes the overall task (e.g., ghost hand position) and sets a simpler sub-goal for the low-level agent (e.g., a target 6D pose for the wrist).
    *   **Low-Level "Controller" Agent**: Receives the sub-goal from the manager and is responsible for the complex, high-frequency motor control required to achieve it.

---

## 4. Project File Structure

```
eeg_rl_control/
├── urdf/
│   ├── robotic_arm.urdf      # Physics-enabled model for the agent
│   └── glove.urdf            # Kinematic model for the target "ghost"
│
├── environment/
│   ├── __init__.py
│   └── arm_env.py            # Gym-compliant custom environment for PyBullet
│
├── agents/
│   ├── __init__.py
│   ├── sac_agent.py          # Implementation of the monolithic SAC agent
│   └── hrl_agent/            # Directory for the Hierarchical RL models
│       ├── __init__.py
│       ├── high_level.py     # High-level "Manager" agent
│       └── low_level.py      # Low-level "Controller" agent
│
├── utils/
│   ├── __init__.py
│   └── helpers.py            # Helper functions (e.g., quaternion math, sensor simulation)
│
├── config.py                 # Central file for all hyperparameters and settings
│
├── train.py                  # Main script to launch training
│
└── README.md                 # Project documentation
```

---

## 5. Setup & Installation (Placeholder)

1.  **Clone the repository:**
    ```sh
    git clone [your-repo-url]
    cd eeg_rl_control
    ```
2.  **Create a virtual environment and install dependencies:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *A `requirements.txt` file should include libraries such as:*
    *   `pybullet`
    *   `gymnasium` (or `gym`)
    *   `torch` (or `tensorflow`)
    *   `numpy`
    *   `stable-baselines3` (optional, if used for SAC implementation)

---

## 6. Usage

To begin training the Reinforcement Learning agent, run the main training script from the root directory:

```sh
python train.py
```

You can modify the `config.py` file to adjust hyperparameters, change simulation settings, or switch between the monolithic and hierarchical training modes.

---

## 7. Future Work

*   Implement the Hierarchical SAC (H-SAC) architecture for improved performance on this complex task.
*   Develop a robust curriculum learning strategy, starting with simpler tasks (e.g., only wrist and finger matching) before moving to full-arm imitation.
*   Integrate the trained models with the physical hardware stack: Nvidia Jetson Orin for model inference and an Arduino/microcontroller for processing glove sensor data.