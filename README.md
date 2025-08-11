# EEG Controlled Remote Robotic Arm - Phase 1: Glove Control Simulation

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RL Library](https://img.shields.io/badge/RL%20Library-Stable%20Baselines3-red.svg)](https://github.com/DLR-RM/stable-baselines3)
[![Physics Engine](https://img.shields.io/badge/Physics-PyBullet-green.svg)](https://pybullet.org/)

> A Reinforcement Learning environment for training a 27-DOF robotic arm to mimic human motion from glove sensor data in a PyBullet simulation.

This repository contains the simulation environment and Reinforcement Learning (RL) models for Phase 1 of the "EEG Controlled Remote Robotic Arm" project. The primary goal of this phase is to develop a robust control system that allows a 27-DOF (Degree of Freedom) robotic arm to mimic the movements of a user's arm and hand in near real-time, based on input from a sensor-equipped glove.

The entire development and training process is conducted within a PyBullet simulation, creating a "digital twin" of the hardware before it is physically assembled.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Simulation Environment](#simulation-environment)
3.  [Reinforcement Learning Approach](#reinforcement-learning-approach)
4.  [Robot and Glove Models (URDF)](#robot-and-glove-models-urdf)
5.  [Simulated Sensor Inputs](#simulated-sensor-inputs)
6.  [Reward System](#reward-system)
7.  [Setup and Installation](#setup-and-installation)
8.  [How to Run](#how-to-run)
9.  [Project Structure](#project-structure)

## Project Overview

The core objective is to translate human motion into robotic action. A user wearing a glove with flex sensors and an IMU will control a full-arm robotic model (from shoulder to fingertips). The RL agent's role is not just to imitate the user's pose but to do so smoothly, efficiently, and intelligently. A key challenge is inferring the shoulder and elbow positions, as direct sensor inputs are only available for the hand and wrist.

This simulation serves as the training ground for our RL models, allowing for rapid iteration and debugging without the constraints of physical hardware.

## Simulation Environment

We use **PyBullet** as our physics simulator for its speed and simplicity. The simulation environment features a "digital twin" setup:

*   **Robotic Arm (`robotic_arm.urdf`):** This is the agent-controlled model. It is a 27-DOF arm that learns to move based on the RL policy.
*   **Ghost Glove Arm (`glove.urdf`):** This is a semi-transparent "ghost" model that represents the user's desired pose. In the simulation, we programmatically set its joint angles to a target configuration. The RL agent's primary goal is to make the `robotic_arm` match the pose of this `glove` arm.

## Reinforcement Learning Approach

Controlling 27 motors simultaneously presents a massive action and observation space. To tackle this, we employ a **Hierarchical Soft Actor-Critic (H-SAC)** architecture. This breaks the complex problem into a hierarchy of manageable tasks.

#### High-Level Policy (Manager - `agent_brain`)
*   **Responsibility:** Strategic positioning of the arm in 3D space.
*   **Observation:** The target and current world positions/orientations of the hand (from IMU data).
*   **Action:** It does **not** control motors directly. Instead, it outputs a **sub-goal** for the low-level policy, specifying a target pose (position + orientation) for the robot's wrist. It operates on a slower timescale (e.g., every 10-15 steps).

#### Low-Level Policies (Workers)
1.  **Reaching Worker (`agent_arm`):**
    *   **Responsibility:** Controls the shoulder (2 DOF) and elbow (2 DOF) joints.
    *   **Goal:** To move the robot's wrist to the target pose defined by the high-level manager.
    *   **Reward:** Based on minimizing the distance between the actual wrist pose and the target wrist pose.

2.  **Grasping Worker (`agent_hand`):**
    *   **Responsibility:** Controls the wrist (2 DOF), palm (1 DOF), and all finger/thumb joints (20 DOF).
    *   **Goal:** To match the fine-motor hand configuration based on simulated flex sensor data.
    *   **Reward:** Based on minimizing the error between its joint angles and the target angles derived from the flex sensors.

This separation of concerns allows the `agent_arm` to focus on the inverse kinematics problem of reaching, while the `agent_hand` focuses on the complex task of hand mimicry.

## Robot and Glove Models (URDF)

Two URDF files define our models, using simple geometries (cylinders, spheres, boxes) for simulation efficiency.

*   `robotic_arm.urdf`: A 27-DOF model representing the physical robot.
*   `glove.urdf`: An identical kinematic model but with additional `tracker` and `flexed` links. These non-physical links are crucial for simulating our sensor data.

## Simulated Sensor Inputs

Since we don't have physical hardware, we simulate sensor readings using the state of the "ghost" `glove.urdf`:

*   **Flex Sensors (14 inputs):** We define `tracker` links on the fingers/thumb tips and `flexed` links on the palm. The Euclidean distance between each pair in the simulation is calculated and mapped to a realistic flex sensor value (0-1023). This provides the goal for the `agent_hand`.
*   **IMU Data (2 sensors):**
    *   `IMU_A`: A tracker on the back of the ghost `glove` hand. Its world position and orientation history serve as the goal for the `agent_brain`.
    *   `IMU_R`: A reference tracker at the robot's base.

## Reward System

The agent's learning is guided by a carefully engineered reward function:
*   ✅ **Goal Matching:** A large, one-time reward is given for perfectly matching the target pose within a tolerance.
*   🏆 **Hierarchical Weighting:** Rewards for correct shoulder and elbow positioning are weighted more heavily than finger positioning to encourage stable gross motor control first.
*   ➡️ **Proximity Reward:** Continuous rewards are given for reducing the Euclidean distance to the target pose.
*   📉 **Penalties:**
    *   **Step Penalty:** A small negative reward for every step taken encourages efficiency.
    *   **Jerkiness Penalty:** A penalty for high joint velocity/acceleration to promote smooth motion.
    *   **Divergence Penalty:** A penalty for moving away from the target pose.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd eeg_robotic_arm_rl
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    The `requirements.txt` file contains all necessary libraries like `pybullet`, `gymnasium`, `numpy`, and `stable-baselines3`.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Train the model:**
    ```bash
    python train.py --agent_type hierarchical --timesteps 1000000
    ```
    -   `--agent_type`: Choose between `monolithic` (for debugging) and `hierarchical`.
    -   `--timesteps`: The total number of training steps.

2.  **Evaluate a trained model:**
    ```bash
    python evaluate.py --agent_path path/to/saved/model.zip
    ```

## Project Structure

```bash
eeg_robotic_arm_rl/
├── urdf/                 # URDF model files
│   ├── robotic_arm.urdf
│   └── glove.urdf
├── envs/                 # Gymnasium environment definitions
│   ├── base_arm_env.py
│   └── robotic_arm_env.py
├── agents/               # RL agent implementations
│   ├── sac.py
│   └── h_sac.py
├── utils/                # Helper scripts and utility functions
│   ├── simulation_utils.py
│   ├── sensor_utils.py
│   └── reward_utils.py
├── config/               # Configuration files
│   ├── simulation_config.py
│   └── training_config.py
├── train.py              # Main script for training agents
├── evaluate.py           # Script for evaluating trained agents
└── README.md             # This file
```