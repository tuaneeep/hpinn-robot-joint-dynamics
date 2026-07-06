# H-PINN: Physics-Informed Neural Network for Collaborative Robot Joint Dynamics

This repository implements a **Hybrid Physics-Informed Neural Network (H-PINN)** designed for model prediction and dynamic parameter identification of collaborative robot (Cobot) joints, specifically modeled after the **UR3e** robot.

## 📌 Project Overview
The core of this project is the integration of physical laws—specifically state-space equations—directly into a Recurrent Neural Network (RNN) architecture. By replacing standard activation functions with custom **Runge-Kutta 4th Order (RK4) Cells**, the network functions as an intelligent numerical solver that adheres to physical principles while learning from data.

### Key Capabilities:
*   **Inverse Problem (Parameter Identification):** Automatically extracts hidden physical parameters such as armature resistance, torque constants, inertia, and friction coefficients from observed data.
*   **Forward Problem (State Prediction):** Predicts system responses (current, angular velocity, deflection angle) with high accuracy, even in the presence of noise.
*   **Physical Integrity:** Ensures outputs maintain real-world physical units (e.g., Amperes, rad/s) and comply with conservation laws.

---

## 🚀 Key Features
*   **Custom RK4 Cells:** Implements a 4-stage integration process ($k_1$ to $k_4$) within the hidden layers to solve Ordinary Differential Equations (ODEs).
*   **Weighted MSE (WMSE) Loss:** A custom loss function that normalizes the contribution of states with vastly different scales (e.g., motor speed vs. micro-radian deflection angles).
*   **Physics-Embedded Weights:** Network weights are mapped to actual physical constants ($R, L, J_m, k_t, etc.$) rather than abstract values.
*   **Non-Negative Constraints:** Applied during training to ensure parameters like mass, inertia, and resistance remain physically plausible.

---

## 🏗️ System Architecture
The project develops three levels of increasing complexity:
1.  **Model 1:** Electro-mechanical dynamics of the **BLDC Motor**.
2.  **Model 2:** Mechanical dynamics of the **Harmonic Drive** (nonlinear stiffness and friction).
3.  **Model 3:** Fully integrated **Electro-Mechanical system**.

### Block Diagram
The pipeline consists of: **Input Data (.csv)** → **Tensor Formatting** → **H-PINN Core (RNN with RK4 Cells)** → **WMSE Loss Calculation** → **Adam Optimizer** → **Identified Parameters ($\hat{\theta}$)**.

---

## 🛠️ Installation & Requirements
*   **Language:** Python 3.x
*   **Frameworks:** TensorFlow 2.20.0, Keras 3.13.2
*   **Environment:** Optimized for Google Colab with GPU acceleration.
*   **Data Source:** MATLAB-generated ODE45 simulations or experimental-style data with Gaussian noise.

---

## 📊 Results Summary
*   **Parameter Accuracy:** Achieved a 0.02% error for armature resistance ($R$) and 1.01% for linear stiffness ($k_1$) in ideal conditions.
*   **Prediction Precision:** The model tracks state trajectories with extremely low Mean Absolute Error (MAE), reaching $0.000011$ rad for highly sensitive deflection angles.
*   **Robustness:** Successfully identified motor rotor inertia ($J_m$) with only a 3.23% error even when trained on noisy data mimicking real-world UR3e sensors.
