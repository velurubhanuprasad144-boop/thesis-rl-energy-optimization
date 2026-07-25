# Simulation Based Reinforcement Learning Framework for Energy Optimised Routing in Urban Smart Grids

This repository contains the implementation of a reinforcement learning (RL) framework designed to manage and optimize energy routing within urban smart grids. The goal of the agent is to prevent cascading blackouts by dynamically reconfiguring the grid topology and balancing line capacities under varying load conditions.

---

## 1. Architecture & Technology Stack

### Simulation Environment: Grid2Op
Grid2Op is utilized as the core simulation backend because it provides a highly realistic, non-linear physics engine tailored specifically for power grid operations. Unlike standard RL environments, Grid2Op accurately models the complex thermodynamics of power lines, alternating current (AC) power flows, and the localized topological actions required for grid management, such as busbar splitting and line switching. 

### Dataset: `l2rpn_case14_sandbox`
The agent is trained and evaluated on the `l2rpn_case14_sandbox` environment. This dataset simulates a lightweight 14-substation grid. It was selected because it is computationally efficient for rapid prototyping and local training, yet perfectly encapsulates the core challenges of smart grid routing, including regional load fluctuations, transmission bottlenecks, and the risk of cascading failures.

### RL Framework: Stable-Baselines3 (SB3)
Stable-Baselines3 provides a set of reliable, industry-standard implementations of deep reinforcement learning algorithms. It integrates seamlessly with OpenAI Gym-style environments and provides robust logging tools (like TensorBoard) for tracking the agent's optimization metrics across thousands of timesteps.

### Algorithm: Proximal Policy Optimization (PPO)
PPO is chosen as the driving algorithm for its balance of sample efficiency and operational stability. Power grid environments feature complex, high-dimensional state spaces. PPO’s clipped objective function mathematically prevents destructively large policy updates during training, ensuring that the AI does not violently overwrite stable routing strategies while exploring new optimizations. 

---

## 2. Codebase Structure

### `test_env.py`
This script acts as the environmental diagnostic tool. It verifies that the Grid2Op backend is correctly installed and that the physics engine can successfully load the `l2rpn_case14_sandbox` dataset without throwing backend errors. 

### `baseline.py`
This script establishes the "Do-Nothing" control metric for the framework. It evaluates the natural decay of the grid when subjected to standard load variations without any topological interventions. 
*   **Current Baseline:** The unmanaged grid survives for **807 timesteps** before suffering a cascading failure. Any successful AI agent must significantly surpass this survival threshold.

### `train_agent.py`
This is the core training module. Standard Grid2Op environments output actions and observations as highly complex dictionaries (a mix of discrete and continuous variables). Because standard RL algorithms require flattened numerical arrays, `train_agent.py` utilizes `GymEnv` and `BoxGymActSpace` to dynamically wrap the Grid2Op dictionary space into a continuous `Box` array. This crucial transformation allows the SB3 PPO `MultiInputPolicy` to interpret the grid's state and output valid topological actions.

---

## 3. Experimental Observations & Training History

The development of the agent has progressed through multiple iterations, primarily focused on balancing grid survival with strict line capacity constraints. 

*   **Version 1: Unconstrained Survival (50,000 steps)**
    *   **Result:** Survived 5,414 timesteps (evaluated deterministically).
    *   **Observation:** Utilizing the default reward function, the agent successfully learned to keep the grid alive, greatly outperforming the 807-step baseline. However, it only optimized for basic survival and ignored energy-optimized routing (balancing the load across power lines).

*   **Version 2: Strict Constraint & Reward Shaping (50,000 steps)**
    *   **Result:** Survived 2,990 timesteps.
    *   **Observation:** A custom reward function was introduced to heavily penalize high transmission line capacity ($\\rho$). The survival time dropped, but the agent successfully learned to make active, load-balancing topological interventions. 

*   **Version 3: Reward Hacking (500,000 steps)**
    *   **Result:** Survived 516 timesteps (Total Score: 32,432).
    *   **Observation:** When given extended training time on the strict constraint reward, the AI exhibited a classic "Reward Hacking" phenomenon. It learned highly aggressive, mathematically exploitative routing tricks that scored massive points in the short term, but pushed the grid into an unrecoverable, fragile state, sacrificing long-term survival.

*   **Version 4: The Composite Reward (50,000 steps)**
    *   **Result:** Survived 3,110 timesteps.
    *   **Observation:** The reward function was mathematically re-weighted to include both a guaranteed *survival bonus* and an *optimization bonus*. This successfully cured the reward hacking from Version 3, forcing the agent to find a sustainable equilibrium between keeping the lights on and keeping the lines cool.

*   **Version 5: Catastrophic Forgetting (500,000 steps)**
    *   **Result:** Survived 968 timesteps.
    *   **Observation:** When training the Composite Reward model for 500,000 steps using PPO's default constant learning rate, the agent suffered from Catastrophic Forgetting. As training progressed, the agent took aggressively large learning steps, overwriting its foundational survival strategies and causing policy destabilization.

*   **Version 6: Learning Rate Scheduler & Overfitting (500,000 steps)**
    *   **Result:** Survived 1,091 timesteps (Total Score: 61,300).
    *   **Observation:** A linear learning rate scheduler was injected into the PPO algorithm to gradually decay the update size to zero. This prevented catastrophic forgetting. However, the agent's massive score alongside its early death indicated severe **Policy Brittleness/Overfitting**. The agent memorized highly complex, hyper-optimized topological maneuvers that scored well but shattered instantly when confronted with unexpected variations in the load distribution.