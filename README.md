# Simulation Based Reinforcement Learning Framework for Energy Optimised Routing in Urban Smart Grids

This repository contains the methodology and code for training a Reinforcement Learning (RL) agent to optimize power routing and prevent blackouts in an urban smart grid simulation.

## Project Architecture
* **Simulation Environment:** [Grid2Op](https://github.com/rte-france/Grid2Op)
* **Dataset:** `l2rpn_case14_sandbox` (A lightweight 14-substation grid)
* **RL Framework:** [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
* **Algorithm:** Proximal Policy Optimization (PPO)

## Repository Structure
1. `test_env.py` - Verifies the Grid2Op installation and loads the dataset.
2. `baseline.py` - Runs the "Do-Nothing" control baseline. 
   * *Current Baseline:* The grid survives for **807 timesteps** before suffering a cascading failure without intervention.
3. `train_agent.py` - Wraps the Grid2Op dictionary action space into a continuous Box array and trains an initial PPO agent.

## Progress
- [x] Local environment configuration and dependency resolution
- [x] Grid2Op initialization and dataset local download
- [x] Established "Do-Nothing" baseline metric
- [x] First successful PPO training loop (Action Space flattened)
- [ ] Evaluate initial agent performance against baseline
- [ ] Extended agent training phase