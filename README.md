# AI for Smart Grids: Optimizing Topology with Reinforcement Learning

This repository contains the official codebase, environments, and trained artificial neural network models for my Master's thesis research (Fall 2026). The project investigates the application of Deep Reinforcement Learning (DRL)—specifically Proximal Policy Optimization (PPO)—to autonomously manage power grid topology, prevent cascading failures, and optimize transmission line loading.

## 📌 Project Abstract
As modern power grids integrate higher volumes of volatile renewable energy, traditional mathematical heuristic solvers struggle with the computational bottlenecks required for real-time dispatching. This project frames electrical grid topology management as a Markov Decision Process (MDP). By training a PPO agent within the Grid2Op framework, the AI learns to actively manipulate busbars and route power dynamically, balancing strict physical grid constraints with maximum operational efficiency.


## ⚙️ Environment and Agent Architecture
* **Simulation Environment:** [Grid2Op](https://grid2op.readthedocs.io/) running the `l2rpn_case14_sandbox` (based on the IEEE 14-bus system).
* **Action Space:** A combinatorial discrete-continuous hybrid, mapped via `BoxGymActSpace` to allow the PPO agent to handle complex topological permutations (node splitting, line reconnections).
* **Algorithm:** Proximal Policy Optimization (PPO) via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), utilizing an Actor-Critic Multi-Layer Perceptron (MLP) policy.

## 🧠 Training Methodology & Customizations
1. **Strict Reward Shaping:** Early agents easily learned to "game" the system by maximizing survival time at the cost of heavily congested lines. The final model introduces a normalized reward function that heavily penalizes transmission line bottlenecks.
2. **Mitigating Catastrophic Forgetting:** Continuous grid environments often cause late-stage performance collapse. The V9 architecture leverages an `EvalCallback` to evaluate the agent on an isolated validation set every 10,000 timesteps, permanently saving the peak-performing weights rather than relying on the end-of-run state.

## 🚀 Installation & Usage

**1. Clone the Repository (Requires Git LFS)**
Because this repository natively hosts the heavy evaluation dataset, Git Large File Storage must be initialized prior to cloning.

    git lfs install
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    git lfs pull

**2. Virtual Environment & Dependencies**
macOS users are encouraged to run this in an isolated virtual environment (`thesis_env`).

    python3 -m venv thesis_env
    source thesis_env/bin/activate
    pip install grid2op stable-baselines3[extra] torch pandas

**3. Running the Evaluations**
The `evaluate_agent.py` script accepts arguments to hot-swap the agent's brain seamlessly.

    # Test the baseline (natural grid failure)
    python baseline.py

    # Evaluate the final optimized V9 agent
    python evaluate_agent.py 

    # Evaluate the intermediate pre-eval agent
    python evaluate_agent.py --model model_pre_eval 

    # Evaluate the raw V1 agent
    python evaluate_agent.py --model model_v1 

## 📊 Benchmark Results & Analysis
All agents were evaluated against the exact same weather and load demand scenario (Environment Seed: 42). 

| Agent Version | Survival (Timesteps) | Total Operational Score | Analytical Takeaway |
| :--- | :--- | :--- | :--- |
| **Baseline** | 807 | 51,564.48 | The natural failure point of the grid without intervention. |
| **Model V1** | 4824 | 36,667.65 | Brute-force survival strategy. Avoided blackouts but operated the grid with extreme inefficiency. |
| **Model Pre-Eval** | 2533 | 143,241.47 | Improved efficiency via custom rewards, but succumbed to catastrophic forgetting during late-stage training. |
| **Model Final (V9)** | **2974** | **181,144.39** | The optimal balance. Extracted at peak performance via callback; safely navigated peak loads while maximizing efficiency. |

*Note: While V1 survived strictly longer in this specific chronic, its operational score demonstrates unacceptable real-world thermal loading. The Final V9 agent successfully balances critical survival with necessary operational safety margins.*