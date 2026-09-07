import os
import grid2op
import numpy as np
from typing import Callable
from grid2op.Reward import BaseReward
from grid2op.gym_compat import GymEnv, BoxGymActSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# 1. Define the Learning Rate Scheduler
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# 2. Define the Perfectly Normalized Reward
class NormalizedReward(BaseReward):
    def initialize(self, env):
        pass
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done or has_error or is_illegal:
            return self.reward_min
        
        obs = env.get_obs()
        max_capacity_usage = np.max(obs.rho)
        
        # Strict normalization: The maximum possible reward per step is exactly 1.0
        survival_weight = 0.5
        optimization_weight = 0.5
        
        survival_score = 1.0 * survival_weight
        optimization_score = max(0.0, 1.0 - max_capacity_usage) * optimization_weight
        
        reward = survival_score + optimization_score
        return float(reward)

# 3. Load the primary Training Environment
env = grid2op.make("l2rpn_case14_sandbox", reward_class=NormalizedReward)
gym_env = GymEnv(env)
gym_env.action_space = BoxGymActSpace(env.action_space)

# 4. Load the parallel Evaluation Environment for the Callback
eval_env_raw = grid2op.make("l2rpn_case14_sandbox", reward_class=NormalizedReward)
eval_env = GymEnv(eval_env_raw)
eval_env.action_space = BoxGymActSpace(eval_env_raw.action_space)

# 5. Configure the Evaluation Callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/model_final_best/', # Update this path
    log_path='./logs/',
    eval_freq=1000,
    deterministic=True,
    render=False
)


print("Initializing PPO Agent with Normalized Reward (Max 1.0)...")
# 6. Create the brain with the scheduler
model = PPO("MultiInputPolicy", gym_env, verbose=1, 
            learning_rate=linear_schedule(0.0003),
            tensorboard_log="./ppo_normalized_tensorboard/")

print("Starting training (500,000 steps). The peak normalized model will be saved automatically...")
# 7. Train the agent
model.learn(total_timesteps=500000, callback=eval_callback)

# 8. Save the final end-state brain as Version 9
model.save("model_final")
print("Training complete! Check the './models/model_final_best/' folder for your peak performing agent.")