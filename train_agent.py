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

# 2. Define the Composite Reward Function 
class CompositeRoutingReward(BaseReward):
    def initialize(self, env):
        pass
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done or has_error or is_illegal:
            return self.reward_min
        
        obs = env.get_obs()
        max_capacity_usage = np.max(obs.rho)
        
        survival_bonus = 1.0 
        optimization_bonus = 1.0 - max_capacity_usage
        
        reward = survival_bonus + optimization_bonus
        return float(max(0.0, reward))

# 3. Load the primary Training Environment
env = grid2op.make("l2rpn_case14_sandbox", reward_class=CompositeRoutingReward)
gym_env = GymEnv(env)
gym_env.action_space = BoxGymActSpace(env.action_space)

# 4. Load the parallel Evaluation Environment for the Callback
eval_env_raw = grid2op.make("l2rpn_case14_sandbox", reward_class=CompositeRoutingReward)
eval_env = GymEnv(eval_env_raw)
eval_env.action_space = BoxGymActSpace(eval_env_raw.action_space)

# 5. Configure the Evaluation Callback
# This will test the agent every 10,000 steps and save the best version automatically
eval_callback = EvalCallback(eval_env, 
                             best_model_save_path='./logs/best_model/',
                             log_path='./logs/results/',
                             eval_freq=10000, 
                             deterministic=True, 
                             render=False)

print("Initializing PPO Agent with Learning Rate Scheduler & Checkpointing...")
# 6. Create the brain WITH the scheduler
model = PPO("MultiInputPolicy", gym_env, verbose=1, 
            learning_rate=linear_schedule(0.0003),
            tensorboard_log="./ppo_composite_500k_tuned_tensorboard/")

print("Starting training (500,000 steps). The best model will be saved automatically...")
# 7. Train the agent with the callback injected
model.learn(total_timesteps=500000, callback=eval_callback)

# 8. Save the final end-state brain as Version 7 (just to have the final state)
model.save("ppo_power_router_v7_final_state")
print("Training complete! Check the './logs/best_model/' folder for your peak performing agent.")