import grid2op
import numpy as np
from typing import Callable
from grid2op.Reward import BaseReward
from grid2op.gym_compat import GymEnv, BoxGymActSpace
from stable_baselines3 import PPO

# 1. Define the Learning Rate Scheduler
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: The initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining starts at 1.0 and linearly decreases to 0.0
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

# 3. Load the environment with the Composite Reward
env = grid2op.make("l2rpn_case14_sandbox", reward_class=CompositeRoutingReward)
gym_env = GymEnv(env)
gym_env.action_space = BoxGymActSpace(env.action_space)

print("Initializing PPO Agent with Learning Rate Scheduler...")
# 4. Create the brain WITH the scheduler
# The default PPO learning rate is 0.0003, which we will gradually decay to 0
model = PPO("MultiInputPolicy", gym_env, verbose=1, 
            learning_rate=linear_schedule(0.0003),
            tensorboard_log="./ppo_composite_500k_tuned_tensorboard/")

print("Starting training (500,000 steps). This will take a while...")
# 5. Train the agent
model.learn(total_timesteps=500000)

# 6. Save the updated brain as Version 6
model.save("ppo_power_router_v6_tuned")
print("Training complete! Version 6 of the AI brain successfully saved.")