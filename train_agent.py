import grid2op
import numpy as np
from grid2op.Reward import BaseReward
from grid2op.gym_compat import GymEnv, BoxGymActSpace
from stable_baselines3 import PPO

# 1. Define the Custom Reward Function for Optimized Routing
class SafeRoutingReward(BaseReward):
    def initialize(self, env):
        pass
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done or has_error or is_illegal:
            return self.reward_min
        
        obs = env.get_obs()
        max_capacity_usage = np.max(obs.rho)
        reward = 1.0 - max_capacity_usage
        
        return float(max(0.0, reward))

# 2. Load the environment and inject the new reward logic
env = grid2op.make("l2rpn_case14_sandbox", reward_class=SafeRoutingReward)
gym_env = GymEnv(env)
gym_env.action_space = BoxGymActSpace(env.action_space)

print("Initializing PPO Agent for 500k Long Run...")
# 3. Create the brain (New TensorBoard folder for the 500k run!)
model = PPO("MultiInputPolicy", gym_env, verbose=1, tensorboard_log="./ppo_500k_tensorboard/")

print("Starting training (500,000 steps). This will take a while...")
# 4. Train the agent for 500k steps
model.learn(total_timesteps=500000)

# 5. Save the updated brain as Version 3
model.save("ppo_power_router_v3_500k")
print("Training complete! Version 3 of the AI brain successfully saved.")