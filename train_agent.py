import grid2op
import numpy as np
from grid2op.Reward import BaseReward
from grid2op.gym_compat import GymEnv, BoxGymActSpace
from stable_baselines3 import PPO

# 1. Define the Composite Reward Function 
class CompositeRoutingReward(BaseReward):
    def initialize(self, env):
        pass
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # Heavy penalty for crashing
        if is_done or has_error or is_illegal:
            return self.reward_min
        
        obs = env.get_obs()
        max_capacity_usage = np.max(obs.rho)
        
        # Component 1: Survival Bonus (Guaranteed points for staying alive)
        survival_bonus = 1.0 
        
        # Component 2: Optimization Bonus (Higher points for cooler lines)
        optimization_bonus = 1.0 - max_capacity_usage
        
        # The AI must survive to collect the optimization points
        reward = survival_bonus + optimization_bonus
        
        return float(max(0.0, reward))

# 2. Load the environment with the new Composite Reward
env = grid2op.make("l2rpn_case14_sandbox", reward_class=CompositeRoutingReward)
gym_env = GymEnv(env)
gym_env.action_space = BoxGymActSpace(env.action_space)

print("Initializing PPO Agent with Composite Reward Function...")
# 3. Create the brain (New TensorBoard folder for the composite run)
model = PPO("MultiInputPolicy", gym_env, verbose=1, tensorboard_log="./ppo_composite_tensorboard/")

print("Starting training (50,000 steps)...")
# 4. Train the agent
model.learn(total_timesteps=50000)

# 5. Save the updated brain as Version 4
model.save("ppo_power_router_v4_composite")
print("Training complete! Version 4 of the AI brain successfully saved.")