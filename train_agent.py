import grid2op
from grid2op.gym_compat import GymEnv, BoxGymActSpace
from stable_baselines3 import PPO

# 1. Load the exact same grid environment
env = grid2op.make("l2rpn_case14_sandbox")

# 2. Wrap the environment so the RL library can understand it
gym_env = GymEnv(env)

# NEW: Convert the complex Dict action space into a simple continuous Box array
gym_env.action_space = BoxGymActSpace(env.action_space)

print("Initializing the PPO Agent...")
# 3. Create the brain (MultiInputPolicy natively handles the Dict observation space)
model = PPO("MultiInputPolicy", gym_env, verbose=1)

print("Starting training (this might take a minute)...")
# 4. Train the agent. We will start with a small number of steps to test it.
model.learn(total_timesteps=2000)

# 5. Save the trained weights to your Mac
model.save("ppo_power_router")
print("Training complete! AI brain successfully saved.")