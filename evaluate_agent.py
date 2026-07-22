import grid2op
from grid2op.gym_compat import GymEnv, BoxGymActSpace
from stable_baselines3 import PPO

# 1. Load the exact same environment and wrappers
env = grid2op.make("l2rpn_case14_sandbox")
gym_env = GymEnv(env)

# Convert the complex Dict action space into a simple continuous Box array
gym_env.action_space = BoxGymActSpace(env.action_space)

# 2. Load your newly trained AI brain
print("Loading the trained PPO agent...")
model = PPO.load("ppo_power_router")

# 3. Reset the grid for a fresh run
obs = gym_env.reset()
# Handle different versions of Gym returning 1 or 2 variables on reset
if isinstance(obs, tuple):
    obs = obs[0]

done = False
step_count = 0
total_reward = 0

print("Handing control of the grid to the AI...")

# 4. Let the AI run the grid
while not done:
    # The AI looks at the grid (obs) and decides the best action
    # deterministic=True means the AI uses its best guess, without random exploration
    action, _states = model.predict(obs, deterministic=True)
    
    # We pass the action to the environment
    step_result = gym_env.step(action)
    
    # Handle different versions of Gym (some return 4 items, some return 5)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        obs, reward, done, info = step_result
        
    total_reward += reward
    step_count += 1

print("\n--- Simulation Finished! ---")
print(f"The AI survived for {step_count} timesteps.")
print(f"Total AI score: {total_reward:.2f}")