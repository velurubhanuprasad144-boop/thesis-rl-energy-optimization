import argparse
import grid2op
from grid2op.gym_compat import GymEnv, BoxGymActSpace
from stable_baselines3 import PPO

# 1. Set up the argument parser
parser = argparse.ArgumentParser(description="Evaluate a trained Grid2Op PPO Agent")
parser.add_argument(
    "--model", 
    type=str, 
    default="model_final", 
    help="Name of the saved model (e.g., model_v1, model_pre_eval, model_final)"
)
args = parser.parse_args()

# 2. Load the exact same environment and wrappers
env = grid2op.make("l2rpn_case14_sandbox")
gym_env = GymEnv(env)

# Convert the complex Dict action space into a simple continuous Box array
gym_env.action_space = BoxGymActSpace(env.action_space)

# 3. Load your newly trained AI brain dynamically
model_path = f"{args.model}.zip"
print(f"Loading the trained PPO agent from: {model_path}...")
# Note: Passing gym_env instead of env ensures the agent sees the correct Box action space
model = PPO.load(model_path, env=gym_env) 

# 4. Reset the grid for a fresh run
# Force the environment to use the exact same scenario every time
env.seed(42) 
obs = gym_env.reset()
# Handle different versions of Gym returning 1 or 2 variables on reset
if isinstance(obs, tuple):
    obs = obs[0]

done = False
step_count = 0
total_reward = 0

print("Handing control of the grid to the AI...")

# 5. Let the AI run the grid
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
print(f"Results for Agent: {args.model}")
print(f"The AI survived for {step_count} timesteps.")
print(f"Total AI score: {total_reward:.2f}")