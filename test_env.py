import grid2op
from stable_baselines3 import PPO

# Initialize the environment with a specific, lightweight sandbox dataset
env = grid2op.make("l2rpn_case14_sandbox")

print("Grid2Op successfully initialized!")
print("Environment name:", env.name)
print("Number of substations:", env.n_sub)
print("Stable-Baselines3 is ready for training.")