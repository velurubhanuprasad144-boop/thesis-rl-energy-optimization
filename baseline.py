import grid2op

# Load the environment
env = grid2op.make("l2rpn_case14_sandbox")
obs = env.reset()

print("Running the 'Do-Nothing' Baseline...")

total_reward = 0
done = False
step_count = 0

# Run the simulation until the grid fails (done = True)
while not done:
    # An empty action means the agent does absolutely nothing
    action = env.action_space() 
    
    # Step the environment forward in time
    obs, reward, done, info = env.step(action)
    
    total_reward += reward
    step_count += 1

print("Simulation Finished!")
print(f"The grid survived for {step_count} timesteps without any AI intervention.")
print(f"Total baseline score (efficiency): {total_reward:.2f}")