import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import numpy as np

# 1. Initialize the environment with the GUI enabled
# render_mode="human" is what opens the PyBullet window
env = gym.make("PandaReach-v3", render_mode="human")

# 2. Setup the SAC (Soft Actor-Critic) model
# We use MultiInputPolicy because the observation is a dictionary 
# containing 'observation', 'achieved_goal', and 'desired_goal'
model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="./panda_reach_tensorboard/")

# 3. Start the learning process
print("Training started! The PyBullet window should now be visible.")
print("The arm will move randomly at first as it explores.")

# 10,000 steps is a good starting point to see it learn the 'Reach' task
model.learn(total_timesteps=10000)

# 4. Save your trained agent
model.save("panda_reach_agent")
print("Training complete! Model saved as 'panda_reach_agent'.")

# Close the window
env.close()