import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import time

# 1. Load the environment
env = gym.make("PandaReach-v3", render_mode="human")

# 2. Load the trained brain
model = SAC.load("panda_reach_agent", env=env)

# 3. Run the robot
obs, info = env.reset()
print("Running the trained agent... Press Ctrl+C in the terminal to stop.")

try:
    while True:
        # Use the model to predict the next best movement
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Target reached! Resetting...")
            time.sleep(1) # Pause for a second to see the success
            obs, info = env.reset()
except KeyboardInterrupt:
    print("Test stopped.")
    env.close()