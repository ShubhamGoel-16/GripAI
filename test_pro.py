import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import time
import numpy as np

# 1. Initialize the environment in 'human' mode to see the live 3D rendering
# We use the standard PandaReach-v3 for testing
env = gym.make("PandaReach-v3", render_mode="human")

# 2. Load the expert model you just saved
# Ensure the filename matches what was printed in your training log
model_path = "panda_expert_muscles.zip"

try:
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path, env=env)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# 3. Evaluation Loop
num_episodes = 10
print(f"Starting evaluation for {num_episodes} episodes. Press Ctrl+C to stop.")

try:
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not (terminated or truncated):
            # Predict the best action (deterministic=True is key for testing)
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action in the simulator
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Print live distance to goal in the terminal
            dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
            print(f"Step: {step_count:2d} | Distance to Goal: {dist:.4f}m", end='\r')
            
            # Small sleep to make the movement look more natural to the human eye
            time.sleep(0.01)

        if info.get("is_success"):
            print(f"\n✨ SUCCESS! Target reached in {step_count} steps.")
        else:
            print(f"\n❌ Timed out after {step_count} steps.")
            
        time.sleep(1) # Pause before resetting to next target

except KeyboardInterrupt:
    print("\n\nTesting stopped by user.")
finally:
    env.close()
    print("Environment closed.")