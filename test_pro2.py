import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import time
import numpy as np

env = gym.make("PandaReachDense-v3", render_mode="human")  # Match training env

try:
    model = SAC.load("panda_expert_muscles", env=env)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

num_episodes = 10
successes = 0

try:
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        step_count = 0
        print(f"\n--- Episode {episode + 1} ---")

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
            print(f"Step: {step_count:2d} | Distance to Goal: {dist:.4f}m", end='\r')
            time.sleep(0.01)

        if info.get("is_success"):
            print(f"\n✨ SUCCESS in {step_count} steps.")
            successes += 1
        else:
            print(f"\n❌ Timed out after {step_count} steps.")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    env.close()
    print(f"\n📊 Final Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print("Environment closed.")