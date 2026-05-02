import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import numpy as np
import time

env = gym.make("PandaReach-v3", render_mode="human")
model = SAC.load("panda_expert_muscles", env=env)

print("🔥 STARTING HEAVY STRESS TESTING")

# --- TEST 1: DYNAMIC TRACKING ---
print("\nTest 1: Dynamic Target Tracking (Moving Goal)")
obs, info = env.reset()
for t in range(500):
    # Manually move the target in a circle
    angle = t * 0.05
    moving_goal = np.array([0.1 * np.cos(angle), 0.1 * np.sin(angle), 0.15])
    
    # Sync simulator and observation
    env.unwrapped.task.goal = moving_goal
    env.unwrapped.task.sim.set_base_pose("target", moving_goal, np.array([0,0,0,1]))
    obs['desired_goal'] = moving_goal
    
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _, info = env.step(action)
    
    if t % 50 == 0:
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        print(f"Tracking Step {t}... Error: {dist:.4f}m")

# --- TEST 2: SENSOR NOISE ---
print("\nTest 2: Robustness to Sensor Noise")
noise_level = 0.02 # 2cm of random noise
successes = 0

for _ in range(10):
    obs, info = env.reset()
    for _ in range(100):
        # ADD NOISE TO OBSERVATION
        noisy_obs = {key: val + np.random.normal(0, noise_level, val.shape) for key, val in obs.items()}
        
        action, _ = model.predict(noisy_obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        if info.get("is_success"):
            successes += 1
            break
print(f"Result: {successes}/10 successful under 2cm noise.")

env.close()