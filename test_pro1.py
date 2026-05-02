import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import numpy as np
import time

# 1. Initialization
env = gym.make("PandaReach-v3", render_mode="human")
model_path = "panda_expert_muscles.zip"

try:
    print(f"Loading expert model from {model_path}...")
    model = SAC.load(model_path, env=env)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ---------------------------------------------------------
# TEST 1: EXTREME BOUNDARIES
# ---------------------------------------------------------
print("🔥 TEST 1: EXTREME BOUNDARY REACHING 🔥")
print("Testing the absolute corners of the 3D workspace...")

# Define the 8 corners of a 30cm x 30cm x 30cm bounding box
bounds = [-0.15, 0.15]
corners = [[x, y, z] for x in bounds for y in bounds for z in [0.05, 0.30]]

for i, target in enumerate(corners):
    obs, info = env.reset()
    target_pos = np.array(target, dtype=np.float32)
    
    # Force the simulator and observation to use the extreme corner
    env.unwrapped.task.goal = target_pos
    env.unwrapped.task.sim.set_base_pose("target", target_pos, np.array([0.0, 0.0, 0.0, 1.0]))
    obs['desired_goal'] = target_pos
    
    print(f"  -> Corner {i+1}/8 {target}: ", end="")
    
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        obs['desired_goal'] = target_pos # Lock the goal
        
        if info.get("is_success"):
            print(f"✅ Reached in {step+1} steps.")
            break
    else:
        print("❌ FAILED.")
    time.sleep(0.5)

# ---------------------------------------------------------
# TEST 2: SENSOR NOISE INJECTION
# ---------------------------------------------------------
print("\n🌪️ TEST 2: SENSOR NOISE INJECTION 🌪️")
print("Adding 2cm of Gaussian noise to the robot's spatial awareness...")

noise_std_dev = 0.02 # 2cm of noise
successes = 0
attempts = 10

for i in range(attempts):
    obs, info = env.reset()
    print(f"  -> Noisy Attempt {i+1}/{attempts}: ", end="")
    
    for step in range(50):
        # Inject noise directly into the observation arrays before prediction
        noisy_obs = obs.copy()
        noisy_obs['observation'] = obs['observation'] + np.random.normal(0, noise_std_dev, obs['observation'].shape)
        
        action, _ = model.predict(noisy_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get("is_success"):
            print(f"✅ Reached through the noise in {step+1} steps.")
            successes += 1
            break
    else:
        print("❌ Missed target.")
        
print(f"Noise Test Score: {successes}/{attempts} ({successes/attempts*100:.0f}%)")
time.sleep(1)

# ---------------------------------------------------------
# TEST 3: DYNAMIC TARGET TRACKING
# ---------------------------------------------------------
print("\n🛸 TEST 3: DYNAMIC EVASION TRACKING 🛸")
print("Target will move in a continuous circle. Robot must track it live.")

obs, info = env.reset()
tracking_errors = []

for t in range(300):
    # Calculate a moving target (circular path)
    angle = t * 0.05
    moving_target = np.array([0.1 * np.cos(angle), 0.1 * np.sin(angle), 0.15], dtype=np.float32)
    
    # Update simulator visually and logically
    env.unwrapped.task.goal = moving_target
    env.unwrapped.task.sim.set_base_pose("target", moving_target, np.array([0.0, 0.0, 0.0, 1.0]))
    obs['desired_goal'] = moving_target
    
    # Predict and execute
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Record the distance error while tracking
    dist = np.linalg.norm(obs['achieved_goal'] - moving_target)
    tracking_errors.append(dist)
    
    if t % 50 == 0:
        print(f"  -> Tracking step {t:3d}... current error: {dist:.4f}m")
        
    time.sleep(0.02)

mean_tracking_error = np.mean(tracking_errors)
print(f"\n🎯 Mean Tracking Error: {mean_tracking_error:.4f}m")
print(f"If this is under 0.05m, your agent is an expert tracker!")

env.close()
print("\nAggressive Testing Complete.")