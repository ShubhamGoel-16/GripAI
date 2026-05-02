import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import numpy as np
import time

# 1. Initialization
env = gym.make("PandaPickAndPlace-v3", render_mode="human")
model_path = "panda_pickandplace_expert.zip"

try:
    print(f"Loading expert model from {model_path}...")
    model = SAC.load(model_path, env=env)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ---------------------------------------------------------
# TEST 1: MAXIMUM ALTITUDE & CORNERS (The Gravity Test)
# ---------------------------------------------------------
print("🔥 TEST 1: EXTREME PAYLOAD DELIVERY 🔥")
print("Testing the absolute corners and maximum height of the 3D workspace...")

# Define extreme corners, specifically forcing high Z-axis lifts (0.25m)
bounds = [-0.15, 0.15]
corners = [[x, y, 0.25] for x in bounds for y in bounds]

for i, target in enumerate(corners):
    obs, info = env.reset()
    target_pos = np.array(target, dtype=np.float32)
    
    # Force the target to the extreme corner
    env.unwrapped.task.goal = target_pos
    env.unwrapped.task.sim.set_base_pose("target", target_pos, np.array([0.0, 0.0, 0.0, 1.0]))
    obs['desired_goal'] = target_pos
    
    print(f"  -> Corner {i+1}/4 {target}: ", end="")
    
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        obs['desired_goal'] = target_pos # Lock goal
        
        if info.get("is_success"):
            print(f"✅ Lifted and placed in {step+1} steps.")
            break
    else:
        print("❌ FAILED (Dropped or missed).")
    time.sleep(0.5)

# ---------------------------------------------------------
# TEST 2: SENSOR NOISE INJECTION (The Grip Stability Test)
# ---------------------------------------------------------
print("\n🌪️ TEST 2: SENSOR NOISE INJECTION 🌪️")
print("Adding Gaussian noise to spatial awareness. Will it drop the block?")

noise_std_dev = 0.015 # 1.5cm of noise (Very high for gripping!)
successes = 0
attempts = 5

for i in range(attempts):
    obs, info = env.reset()
    print(f"  -> Noisy Attempt {i+1}/{attempts}: ", end="")
    
    for step in range(100):
        # Inject noise into the observation before prediction
        noisy_obs = obs.copy()
        noisy_obs['observation'] = obs['observation'] + np.random.normal(0, noise_std_dev, obs['observation'].shape)
        
        action, _ = model.predict(noisy_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get("is_success"):
            print(f"✅ Held grip through the noise! ({step+1} steps)")
            successes += 1
            break
    else:
        print("❌ Dropped block or missed target.")
        
print(f"Noise Test Score: {successes}/{attempts} ({successes/attempts*100:.0f}%)")
time.sleep(1)

# ---------------------------------------------------------
# TEST 3: THE "MOVING BASKET" (Dynamic Tracking)
# ---------------------------------------------------------
print("\n🛸 TEST 3: THE MOVING BASKET 🛸")
print("Target will orbit the workspace. Robot must carry the block and track it live.")

obs, info = env.reset()
tracking_errors = []

for t in range(400):
    # Calculate a moving target (circular orbit, high in the air)
    angle = t * 0.03
    moving_target = np.array([0.1 * np.cos(angle), 0.1 * np.sin(angle), 0.20], dtype=np.float32)
    
    # Update simulator visually and logically
    env.unwrapped.task.goal = moving_target
    env.unwrapped.task.sim.set_base_pose("target", moving_target, np.array([0.0, 0.0, 0.0, 1.0]))
    obs['desired_goal'] = moving_target
    
    # Predict and execute
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Record the distance between the BLOCK (not the gripper) and the moving target
    block_pos = obs['achieved_goal'] 
    dist = np.linalg.norm(block_pos - moving_target)
    tracking_errors.append(dist)
    
    if t % 50 == 0:
        print(f"  -> Carrying block... distance to moving target: {dist:.4f}m")
        
    time.sleep(0.02)

mean_tracking_error = np.mean(tracking_errors)
print(f"\n🎯 Mean Payload Tracking Error: {mean_tracking_error:.4f}m")
if mean_tracking_error < 0.05:
    print("🔥 Your model is an absolute beast. It successfully chased a moving target while fighting gravity!")

env.close()
print("\nAggressive Testing Complete.")