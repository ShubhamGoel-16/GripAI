import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import numpy as np
import time

env = gym.make("PandaReach-v3", render_mode="human")
model = SAC.load("panda_expert_muscles", env=env)

# Define the 4 corners of the table at a specific height
extreme_goals = [
    [ 0.15,  0.15, 0.05], # Front Left
    [ 0.15, -0.15, 0.05], # Front Right
    [-0.15,  0.15, 0.20], # Back Left High
    [-0.15, -0.15, 0.20]  # Back Right High
]

print("🚀 Starting Heavy Stress Test: Boundary Corners")

for i, target in enumerate(extreme_goals):
    obs, info = env.reset()
    
    # Force the environment to use our extreme goal
    target_node = np.array(target, dtype=np.float32)
    env.unwrapped.task.goal = target_node
    env.unwrapped.task.sim.set_base_pose("target", target_node, np.array([0,0,0,1]))
    obs['desired_goal'] = target_node
    
    print(f"Testing Corner {i+1}: {target}")
    
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        obs['desired_goal'] = target_node # Keep goal fixed
        
        if info.get("is_success"):
            print(f"✅ Reached corner in {step} steps.")
            time.sleep(0.5)
            break
    else:
        print(f"❌ FAILED to reach corner {target}")

env.close()