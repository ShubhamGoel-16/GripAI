import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import numpy as np

# 1. Load the Muscles (RL Agent)
env = gym.make("PandaReach-v3", render_mode="human")
model = SAC.load("panda_reach_agent", env=env)

def get_coordinates_from_command(command):
    """
    In a full project, this would be an LLM API call.
    For now, let's simulate the LLM's 'Spatial Reasoning'.
    """
    command = command.lower()
    # Basic spatial mapping logic
    x, y, z = 0.0, 0.0, 0.1 # Default center
    
    if "left" in command: y -= 0.15
    if "right" in command: y += 0.15
    if "front" in command: x += 0.15
    if "back" in command: x -= 0.15
    if "high" in command: z += 0.1
    
    return np.array([x, y, z], dtype=np.float32)

# 2. Interactive Loop
try:
    while True:
        user_input = input("Where should the robot move? (e.g., 'front left', 'back right high'): ")
        if user_input.lower() in ['exit', 'quit']: break
        
        # LLM 'Interprets' the goal
        target_goal = get_coordinates_from_command(user_input)
        print(f"LLM set goal to: {target_goal}")

        obs, info = env.reset()
        # Manually override the environment's random goal with our LLM goal
        obs['desired_goal'] = target_goal
        env.unwrapped.task.goal = target_goal 

        # Let the RL agent try to reach the LLM's goal
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # Re-insert our manual goal into the observation
            obs['desired_goal'] = target_goal
            
            if terminated or truncated:
                print("Goal reached successfully!")
                break
                
except KeyboardInterrupt:
    pass
finally:
    env.close()