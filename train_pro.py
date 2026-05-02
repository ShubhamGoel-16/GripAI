import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import os

# 1. Create the environment
# We use 'PandaReachDense-v3' so the robot learns by getting "warmer/colder" feedback
env = gym.make("PandaReachDense-v3", render_mode="human")

# 2. Initialize the Model
# We'll train for more steps this time to ensure it becomes an "Expert"
total_timesteps = 50000 

model = SAC(
    "MultiInputPolicy", 
    env, 
    verbose=1, 
    learning_rate=1e-3, # Slightly faster learning rate
    buffer_size=100000,
    batch_size=256,
    gamma=0.95,         # Focus on immediate success
    tau=0.05,           # Faster target network updates
    tensorboard_log="./logs/"
)

print(f"🚀 Starting Live Training for {total_timesteps} steps...")
print("The GUI window will open. Watch the 'ep_rew_mean' in the console.")

try:
    # 3. Train the agent
    # This will open the PyBullet window and show the arm moving
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    
    # 4. Save the expert brain
    model.save("panda_expert_muscles")
    print("✅ expert model saved as 'panda_expert_muscles'")

except KeyboardInterrupt:
    print("Training interrupted by user. Saving progress...")
    model.save("panda_partial_muscles")

finally:
    env.close()