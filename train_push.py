import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import os

# ── 1. Create Environments ─────────────────────────────────────────────────────
ENV_ID = "PandaPush-v3"

train_env = gym.make(ENV_ID)                          # No render for speed
eval_env  = gym.make(ENV_ID, render_mode="human")     # Render only during eval

# ── 2. Create Directories ──────────────────────────────────────────────────────
os.makedirs("./logs/push/",         exist_ok=True)
os.makedirs("./best_model/push/",   exist_ok=True)
os.makedirs("./checkpoints/push/",  exist_ok=True)

# ── 3. Initialize SAC + HER ────────────────────────────────────────────────────
# HER is critical here — PandaPush has sparse rewards so without HER
# the agent almost never discovers success by random exploration alone
model = SAC(
    "MultiInputPolicy",
    train_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,                  # Relabel 4 extra goals per transition
        goal_selection_strategy="future",  # Best strategy for manipulation tasks
    ),
    verbose=1,
    learning_rate=1e-3,
    buffer_size=1_000_000,                 # Large buffer needed for HER
    batch_size=512,
    gamma=0.95,
    tau=0.05,
    learning_starts=1000,                  # Collect some experience before training
    tensorboard_log="./logs/push/"
)

# ── 4. Callbacks ───────────────────────────────────────────────────────────────
# Saves the best model whenever eval improves
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/push/",
    log_path="./logs/push/",
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1
)

# Saves a checkpoint every 50k steps so you never lose progress
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/push/",
    name_prefix="panda_push"
)

# ── 5. Train ───────────────────────────────────────────────────────────────────
total_timesteps = 200_000   # Push needs more steps than Reach

print(f"🚀 Starting PandaPush Training for {total_timesteps:,} steps...")
print("Watch 'success_rate' in the console — aim for > 0.8 before stopping.\n")

try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=20,
        progress_bar=True           # pip install rich  →  gives a clean progress bar
    )

    model.save("panda_push_expert")
    print("\n✅ Training complete! Model saved as 'panda_push_expert'")

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted. Saving partial model...")
    model.save("panda_push_partial")
    print("💾 Saved as 'panda_push_partial'")

finally:
    train_env.close()
    eval_env.close()
    print("Environments closed.")