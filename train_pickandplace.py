import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import os

# ── 1. Create Environments ─────────────────────────────────────────────────────
ENV_ID = "PandaPickAndPlace-v3"

train_env = gym.make(ENV_ID)                        # No render for speed
eval_env  = gym.make(ENV_ID, render_mode="human")   # Render only during eval

# ── 2. Create Directories ──────────────────────────────────────────────────────
os.makedirs("./logs/pickandplace/",        exist_ok=True)
os.makedirs("./best_model/pickandplace/",  exist_ok=True)
os.makedirs("./checkpoints/pickandplace/", exist_ok=True)

# ── 3. Initialize SAC + HER ────────────────────────────────────────────────────
# PickAndPlace is significantly harder than Push:
# The arm must LIFT the object (not just slide it), requiring
# more exploration and a larger buffer to store diverse experiences
model = SAC(
    "MultiInputPolicy",
    train_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,                   # Relabel 4 extra goals per transition
        goal_selection_strategy="future",   # Best strategy for manipulation tasks
    ),
    verbose=1,
    learning_rate=1e-3,
    buffer_size=1_000_000,                  # Large buffer essential for HER
    batch_size=512,                         
    gamma=0.95,
    tau=0.05,
    learning_starts=5000,                   # Collect more experience before training
                                            # (more than Push since task is harder)
    tensorboard_log="./logs/pickandplace/"
)

# ── 4. Callbacks ───────────────────────────────────────────────────────────────
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/pickandplace/",
    log_path="./logs/pickandplace/",
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/pickandplace/",
    name_prefix="panda_pickandplace"
)

# ── 5. Train ───────────────────────────────────────────────────────────────────
total_timesteps = 2_000_000     # 5x more than Push — this task needs patience

print(f"🚀 Starting PandaPickAndPlace Training for {total_timesteps:,} steps...")
print("⚠️  Success rate will stay at 0 for the first ~200k steps — this is NORMAL.")
print("🎯 Target: success_rate > 0.7 before stopping.\n")

try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=20,
        progress_bar=True
    )

    model.save("panda_pickandplace_expert")
    print("\n✅ Training complete! Model saved as 'panda_pickandplace_expert'")

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted. Saving partial model...")
    model.save("panda_pickandplace_partial")
    print("💾 Saved as 'panda_pickandplace_partial'")

finally:
    train_env.close()
    eval_env.close()
    print("Environments closed.")