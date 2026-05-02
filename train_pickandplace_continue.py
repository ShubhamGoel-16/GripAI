import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import os

ENV_ID = "PandaPickAndPlace-v3"
train_env = gym.make(ENV_ID)
eval_env  = gym.make(ENV_ID, render_mode="human")

os.makedirs("./logs/pickandplace_v3/",        exist_ok=True)
os.makedirs("./best_model/pickandplace_v3/",  exist_ok=True)
os.makedirs("./checkpoints/pickandplace_v3/", exist_ok=True)

print("Loading best model from ORIGINAL run (before catastrophic forgetting)...")

# ── Load from the ORIGINAL best model, NOT the v3 one ─────────────────────────
model = SAC.load(
    "./best_model/pickandplace/best_model",   # ← Original 55% model
    env=train_env,
    custom_objects={
        "replay_buffer_class": HerReplayBuffer,
        "replay_buffer_kwargs": dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        )
    }
)

# ── Set ALL params directly on model after loading ────────────────────────────
model.learning_starts  = 2000    # Wait before sampling
model.learning_rate    = 5e-4    # Fine-tuning rate
model.gradient_steps   = 1       # ← THE KEY FIX: 1 update per step, not 4
model.batch_size       = 512
print("✅ Model loaded and params set correctly!")
print(f"   learning_rate  : {model.learning_rate}")
print(f"   gradient_steps : {model.gradient_steps}")
print(f"   learning_starts: {model.learning_starts}")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/pickandplace_v3/",
    log_path="./logs/pickandplace_v3/",
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/pickandplace_v3/",
    name_prefix="panda_pickandplace_v3"
)

additional_steps = 500_000

print(f"\n🚀 Fine-tuning for {additional_steps:,} steps...")
print("🎯 Success rate should stay above 55% immediately and climb to 70%+\n")

try:
    model.learn(
        total_timesteps=additional_steps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=20,
        progress_bar=True,
        reset_num_timesteps=True
    )

    model.save("panda_pickandplace_expert_v3")
    print("\n✅ Done! Saved as 'panda_pickandplace_expert_v3'")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted. Saving...")
    model.save("panda_pickandplace_partial_v3")

finally:
    train_env.close()
    eval_env.close()
    print("Environments closed.")