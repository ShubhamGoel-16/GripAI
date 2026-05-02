import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Try to read TensorBoard logs ───────────────────────────────────────────────
try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
    TB_AVAILABLE = True
except ImportError:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        TB_AVAILABLE = "accumulator"
    except ImportError:
        TB_AVAILABLE = False
        print("❌ TensorBoard not found. Run: pip install tensorflow OR tensorboard")
        exit()

# ── Output Directory setup ─────────────────────────────────────────────────────
OUT_DIR = "./results7/"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "legend.fontsize":   11,
})

COLORS = {
    "reach":        "#3fb950",   # Green
    "push":         "#58a6ff",   # Blue
    "pickandplace": "#f78166",   # Orange-red
}

LABELS = {
    "reach":        "PandaReach (Dense)",
    "push":         "PandaPush",
    "pickandplace": "PandaPickAndPlace",
}

# ── DIRECT TARGET DIRECTORIES ──────────────────────────────────────────────────
LOG_DIRS = {
    "reach":        "./logs/SAC_2/",
    "push":         "./logs/push/SAC_2/",
    "pickandplace": "./logs/pickandplace/SAC_6/"
}

# ── Helper functions ───────────────────────────────────────────────────────────
def smooth(values, weight=0.88):
    if len(values) == 0: return np.array([])
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)

def format_x(x, _):
    return f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"

# ── THIS IS THE ONLY CHANGED FUNCTION ──────────────────────────────────────────
def get_event_file(log_dir):
    """Finds the LARGEST events file in the directory (the longest training run)."""
    if not os.path.exists(log_dir): 
        return None
        
    largest_file = None
    max_size = 0
    
    for fname in os.listdir(log_dir):
        path = os.path.join(log_dir, fname)
        if os.path.isfile(path) and "events.out" in fname:
            file_size = os.path.getsize(path)
            if file_size > max_size:
                max_size = file_size
                largest_file = path
                
    return largest_file
# ───────────────────────────────────────────────────────────────────────────────

def get_data(log_dir, tags):
    """Checks multiple possible tags (e.g., eval vs rollout) and returns the first match."""
    target_file = get_event_file(log_dir)
    if not target_file: return [], []
    
    if isinstance(tags, str):
        tags = [tags]
        
    for tag in tags:
        steps, values = [], []
        try:
            if TB_AVAILABLE == "accumulator":
                ea = EventAccumulator(target_file)
                ea.Reload()
                if tag in ea.Tags().get("scalars", []):
                    for e in ea.Scalars(tag):
                        steps.append(e.step)
                        values.append(e.value)
            else:
                for event in summary_iterator(target_file):
                    for v in event.summary.value:
                        if v.tag == tag:
                            steps.append(event.step)
                            values.append(v.simple_value)
        except Exception: pass
        
        if steps:
            pairs = sorted(zip(steps, values))
            steps, values = zip(*pairs)
            return list(steps), list(values)
            
    return [], []

# ── Extract All Data ───────────────────────────────────────────────────────────
data = {}
for name, log_dir in LOG_DIRS.items():
    # Pass a list of tags so it checks eval first, then falls back to rollout
    sr_s, sr_v = get_data(log_dir, ["eval/success_rate", "rollout/success_rate"])
    rew_s, rew_v = get_data(log_dir, "rollout/ep_rew_mean")
    len_s, len_v = get_data(log_dir, "rollout/ep_len_mean") 

    if not sr_s: print(f"⚠️  No success rate data found in {log_dir}")
    if not rew_s: print(f"⚠️  No reward data found in {log_dir}")

    data[name] = {
        "sr_steps": np.array(sr_s), "sr_vals": np.array(sr_v),
        "rew_steps": np.array(rew_s), "rew_vals": np.array(rew_v),
        "len_steps": np.array(len_s), "len_vals": np.array(len_v),
    }

print(f"📊 Extracting data and saving to {OUT_DIR}...")

# ===============================================================================
# GRAPH 1: OVERLAID SUCCESS RATE (The "Curriculum" Slide)
# ===============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title("Task Complexity vs. Convergence Time\n(Success Rate)", fontsize=14, color="#ffffff", pad=15)
ax1.set_xlabel("Training Timesteps", fontsize=12)
ax1.set_ylabel("Success Rate", fontsize=12)
ax1.set_ylim(-0.05, 1.1)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
ax1.grid(True)

for name, d in data.items():
    if len(d["sr_steps"]) > 1:
        s = smooth(d["sr_vals"], 0.8)
        ax1.fill_between(d["sr_steps"], d["sr_vals"], color=COLORS[name], alpha=0.1)
        ax1.plot(d["sr_steps"], s, color=COLORS[name], linewidth=3, label=LABELS[name])

ax1.axhline(y=0.9, color="#ffffff", linestyle=":", alpha=0.4, linewidth=1.5)
ax1.legend(loc="lower right")
fig1.savefig(f"{OUT_DIR}1_Success_Rate_Comparison.png", dpi=200, bbox_inches="tight", facecolor=fig1.get_facecolor())

# ===============================================================================
# GRAPH 2: EPISODE REWARD (The "Optimization" Slide)
# ===============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Policy Optimization (Mean Episode Reward)", fontsize=16, color="#ffffff", y=1.05)
task_order = ["reach", "push", "pickandplace"]

for ax, name in zip(axes2, task_order):
    d = data[name]
    ax.set_title(LABELS[name], fontsize=12, color=COLORS[name])
    ax.set_xlabel("Timesteps")
    ax.grid(True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    
    if len(d["rew_steps"]) > 1:
        s = smooth(d["rew_vals"], 0.9)
        ax.fill_between(d["rew_steps"], d["rew_vals"], color=COLORS[name], alpha=0.15)
        ax.plot(d["rew_steps"], s, color=COLORS[name], linewidth=2.5)
        ax.annotate(f"Final: {d['rew_vals'][-1]:.1f}", xy=(d["rew_steps"][-1], s[-1]), color="#ffffff")

fig2.savefig(f"{OUT_DIR}2_Episode_Rewards.png", dpi=200, bbox_inches="tight", facecolor=fig2.get_facecolor())

# ===============================================================================
# GRAPH 3: EPISODE LENGTH (The "Efficiency" Slide)
# ===============================================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle("Kinematic Efficiency (Mean Episode Length)", fontsize=16, color="#ffffff", y=1.05)

for ax, name in zip(axes3, task_order):
    d = data[name]
    ax.set_title(LABELS[name], fontsize=12, color=COLORS[name])
    ax.set_xlabel("Timesteps")
    if name == "reach": ax.set_ylabel("Steps to Goal")
    ax.grid(True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    
    if len(d["len_steps"]) > 1:
        s = smooth(d["len_vals"], 0.9)
        ax.fill_between(d["len_steps"], d["len_vals"], color=COLORS[name], alpha=0.15)
        ax.plot(d["len_steps"], s, color=COLORS[name], linewidth=2.5)
        ax.annotate(f"{d['len_vals'][-1]:.1f} steps", xy=(d["len_steps"][-1], s[-1]), color="#ffffff")

fig3.savefig(f"{OUT_DIR}3_Episode_Lengths.png", dpi=200, bbox_inches="tight", facecolor=fig3.get_facecolor())

# ===============================================================================
# GRAPH 4: THE FINAL SCORECARD
# ===============================================================================
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.set_title("Final Convergence Benchmarks", fontsize=14, color="#ffffff", pad=15)
ax4.set_ylabel("Maximum Success Rate Achieved")
ax4.set_ylim(0, 1.15)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax4.grid(True, axis="y")

best_sr = {name: float(max(d["sr_vals"])) if len(d["sr_vals"]) > 0 else 0.0 for name, d in data.items()}

bars = ax4.bar(["Reach", "Push", "Pick & Place"], 
               [best_sr["reach"], best_sr["push"], best_sr["pickandplace"]], 
               color=[COLORS["reach"], COLORS["push"], COLORS["pickandplace"]], width=0.6)

for bar, val in zip(bars, [best_sr["reach"], best_sr["push"], best_sr["pickandplace"]]):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, 
             f"{val*100:.0f}%", ha="center", va="bottom", fontsize=14, fontweight="bold", color="#ffffff")

fig4.savefig(f"{OUT_DIR}4_Final_Scorecard.png", dpi=200, bbox_inches="tight", facecolor=fig4.get_facecolor())

print(f"✅ All 4 presentation graphs successfully pulled from your exact final logs and saved to {OUT_DIR}!")
plt.show()