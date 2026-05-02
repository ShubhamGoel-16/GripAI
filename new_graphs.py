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
    "legend.fontsize":   9,
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

# ── Log paths — exactly matching your training scripts ─────────────────────────
LOG_DIRS = {
    "reach":        "./logs/",
    "push":         "./logs/push/",
    "pickandplace": "./logs/pickandplace/",
}

# ── Helper functions ───────────────────────────────────────────────────────────
def smooth(values, weight=0.88):
    if len(values) == 0: return np.array([])
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)

def read_tb_accumulator(log_dir, tag):
    """Read TensorBoard logs using EventAccumulator."""
    steps, values = [], []
    if not os.path.exists(log_dir):
        return [], []
    for root, _, files in os.walk(log_dir):
        for fname in files:
            if "events.out" not in fname:
                continue
            path = os.path.join(root, fname)
            try:
                ea = EventAccumulator(path)
                ea.Reload()
                if tag in ea.Tags().get("scalars", []):
                    for e in ea.Scalars(tag):
                        steps.append(e.step)
                        values.append(e.value)
            except Exception:
                pass
    if steps:
        pairs = sorted(zip(steps, values))
        steps, values = zip(*pairs)
    return list(steps), list(values)

def read_tb_iterator(log_dir, tag):
    """Read TensorBoard logs using summary_iterator."""
    steps, values = [], []
    if not os.path.exists(log_dir):
        return [], []
    for root, _, files in os.walk(log_dir):
        for fname in files:
            if "events.out" not in fname:
                continue
            path = os.path.join(root, fname)
            try:
                for event in summary_iterator(path):
                    for v in event.summary.value:
                        if v.tag == tag:
                            steps.append(event.step)
                            values.append(v.simple_value)
            except Exception:
                pass
    if steps:
        pairs = sorted(zip(steps, values))
        steps, values = zip(*pairs)
    return list(steps), list(values)

def get_data(log_dir, tag):
    """Try TensorBoard first, fall back gracefully."""
    if TB_AVAILABLE == "accumulator":
        return read_tb_accumulator(log_dir, tag)
    elif TB_AVAILABLE:
        return read_tb_iterator(log_dir, tag)
    return [], []

# ── Load all data strictly from logs ───────────────────────────────────────────
data = {}
for name, log_dir in LOG_DIRS.items():
    # SB3 EvalCallback logs to eval/success_rate
    # SB3 rollout logs to rollout/ep_rew_mean
    sr_steps,  sr_vals  = get_data(log_dir, "eval/success_rate")
    rew_steps, rew_vals = get_data(log_dir, "rollout/ep_rew_mean")

    if not sr_steps:
        print(f"⚠️  No success rate logs found for {name} in {log_dir}")
    if not rew_steps:
        print(f"⚠️  No reward logs found for {name} in {log_dir}")

    data[name] = {
        "sr_steps":  np.array(sr_steps),
        "sr_vals":   np.array(sr_vals),
        "rew_steps": np.array(rew_steps),
        "rew_vals":  np.array(rew_vals),
    }

# ── Build Figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    "GripAI — Hierarchical Robotic Control via RL\nTraining Results",
    fontsize=20, fontweight="bold", color="#ffffff", y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

# ── Plot 1: Success Rate comparison (all tasks) ────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_title("Success Rate During Training", fontsize=13, pad=10, color="#ffffff")
ax1.set_xlabel("Training Timesteps")
ax1.set_ylabel("Success Rate")
ax1.set_ylim(-0.05, 1.1)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
ax1.grid(True)

for name, d in data.items():
    if len(d["sr_steps"]) > 1:
        s = smooth(d["sr_vals"])
        ax1.fill_between(d["sr_steps"], d["sr_vals"],
                         color=COLORS[name], alpha=0.08)
        ax1.plot(d["sr_steps"], d["sr_vals"],
                 color=COLORS[name], alpha=0.2, linewidth=1)
        ax1.plot(d["sr_steps"], s,
                 color=COLORS[name], linewidth=2.5, label=LABELS[name])

ax1.axhline(y=0.7, color="#ffffff", linestyle=":", alpha=0.3, linewidth=1)
ax1.text(ax1.get_xlim()[1] * 0.01, 0.72, "Target 70%",
         color="#ffffff", alpha=0.4, fontsize=9)
ax1.legend(loc="lower right")

# ── Plot 2: Final Success Rate bar chart ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title("Final Success Rates", fontsize=13, pad=10, color="#ffffff")
ax2.set_ylabel("Success Rate")
ax2.set_ylim(0, 1.15)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax2.grid(True, axis="y")

# Extract final success rates safely (defaults to 0 if logs are missing)
final_sr = {}
for name, d in data.items():
    if len(d["sr_vals"]) > 0:
        final_sr[name] = float(d["sr_vals"][-1])
    else:
        final_sr[name] = 0.0 

bar_labels = ["Reach", "Push", "Pick &\nPlace"]
bar_vals   = [final_sr["reach"], final_sr["push"], final_sr["pickandplace"]]
bar_colors = [COLORS["reach"],   COLORS["push"],   COLORS["pickandplace"]]

bars = ax2.bar(bar_labels, bar_vals, color=bar_colors,
               width=0.5, edgecolor="#0d1117", linewidth=1.5)
ax2.set_xticks(range(len(bar_labels)))
ax2.set_xticklabels(bar_labels, fontsize=10)

for bar, val in zip(bars, bar_vals):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.03,
        f"{val*100:.0f}%" if val > 0 else "N/A",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold", color="#ffffff"
    )

# ── Plots 3-5: Individual reward curves ───────────────────────────────────────
task_order  = ["reach", "push", "pickandplace"]
task_titles = ["Reach — Episode Reward",
               "Push — Episode Reward",
               "PickAndPlace — Episode Reward"]

for i, (name, title) in enumerate(zip(task_order, task_titles)):
    ax = fig.add_subplot(gs[1, i])
    ax.set_title(title, fontsize=10, pad=8, color="#ffffff")
    ax.set_xlabel("Timesteps", fontsize=9)
    ax.set_ylabel("Mean Reward", fontsize=9)
    ax.grid(True)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
    )

    d = data[name]
    if len(d["rew_steps"]) > 1:
        s = smooth(d["rew_vals"], weight=0.92)
        ax.fill_between(d["rew_steps"], d["rew_vals"],
                        alpha=0.12, color=COLORS[name])
        ax.plot(d["rew_steps"], d["rew_vals"],
                color=COLORS[name], alpha=0.2, linewidth=0.8)
        ax.plot(d["rew_steps"], s,
                color=COLORS[name], linewidth=2.2)

        # Annotate final value
        ax.annotate(
            f"  final: {d['rew_vals'][-1]:.1f}",
            xy=(d["rew_steps"][-1], s[-1]),
            color=COLORS[name], fontsize=8
        )
    else:
        ax.text(0.5, 0.5, "No Reward Data", ha='center', va='center', transform=ax.transAxes, color='#8b949e')

# ── Save & Show ────────────────────────────────────────────────────────────────
os.makedirs("./results1/", exist_ok=True)
out_path = "./results1/training_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\n✅ Saved to {out_path}")
plt.show()