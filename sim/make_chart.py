"""Comparison chart (Day 13): reads results.json from evaluate.py -> comparison.png"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
res = json.load(open(path))
names = [n for n in res if n != "random"]  # random shown for scale only if asked
names = list(res)
metrics = [("coverage", "Desk covered (fraction)"), ("debris", "Debris collected (fraction)"),
           ("pens", "Pens delivered (fraction)"), ("steps", "Steps used (fewer = faster)")]

colors = {"random": "#9a9a9a", "backup pattern": "#6b8fc4", "backup + edge guard": "#3f6aa8",
          "PPO model": "#e08a3c", "PPO + edge guard": "#c0642a"}
fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
for ax, (k, title) in zip(axes, metrics):
    vals = [res[n][k] for n in names]
    ax.bar(range(len(names)), vals, color=[colors.get(n, "#888") for n in names])
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" + ", "\n+ ").replace(" ", "\n", 1) if " " in n else n for n in names],
                       fontsize=7)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}" if k != "steps" else f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax = axes[4]
hits = [res[n]["edge_hit_episodes"] for n in names]
ax.bar(range(len(names)), hits, color=[colors.get(n, "#888") for n in names])
ax.set_title(f"Episodes with an edge hit\n(out of {res[names[0]]['episodes']})", fontsize=10)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace(" + ", "\n+ ").replace(" ", "\n", 1) if " " in n else n for n in names], fontsize=7)
for i, v in enumerate(hits):
    ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("comparison.png", dpi=160)
print("wrote comparison.png")
