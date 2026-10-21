"""Figure: SEED-VIG EEG montage. The 60 scalp positions (cerebellar CB1/CB2 omitted) of the electrode-position file distributed
with the dataset (data/SEED-VIG/channel_62_pos.locs, polar EEGLAB format), with the 17 recorded
channels marked by region (temporal, posterior) as grouped in the compact features.
Output: figures/seedvig/fig_seedvig_montage.{pdf,png}"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LOCS = os.path.join(ROOT, "data", "SEED-VIG", "channel_62_pos.locs")
FIG = os.path.join(ROOT, "figures", "seedvig")
TEMPORAL = ["FT7", "FT8", "T7", "T8", "TP7", "TP8"]
POSTERIOR = ["CP1", "CP2", "P1", "PZ", "P2", "PO3", "POZ", "PO4", "O1", "OZ", "O2"]

rows = [l.split() for l in open(LOCS) if l.strip()]
names = [r[3] for r in rows]
theta = np.deg2rad([float(r[1]) for r in rows]); rad = np.array([float(r[2]) for r in rows])
x, y = rad * np.sin(theta), rad * np.cos(theta)
R = rad.max() * 1.13

plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(4.6, 4.8))
t = np.linspace(0, 2 * np.pi, 300)
ax.plot(R * np.cos(t), R * np.sin(t), color="#444", lw=1.2)
ax.plot([-0.08 * R, 0, 0.08 * R], [R * 0.995, R * 1.09, R * 0.995], color="#444", lw=1.2)          # nose
for s in (-1, 1):
    ax.add_patch(matplotlib.patches.Ellipse((s * R * 1.035, 0), 0.07 * R, 0.25 * R, fill=False, color="#444", lw=1.2))
groups = [(TEMPORAL, "#eb6834", "Temporal (6)"), (POSTERIOR, "#2a78d6", "Posterior (11)")]
used = {n.upper() for g, _, _ in groups for n in g}
for xi, yi, n in zip(x, y, names):
    if n.upper() not in used and not n.startswith("CB"):
        ax.scatter(xi, yi, s=14, color="#d0d0d0", zorder=2)
        ax.text(xi, yi - 0.032, n, ha="center", va="top", fontsize=4.6, color="#9a9a9a")
for g, col, lab in groups:
    idx = [i for i, n in enumerate(names) if n.upper() in g]
    ax.scatter(x[idx], y[idx], s=95, color=col, edgecolor="white", lw=0.8, zorder=3, label=lab)
    for i in idx:
        ax.text(x[i], y[i] - 0.04, names[i].replace("PZ", "Pz").replace("POZ", "POz").replace("OZ", "Oz"),
                ha="center", va="top", fontsize=6.2, fontweight="bold", color="#222", zorder=4)
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-R * 1.15, R * 1.15); ax.set_ylim(-R * 1.12, R * 1.15)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, fontsize=7.5, markerscale=0.8)
os.makedirs(FIG, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_seedvig_montage.{ext}"), dpi=220, bbox_inches="tight")
print("saved")
