"""Figure: effect of artifact cleaning on SEED-VIG (example window + power spectra).
Output: figures/seedvig/fig_seedvig_cleaning.{pdf,png}"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
OUT = "../../results/seedvig"; FIG = "../../figures/seedvig"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
R = np.load(f"{OUT}/raw_windows.npy", mmap_mode="r"); C = np.load(f"{OUT}/clean_windows.npy", mmap_mode="r")
m = np.load(f"{OUT}/clean_meta.npz"); ch = list(m["channels"]); s = m["session"]
rep = json.load(open(f"{OUT}/cleaning_report.json"))
sess = "17_20150925_noon"; idx = np.where(s == sess)[0]
# window where cleaning changed the temporal channels most
d = np.abs(np.asarray(R[idx, :6]) - np.asarray(C[idx, :6])).mean(axis=(1, 2)); w = idx[int(np.argmax(d))]
t = np.arange(800) / 100
fig = plt.figure(figsize=(10, 5.6)); gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], hspace=0.62, wspace=0.28)
for k, c in enumerate(["FT7", "T8"]):
    ax = fig.add_subplot(gs[k, 0]); j = ch.index(c)
    ax.plot(t, R[w, j], color="#9aa0a6", lw=0.8, label="Filtered only")
    ax.plot(t, C[w, j], color="#2a78d6", lw=0.8, label="Cleaned")
    ax.set_ylabel(f"{c} (µV)"); ax.set_xlim(0, 8)
    if k == 0: ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8); ax.set_title(f"(a) Example 8-s window, session {sess.split('_')[0]}", loc="left")
ax = fig.add_subplot(gs[2, 0])
ax.plot(t, R[w, 18], color="#eb6834", lw=0.8, label="EOG-H"); ax.plot(t, R[w, 17], color="#4a3aa7", lw=0.8, label="EOG-V")
ax.set_ylabel("EOG (µV)"); ax.set_xlabel("Time (s)"); ax.set_xlim(0, 8); ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
ax = fig.add_subplot(gs[0:2, 1])
for X, col, lab in [(R, "#9aa0a6", "Filtered only"), (C, "#2a78d6", "Cleaned")]:
    f, P = welch(np.asarray(X[::10, :6], dtype=float), fs=100, nperseg=200, axis=-1)
    ax.semilogy(f, P.mean(axis=(0, 1)), color=col, lw=1.4, label=lab)
ax.set_xlim(1, 45); ax.set_ylim(0.2, 40); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (µV²/Hz)")
ax.set_title("(b) Temporal channels, all sessions", loc="left"); ax.legend(frameon=False, fontsize=8)
ax = fig.add_subplot(gs[2, 1]); sm = rep["summary"]
vals = [sm["bad_channels_total"], sm["ica_components_removed_total"], sm["windows_flagged"]]
bars = ax.barh(["Interpolated\nchannels", "Removed ocular\nICA components", "Flagged\nwindows"], vals, color="#2a78d6")
for b, v in zip(bars, vals): ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v}", va="center", fontsize=8)
ax.set_title("(c) Cleaning summary (23 sessions)", loc="left", pad=8); ax.set_xlim(0, max(vals) * 1.25)
for ext in ("pdf", "png"): fig.savefig(f"{FIG}/fig_seedvig_cleaning.{ext}", dpi=200, bbox_inches="tight")
print("saved", w, sess)
