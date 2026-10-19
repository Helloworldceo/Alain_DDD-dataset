"""
Statistics and figure for the leave-one-person-out results.

For each task, the per-person scores of all models are summarized (mean, SD over
21 people) and every pair of models is compared with the two-sided Wilcoxon
signed-rank test across people; p-values are Holm-corrected over all pairs.
Effect size: matched-pairs rank-biserial correlation r_rb.

Run:     python3 analyze_lopo.py [signals=clean]
Output:  results/seedvig/stats_<task>_<signals>.json
         figures/seedvig/fig_lopo_per_person.{pdf,png}
"""
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, rankdata

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RES = os.path.join(ROOT, "results", "seedvig")
FIG = os.path.join(ROOT, "figures", "seedvig")
SIGNALS = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("signals=")), "clean")
ORDER = ["SVR", "MLP", "MLP-compact", "LSTM", "KAN", "CNN-LSTM", "CNN-LSTM-KAN"]
LABEL = {"SVR": "SVR/SVM", "MLP": "MLP", "MLP-compact": "MLP\n(compact)", "LSTM": "LSTM", "KAN": "KAN", "CNN-LSTM": "CNN-LSTM", "CNN-LSTM-KAN": "CNN-LSTM\n+KAN head"}
METRICS = {"reg": ["cor", "rmse"], "cls": ["acc", "f1"]}
HIGHER = {"cor": True, "rmse": False, "acc": True, "f1": True}


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def analyze(task):
    path = os.path.join(RES, f"lopo_{task}_{SIGNALS}.json")
    if not os.path.exists(path):
        return None
    sc = json.load(open(path))["scores"]
    persons = sorted({k.split("|")[1] for k in sc}, key=int)
    models = [m for m in ORDER if all(f"{m}|{p}" in sc for p in persons)]
    out = {"persons": persons, "models": models, "summary": {}, "pairs": {}}
    for met in METRICS[task]:
        vals = {m: np.array([sc[f"{m}|{p}"][met] for p in persons]) for m in models}
        out["summary"][met] = {m: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)),
                                       median=float(np.nanmedian(v))) for m, v in vals.items()}
        pairs = list(itertools.combinations(models, 2))
        ps, rows = [], []
        for a, b in pairs:
            ok = ~np.isnan(vals[a]) & ~np.isnan(vals[b])
            d = vals[a][ok] - vals[b][ok]
            p = float(wilcoxon(vals[a][ok], vals[b][ok]).pvalue) if np.any(d != 0) else 1.0
            better = (d > 0).sum() if HIGHER[met] else (d < 0).sum()
            rows.append(dict(a=a, b=b, mean_diff=float(d.mean()), p=p, r_rb=rank_biserial(d),
                             a_better_in=int(better), n=int(ok.sum())))
            ps.append(p)
        for r, pa in zip(rows, holm(ps)):
            r["p_holm"] = float(pa)
        out["pairs"][met] = rows
        out[f"values_{met}"] = {m: vals[m].tolist() for m in models}
    json.dump(out, open(os.path.join(RES, f"stats_{task}_{SIGNALS}.json"), "w"), indent=1)
    for met in METRICS[task]:
        print(task, met, {m: f"{v['mean']:.3f}±{v['sd']:.3f}" for m, v in out["summary"][met].items()})
    return out


def figure(stats):
    os.makedirs(FIG, exist_ok=True)
    panels = [(t, m) for t in ("reg", "cls") if stats.get(t) for m in METRICS[t]]
    title = {"cor": "PERCLOS: correlation (COR)", "rmse": "PERCLOS: RMSE", "acc": "3 classes: accuracy",
             "f1": "3 classes: macro-F1"}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4), squeeze=False)
    for ax, (t, met) in zip(axes[0], panels):
        s = stats[t]
        data = [np.array(s[f"values_{met}"][m]) for m in s["models"]]
        bp = ax.boxplot([v[~np.isnan(v)] for v in data], positions=range(len(data)), widths=0.55,
                        showfliers=False, patch_artist=True, medianprops=dict(color="#555", lw=1),
                        boxprops=dict(facecolor="#eef3fb", edgecolor="#9aa0a6", lw=0.8),
                        whiskerprops=dict(color="#9aa0a6", lw=0.8), capprops=dict(color="#9aa0a6", lw=0.8))
        for k, m in enumerate(s["models"]):
            v = np.array(s[f"values_{met}"][m])
            x = k + (np.random.default_rng(k).random(len(v)) - 0.5) * 0.35
            ax.scatter(x, v, s=10, color="#9aa0a6", alpha=0.8, zorder=2)
            ax.plot([k - 0.3, k + 0.3], [np.nanmean(v)] * 2, color="#2a78d6", lw=2.2, zorder=3)
        ax.set_xticks(range(len(s["models"])), [LABEL[m] for m in s["models"]], rotation=45, ha="right", fontsize=7.5)
        ax.set_title(title[met], loc="left", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_lopo_per_person_{SIGNALS}.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    stats = {t: analyze(t) for t in ("reg", "cls")}
    figure(stats)
