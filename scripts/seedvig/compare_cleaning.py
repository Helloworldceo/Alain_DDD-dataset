"""Cleaning ablation: per-model comparison of cleaned vs filtered-only signals (PERCLOS,
leave-one-person-out). Wilcoxon signed-rank across the 21 people, Holm over models.
Output: results/seedvig/cleaning_ablation.json"""
import json, os
import numpy as np
from scipy.stats import wilcoxon
from analyze_lopo import holm, RES
c = json.load(open(os.path.join(RES, "lopo_reg_clean.json")))["scores"]
r = json.load(open(os.path.join(RES, "lopo_reg_raw.json")))["scores"]
persons = sorted({k.split("|")[1] for k in c}, key=int)
models = ["SVR", "MLP", "MLP-compact", "LSTM", "KAN", "CNN-LSTM", "CNN-LSTM-KAN"]
out = {}
for met in ("cor", "rmse"):
    rows, ps = [], []
    for m in models:
        a = np.array([c[f"{m}|{p}"][met] for p in persons]); b = np.array([r[f"{m}|{p}"][met] for p in persons])
        p = float(wilcoxon(a, b).pvalue)
        better = int(((a > b) if met == "cor" else (a < b)).sum())
        rows.append(dict(model=m, clean=float(a.mean()), filtered=float(b.mean()), diff=float((a - b).mean()),
                         p=p, clean_better_in=better)); ps.append(p)
    for row, pa in zip(rows, holm(ps)): row["p_holm"] = float(pa)
    out[met] = rows
    for row in rows:
        print(f"{met} {row['model']:13s} filtered {row['filtered']:.3f} -> clean {row['clean']:.3f} "
              f"({row['diff']:+.3f}) better in {row['clean_better_in']}/21, p_holm {row['p_holm']:.3f}")
json.dump(out, open(os.path.join(RES, "cleaning_ablation.json"), "w"), indent=1)
