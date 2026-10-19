"""Modality ablation for the KAN (PERCLOS, leave-one-person-out, 3 seeds): EEG part of the
compact set (13 features), EOG part (6 features), and all 19 features (from the main run).
Output: results/seedvig/modality_reg.json"""
import json, os
import numpy as np
from train_lib import RES, Data, person_metrics, predict, split_fold, train_one
hp = json.load(open(os.path.join(RES, "pilot_reg.json")))["selection"]["KAN"]["selected"]
SUBSETS = {"KAN, EEG compact (13)": list(range(13)), "KAN, EOG compact (6)": list(range(13, 19))}
out_path = os.path.join(RES, "modality_reg.json")
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
d = Data("clean", load_raw=False); persons = sorted(set(d.person), key=int)
for name, cols in SUBSETS.items():
    h = dict(hp, cols=cols)
    for fid, p in enumerate(persons):
        key = f"{name}|{p}"
        if key in res: continue
        tr, va, te, _ = split_fold(d, p, fid)
        preds = [predict(*train_one(d, "compact", "reg", h, tr, va, seed=s)[:2], te) for s in range(3)]
        res[key] = person_metrics("reg", d.perclos[te], np.mean(preds, 0))
        json.dump(res, open(out_path, "w"), indent=1)
    rows = [res[f"{name}|{p}"] for p in persons]
    print(name, "COR %.3f RMSE %.3f" % (np.nanmean([r["cor"] for r in rows]), np.mean([r["rmse"] for r in rows])), flush=True)
