"""
Hyperparameter sensitivity (PERCLOS, leave-one-person-out): the runner-up configurations
of the pilot are evaluated on all 21 folds with 3 seeds, to check that the model ranking
does not depend on the single global hyperparameter selection.
Output: results/seedvig/sensitivity_reg.json
"""
import json, os, time
import numpy as np
from train_lib import RES, Data, person_metrics, predict, split_fold, train_one
CONFIGS = {
    "KAN (H=32, G=5, lr=3e-4)": ("compact", dict(width=32, grid=5, lr=3e-4)),
    "KAN (H=8, G=5, lr=1e-3)": ("compact", dict(width=8, grid=5, lr=1e-3)),
    "LSTM (L=10, lr=1e-3)": ("seq", dict(L=10, lr=1e-3)),
    "LSTM (L=20, lr=3e-4)": ("seq", dict(L=20, lr=3e-4)),
}
out_path = os.path.join(RES, "sensitivity_reg.json")
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
d = Data("clean", load_raw=False)
persons = sorted(set(d.person), key=int)
for name, (kind, hp) in CONFIGS.items():
    for fid, p in enumerate(persons):
        key = f"{name}|{p}"
        if key in res: continue
        tr, va, te, _ = split_fold(d, p, fid)
        preds = [predict(*train_one(d, kind, "reg", hp, tr, va, seed=s)[:2], te) for s in range(3)]
        res[key] = person_metrics("reg", d.perclos[te], np.mean(preds, 0))
        json.dump(res, open(out_path, "w"), indent=1)
    rows = [res[f"{name}|{p}"] for p in persons]
    print(name, "COR %.3f RMSE %.3f" % (np.nanmean([r["cor"] for r in rows]), np.mean([r["rmse"] for r in rows])), flush=True)
