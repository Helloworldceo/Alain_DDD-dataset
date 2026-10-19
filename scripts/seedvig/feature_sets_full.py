"""Feature-set comparison with the reference SVR using the full training data and exactly the
splits of the main experiment (training people only; flagged windows excluded from training).
Output: results/seedvig/feature_sets_full.json"""
import json, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from extract_features import causal_ma
from train_lib import RES, split_fold, Data
c = np.load(os.path.join(RES, "features_causal.npz")); o = np.load(os.path.join(RES, "features.npz"))
d = Data("clean", load_raw=False); s, t, y = d.session, d.t, d.perclos
assert (o["session"] == s).all()
sets = {
    "SEED-VIG features (session-smoothed DE) + EOG": (np.hstack([o["de"], o["eog"]]), 121, False),
    "Filtered only: DE + EOG": (causal_ma(np.hstack([c["de_raw"], c["eog_raw"]]), s, t), 91, True),
    "Cleaned: DE + EOG": (causal_ma(np.hstack([c["de_clean"], c["eog_clean"]]), s, t), 91, True),
    "Cleaned: DE only": (causal_ma(c["de_clean"], s, t), 85, True),
    "Cleaned: EOG only": (causal_ma(c["eog_clean"], s, t), 6, True),
    "Cleaned: compact set": (causal_ma(c["compact_clean"], s, t), 19, True),
}
persons = sorted(set(d.person), key=int); out = {}
for name, (X, dim, causal) in sets.items():
    C, R = [], []
    for fid, p in enumerate(persons):
        tr, _, te, _ = split_fold(d, p, fid)
        sc = StandardScaler().fit(X[tr])
        pr = np.clip(SVR(C=1.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te])), 0, 1)
        C.append(np.corrcoef(pr, y[te])[0, 1]); R.append(np.sqrt(np.mean((pr - y[te]) ** 2)))
    out[name] = dict(dim=dim, causal=causal, cor=float(np.nanmean(C)), rmse=float(np.mean(R)))
    print(f"{name:48s} COR {out[name]['cor']:.3f} RMSE {out[name]['rmse']:.3f}", flush=True)
json.dump(out, open(os.path.join(RES, "feature_sets_full.json"), "w"), indent=1)
