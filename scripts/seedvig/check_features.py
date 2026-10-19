"""Sanity checks of the causal features: agreement with the official features,
relation of the EOG features to PERCLOS, and a quick leave-one-person-out SVR
comparison of feature sets. Output: results/seedvig/feature_checks.json"""
import json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from extract_features import causal_ma
from splits import lopo_folds
OUT = "../../results/seedvig"
c = np.load(f"{OUT}/features_causal.npz"); o = np.load(f"{OUT}/features.npz")
assert (c["session"] == o["session"]).all()
s, t, y, person = c["session"], c["t"], c["perclos"], c["person"]
res = {}
# 1. agreement with official moving-average DE (centered smoothing, so compare with our 5-window causal mean)
import scipy.io as sio, glob
offma = []
for sess in list(dict.fromkeys(s)):
    e = sio.loadmat(f"../../data/SEED-VIG/EEG_Feature_5Bands/{sess}.mat")["de_movingAve"]
    offma.append(e.transpose(1, 0, 2).reshape(885, -1))
offma = np.concatenate(offma)
ours = causal_ma(c["de_raw"], s, t, 5)
r = [np.corrcoef(ours[s == q, j], offma[s == q, j])[0, 1] for q in np.unique(s) for j in range(85)]
res["corr_our_DE_vs_official_DE_median"] = round(float(np.nanmedian(r)), 3)
# 2. EOG features vs PERCLOS (Spearman, pooled within session ranks averaged)
names = list(c["compact_names"])
for j, nme in enumerate(names[13:]):
    rho = [spearmanr(c["eog_clean"][s == q, j], y[s == q])[0] for q in np.unique(s)]
    res[f"spearman_{nme}_vs_PERCLOS_median"] = round(float(np.nanmedian(rho)), 3)
# 3. quick LOPO SVR
def lopo(X):
    C, R = [], []
    for p, tr, te in lopo_folds(person):
        sc = StandardScaler().fit(X[tr])
        idx = np.random.default_rng(0).choice(tr, 6000, replace=False)
        pr = SVR(C=1).fit(sc.transform(X[idx]), y[idx]).predict(sc.transform(X[te]))
        C.append(np.corrcoef(pr, y[te])[0, 1]); R.append(np.sqrt(np.mean((pr - y[te]) ** 2)))
    return round(float(np.nanmean(C)), 3), round(float(np.mean(R)), 3)
sets = {
    "official DE(LDS) + EOG36": np.hstack([o["de"], o["eog"]]),
    "ours raw DE + EOG6 (causal MA5)": causal_ma(np.hstack([c["de_raw"], c["eog_raw"]]), s, t),
    "ours clean DE + EOG6 (causal MA5)": causal_ma(np.hstack([c["de_clean"], c["eog_clean"]]), s, t),
    "ours clean compact19 (causal MA5)": causal_ma(c["compact_clean"], s, t),
    "ours clean DE only (causal MA5)": causal_ma(c["de_clean"], s, t),
    "ours clean EOG6 only (causal MA5)": causal_ma(c["eog_clean"], s, t),
}
for k, X in sets.items():
    res[f"LOPO_SVR {k} [COR, RMSE]"] = lopo(X)
    print(k, res[f"LOPO_SVR {k} [COR, RMSE]"], flush=True)
json.dump(res, open(f"{OUT}/feature_checks.json", "w"), indent=1)
print(json.dumps(res, indent=1))
