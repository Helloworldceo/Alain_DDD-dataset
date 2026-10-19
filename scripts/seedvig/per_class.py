"""Per-class F1 and pooled confusion matrices for the three-class task (from saved predictions).
Per-class F1 is averaged over the people whose data contain that class.
Output: results/seedvig/per_class_cls.json"""
import json, os
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from train_lib import RES, Data, split_fold
d = Data("clean", load_raw=False)
preds = np.load(os.path.join(RES, "lopo_cls_clean_pred.npz"))
persons = sorted(set(d.person), key=int)
models = ["SVR", "MLP", "MLP-compact", "LSTM", "KAN", "CNN-LSTM", "CNN-LSTM-KAN"]
out = {}
for m in models:
    per = {0: [], 1: [], 2: []}; Y = []; P = []
    for fid, p in enumerate(persons):
        _, _, te, _ = split_fold(d, p, fid)
        y = d.y3[te]; pr = preds[f"{m}|{p}"]; pr = pr.argmax(1) if pr.ndim == 2 else pr.astype(int)
        f = f1_score(y, pr, labels=[0, 1, 2], average=None, zero_division=0)
        for c in range(3):
            if (y == c).any(): per[c].append(f[c])
        Y.append(y); P.append(pr)
    Y = np.concatenate(Y); P = np.concatenate(P)
    cm = confusion_matrix(Y, P, labels=[0, 1, 2])
    out[m] = dict(f1_awake=float(np.mean(per[0])), f1_tired=float(np.mean(per[1])), f1_drowsy=float(np.mean(per[2])),
                  n_people=[len(per[c]) for c in range(3)], confusion=cm.tolist(),
                  recall=(cm.diagonal() / cm.sum(1)).round(3).tolist())
    print(f"{m:13s} F1 awake {out[m]['f1_awake']:.3f} tired {out[m]['f1_tired']:.3f} drowsy {out[m]['f1_drowsy']:.3f} "
          f"people {out[m]['n_people']} recall {out[m]['recall']}")
json.dump(out, open(os.path.join(RES, "per_class_cls.json"), "w"), indent=1)
