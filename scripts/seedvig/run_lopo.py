"""
Main leave-one-person-out experiment (paper Sections III-F, III-G).

Every model uses the configuration selected in the pilot (results/seedvig/pilot_<task>.json).
For each of the 21 folds, each neural model is trained with 3 seeds on the training
people (early stopping on the 3 validation people) and its test predictions are averaged
over seeds. The SVR (regression) / SVM (classification) reference is trained once on the
same training people. Scores are computed per test person.

Run:     python3 run_lopo.py reg|cls [signals=clean|raw] [models=LSTM,KAN,...]
Output:  results/seedvig/lopo_<task>_<signals>.json   (per-person scores, run info)
         results/seedvig/lopo_<task>_<signals>_pred.npz (test predictions per model)
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from train_lib import RES, Data, person_metrics, predict, split_fold, train_one

TASK = sys.argv[1]
SIGNALS = next((a.split("=")[1] for a in sys.argv[2:] if a.startswith("signals=")), "clean")
ONLY = next((a.split("=")[1].split(",") for a in sys.argv[2:] if a.startswith("models=")), None)
SEEDS = [0, 1, 2]
KINDS = {"LSTM": "seq", "KAN": "compact", "CNN-LSTM": "raw", "CNN-LSTM-KAN": "raw", "MLP": "flat",
         "MLP-compact": "compactmlp"}


def main():
    sel = json.load(open(os.path.join(RES, f"pilot_{TASK}.json")))["selection"]
    data = Data(SIGNALS)
    persons = sorted(set(data.person), key=int)
    out = os.path.join(RES, f"lopo_{TASK}_{SIGNALS}.json")
    pred_path = out.replace(".json", "_pred.npz")
    res = json.load(open(out)) if os.path.exists(out) else {"scores": {}, "info": {}}
    preds = dict(np.load(pred_path)) if os.path.exists(pred_path) else {}
    models = ONLY or (["SVR"] + list(KINDS))
    y = data.target(TASK)
    for name in models:
        for fold_id, person in enumerate(persons):
            key = f"{name}|{person}"
            if key in res["scores"]:
                continue
            train, val, test, _ = split_fold(data, person, fold_id)
            t0 = time.time()
            if name == "SVR":
                sc = StandardScaler().fit(data.feat91_ma[train])
                Xtr, Xte = sc.transform(data.feat91_ma[train]), sc.transform(data.feat91_ma[test])
                if TASK == "reg":
                    p = np.clip(SVR(C=1.0).fit(Xtr, y[train]).predict(Xte), 0, 1)
                else:
                    p = SVC(C=1.0, class_weight="balanced").fit(Xtr, y[train]).predict(Xte)
                info = {}
            else:
                hp = sel[name]["selected"]
                ps, infos = [], []
                for seed in SEEDS:
                    m, b, inf = train_one(data, KINDS[name], TASK, hp, train, val, seed=seed)
                    ps.append(predict(m, b, test))
                    infos.append(inf)
                p = np.mean(ps, axis=0)
                info = dict(hp=hp, epochs=[i["stopped_epoch"] for i in infos])
            res["scores"][key] = person_metrics(TASK, y[test], p)
            res["info"][key] = dict(info, seconds=round(time.time() - t0, 1), n_test=int(len(test)))
            preds[f"{name}|{person}"] = np.asarray(p, dtype=np.float32)
            json.dump(res, open(out, "w"), indent=1)
            np.savez_compressed(pred_path, **preds)
            print(f"{name:12s} person {person:>2s} {res['scores'][key]} ({res['info'][key]['seconds']}s)", flush=True)
    # summary
    summ = {}
    for name in models:
        rows = [res["scores"][f"{name}|{p}"] for p in persons]
        summ[name] = {k: [float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows]))]
                      for k in rows[0]}
        print("SUMMARY", name, {k: f"{v[0]:.3f}±{v[1]:.3f}" for k, v in summ[name].items()}, flush=True)
    res["summary"] = summ
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
