"""Recompute per-person classification scores from saved predictions (no retraining),
using macro-F1 over the classes present in each person's data.
Run: python3 rescore_cls.py [signals=clean]"""
import json, os, sys
import numpy as np
from train_lib import RES, Data, person_metrics, split_fold
SIGNALS = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("signals=")), "clean")
path = os.path.join(RES, f"lopo_cls_{SIGNALS}.json")
res = json.load(open(path)); preds = np.load(path.replace(".json", "_pred.npz"))
d = Data(SIGNALS, load_raw=False)
persons = sorted(set(d.person), key=int)
changed = 0
for key in list(res["scores"]):
    name, person = key.split("|")
    _, _, test, _ = split_fold(d, person, persons.index(person))
    new = person_metrics("cls", d.y3[test], preds[key])
    assert abs(new["acc"] - res["scores"][key]["acc"]) < 1e-9, key
    changed += abs(new["f1"] - res["scores"][key]["f1"]) > 1e-9
    res["scores"][key] = new
json.dump(res, open(path, "w"), indent=1)
print("rescored", len(res["scores"]), "entries; F1 changed in", changed)
