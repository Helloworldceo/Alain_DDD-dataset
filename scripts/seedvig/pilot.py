"""
Pilot run: hyperparameter selection on validation people (paper Section III-F).

For the first three leave-one-person-out folds (test persons 1, 2, 3), every
configuration in the pre-stated grids is trained once (seed 0) on the training
people, and its best validation loss on the three validation people is recorded.
The configuration with the lowest mean validation loss is selected per model and
then fixed for all folds. Test-person scores are NOT used for selection.

Grids:
  LSTM (seq):           L in {5, 10, 20}, lr in {1e-3, 3e-4}
  KAN (compact):        H in {8, 16, 32}, G in {5, 10}, lr in {1e-3, 3e-4}
  CNN-LSTM linear head: lr in {1e-3, 3e-4}
  CNN-LSTM KAN head:    lr in {1e-3, 3e-4}
  MLP (flat):           lr in {1e-3, 3e-4}
  MLP on compact set:   lr in {1e-3, 3e-4}  (control for the KAN)

Run:     python3 pilot.py [reg|cls]
Output:  results/seedvig/pilot_<task>.json
"""
import itertools
import json
import os
import sys
import time

import numpy as np

from train_lib import RES, Data, split_fold, train_one

TASK = sys.argv[1] if len(sys.argv) > 1 else "reg"
LRS = [1e-3, 3e-4]
GRIDS = {
    "LSTM": ("seq", [dict(L=L, lr=lr) for L, lr in itertools.product([5, 10, 20], LRS)]),
    "KAN": ("compact", [dict(width=h, grid=g, lr=lr) for h, g, lr in itertools.product([8, 16, 32], [5, 10], LRS)]),
    "CNN-LSTM": ("raw", [dict(head="linear", lr=lr) for lr in LRS]),
    "CNN-LSTM-KAN": ("raw", [dict(head="kan", lr=lr) for lr in LRS]),
    "MLP": ("flat", [dict(lr=lr) for lr in LRS]),
    "MLP-compact": ("compactmlp", [dict(lr=lr) for lr in LRS]),
}
PILOT_PERSONS = ["1", "2", "3"]


def main():
    data = Data("clean")
    out_path = os.path.join(RES, f"pilot_{TASK}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res = res.get("runs", res)
    for fold_id, person in enumerate(PILOT_PERSONS):
        train, val, test, val_people = split_fold(data, person, fold_id)
        print(f"fold test={person} val_people={val_people} n_train={len(train)} n_val={len(val)}", flush=True)
        for name, (kind, grid) in GRIDS.items():
            for hp in grid:
                key = f"{name}|{json.dumps(hp, sort_keys=True)}|{person}"
                if key in res:
                    continue
                t0 = time.time()
                _, _, info = train_one(data, kind, TASK, hp, train, val, seed=0)
                info.update(seconds=round(time.time() - t0, 1), val_people=val_people)
                res[key] = info
                print(f"  {name:12s} {hp} val_loss={info['best_val_loss']:.5f} "
                      f"best_epoch={info['stopped_epoch']} ({info['seconds']}s)", flush=True)
                json.dump(res, open(out_path, "w"), indent=1)
    # selection
    summary = {}
    for name, (kind, grid) in GRIDS.items():
        rows = []
        for hp in grid:
            v = [res[f"{name}|{json.dumps(hp, sort_keys=True)}|{p}"]["best_val_loss"] for p in PILOT_PERSONS]
            e = [res[f"{name}|{json.dumps(hp, sort_keys=True)}|{p}"]["stopped_epoch"] for p in PILOT_PERSONS]
            rows.append(dict(hp=hp, mean_val_loss=float(np.mean(v)), val_losses=v, best_epochs=e))
        rows.sort(key=lambda r: r["mean_val_loss"])
        summary[name] = dict(selected=rows[0]["hp"], ranking=rows)
        print(f"SELECTED {name}: {rows[0]['hp']} mean val loss {rows[0]['mean_val_loss']:.5f}", flush=True)
    json.dump(dict(task=TASK, runs=res, selection=summary), open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
