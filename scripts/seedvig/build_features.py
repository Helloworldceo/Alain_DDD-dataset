"""
Build the SEED-VIG feature dataset.

For each of the 23 sessions (21 people) it collects, for the 885 consecutive
8-second samples:
  - EEG differential entropy (DE, LDS-smoothed), 17 channels x 5 bands = 85 features
  - EEG power spectral density (PSD, LDS-smoothed), 17 x 5 = 85 features
  - forehead EOG features (ICA separation), 36 features
  - PERCLOS vigilance label (continuous, 0 = awake ... 1 = drowsy)
  - 3-class label: awake (<0.35), tired (0.35-0.7), drowsy (>=0.7)
  - person id (sessions of the same person share an id) and session id

Run:     python3 build_features.py
Output:  results/seedvig/features.npz
"""
import glob
import os

import numpy as np
import scipy.io as sio

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(ROOT, "data", "SEED-VIG")
OUT = os.path.join(ROOT, "results", "seedvig")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def three_class(p):
    return np.digitize(p, [0.35, 0.7]).astype(np.int64)


def main():
    os.makedirs(OUT, exist_ok=True)
    sessions = sorted(os.path.basename(f) for f in glob.glob(os.path.join(DATA, "perclos_labels", "*.mat")))
    de, psd, eog, perclos, person, session, t = [], [], [], [], [], [], []
    for s in sessions:
        e = sio.loadmat(os.path.join(DATA, "EEG_Feature_5Bands", s))
        o = sio.loadmat(os.path.join(DATA, "EOG_Feature", s))
        p = sio.loadmat(os.path.join(DATA, "perclos_labels", s))["perclos"].ravel()
        n = len(p)
        de.append(e["de_LDS"].transpose(1, 0, 2).reshape(n, -1))      # (885, 17*5)
        psd.append(e["psd_LDS"].transpose(1, 0, 2).reshape(n, -1))
        eog.append(o["features_table_ica"])
        perclos.append(p)
        person.extend([s.split("_")[0]] * n)
        session.extend([s[:-4]] * n)
        t.append(np.arange(n))
    out = dict(
        de=np.concatenate(de).astype(np.float32),
        psd=np.concatenate(psd).astype(np.float32),
        eog=np.concatenate(eog).astype(np.float32),
        perclos=np.concatenate(perclos).astype(np.float32),
        person=np.array(person),
        session=np.array(session),
        t=np.concatenate(t),
        feature_names_eeg=np.array([f"ch{c+1}_{b}" for c in range(17) for b in BANDS]),
    )
    out["y3"] = three_class(out["perclos"])
    np.savez_compressed(os.path.join(OUT, "features.npz"), **out)
    print(f"{len(sessions)} sessions, {len(np.unique(out['person']))} people, {len(out['perclos'])} samples")
    print("DE", out["de"].shape, "PSD", out["psd"].shape, "EOG", out["eog"].shape)
    print("3-class counts (awake/tired/drowsy):", np.bincount(out["y3"]))
    print("non-finite values: DE", (~np.isfinite(out["de"])).sum(), "EOG", (~np.isfinite(out["eog"])).sum())


if __name__ == "__main__":
    main()
