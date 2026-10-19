"""
Causal feature extraction from SEED-VIG windows (filtered-only and cleaned).

Every feature of a window uses only that window's samples, so the features can
be computed in real time. Temporal smoothing, when used, is a causal moving
average over the current and previous windows (see causal_ma).

EEG (17 channels x 5 bands = 85 features):
  differential entropy DE = 0.5 * log(2*pi*e * P_band), P_band from Welch's PSD
  (2-s Hann segments, 50% overlap); bands delta 1-4, theta 4-8, alpha 8-14,
  beta 14-31, gamma 31-45 Hz.
EOG (6 features):
  blink rate, mean blink amplitude (prominence), mean blink duration (width at
  half prominence) and blink-time fraction (share of samples above threshold),
  from the 10-Hz low-passed vertical EOG; the polarity of the vertical EOG is set
  per session from its skewness so that blinks are positive, and blinks are peaks
  0.05-3 s wide exceeding a session-level threshold of median + 3 robust SDs; saccade rate (horizontal EOG velocity peaks above median + 4 robust
  SDs of the session), and the slow-eye-movement fraction (0.1-1 Hz power of the
  horizontal EOG relative to 0.1-15 Hz).
Compact set for KAN (19 features):
  mean DE per band over the temporal (FT7, FT8, T7, T8, TP7, TP8) and the
  posterior (CP1 ... O2) channels (10), posterior power ratios theta/beta,
  alpha/beta, (theta+alpha)/beta (3), and the 6 EOG features.

Run:     python3 extract_features.py
Output:  results/seedvig/features_causal.npz
"""
import os

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from scipy.stats import skew

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "results", "seedvig")
FS = 100
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 14), ("beta", 14, 31), ("gamma", 31, 45)]
TEMPORAL, POSTERIOR = list(range(0, 6)), list(range(6, 17))


def robust_sd(x):
    return np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-12


def band_powers(X):
    """X: (n, ch, samples) -> band power (n, ch, 5)."""
    f, P = welch(X, fs=FS, nperseg=2 * FS, axis=-1)
    df = f[1] - f[0]
    return np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) * df for _, lo, hi in BANDS], axis=-1)


def eeg_features(X):
    out = []
    for i in range(0, len(X), 2000):
        out.append(band_powers(np.asarray(X[i:i + 2000, :17], dtype=np.float64)))
    return np.concatenate(out)


def eog_features(V, H, session):
    """V, H: (n, samples) vertical / horizontal EOG; thresholds are set per session (label-blind)."""
    sos_b = butter(4, 10, btype="low", fs=FS, output="sos")
    feats = np.zeros((len(V), 6))
    for s in np.unique(session):
        idx = np.where(session == s)[0]
        v = sosfiltfilt(sos_b, V[idx].reshape(-1))
        if skew(v) < 0:
            v = -v
        v = v.reshape(len(idx), -1)
        thr = np.median(v) + 3 * robust_sd(v)
        hv = np.abs(np.diff(H[idx], axis=1)) * FS
        sthr = np.median(hv) + 4 * robust_sd(hv)
        for j, w in enumerate(v):
            pk, pr = find_peaks(w, height=thr, distance=int(0.2 * FS), width=(int(0.05 * FS), 3 * FS))
            n = len(pk)
            feats[idx[j], 0] = n / 8.0 * 60.0                                   # blinks per minute
            feats[idx[j], 1] = pr["prominences"].mean() if n else 0.0
            feats[idx[j], 2] = (pr["widths"].mean() / FS) if n else 0.0          # seconds
            feats[idx[j], 3] = (w > thr).mean()
            sp, _ = find_peaks(hv[j], height=sthr, distance=int(0.1 * FS))
            feats[idx[j], 4] = len(sp) / 8.0 * 60.0                             # saccades per minute
            f, P = welch(H[idx[j]], fs=FS, nperseg=len(H[idx[j]]))
            feats[idx[j], 5] = P[(f >= 0.1) & (f < 1)].sum() / (P[(f >= 0.1) & (f < 15)].sum() + 1e-12)
    return feats


def compact(bp, de, eog):
    tde = de[:, TEMPORAL].mean(1)
    pde = de[:, POSTERIOR].mean(1)
    pb = bp[:, POSTERIOR].mean(1)
    ratios = np.log(np.stack([pb[:, 1] / pb[:, 3], pb[:, 2] / pb[:, 3], (pb[:, 1] + pb[:, 2]) / pb[:, 3]], 1))
    return np.hstack([tde, pde, ratios, eog])


COMPACT_NAMES = ([f"temporal_DE_{b}" for b, _, _ in BANDS] + [f"posterior_DE_{b}" for b, _, _ in BANDS] +
                 ["log_theta/beta", "log_alpha/beta", "log_(theta+alpha)/beta",
                  "blink_rate", "blink_amplitude", "blink_duration", "blink_time_fraction",
                  "saccade_rate", "slow_eye_movement_fraction"])


def causal_ma(F, session, t, k=5):
    """Mean of the current and the previous k-1 windows of the same session."""
    out = np.empty_like(F, dtype=np.float64)
    for s in np.unique(session):
        idx = np.where(session == s)[0]
        idx = idx[np.argsort(t[idx])]
        c = np.cumsum(np.vstack([np.zeros((1, F.shape[1])), F[idx]]), axis=0)
        n = np.arange(1, len(idx) + 1)
        lo = np.maximum(n - k, 0)
        out[idx] = (c[n] - c[lo]) / (n - lo)[:, None]
    return out


def main():
    meta = np.load(os.path.join(OUT, "clean_meta.npz"))
    session = meta["session"]
    res = dict(perclos=meta["perclos"], y3=meta["y3"], person=meta["person"], session=session, t=meta["t"],
               flag=meta["flag"], compact_names=np.array(COMPACT_NAMES))
    for name in ("raw", "clean"):
        X = np.load(os.path.join(OUT, f"{name}_windows.npy"), mmap_mode="r")
        bp = eeg_features(X)
        de = 0.5 * np.log(2 * np.pi * np.e * bp + 1e-12)
        eog = eog_features(np.asarray(X[:, 17], dtype=np.float64), np.asarray(X[:, 18], dtype=np.float64), session)
        res[f"de_{name}"] = de.reshape(len(de), -1).astype(np.float32)
        res[f"eog_{name}"] = eog.astype(np.float32)
        res[f"compact_{name}"] = compact(bp, de, eog).astype(np.float32)
        print(name, "DE", res[f"de_{name}"].shape, "EOG", eog.shape, "compact", res[f"compact_{name}"].shape,
              "non-finite", int((~np.isfinite(res[f"compact_{name}"])).sum()), flush=True)
    np.savez_compressed(os.path.join(OUT, "features_causal.npz"), **res)


if __name__ == "__main__":
    main()
