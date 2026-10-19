"""
Build the SEED-VIG raw-signal window dataset.

Each session's continuous recording is cut into the 885 consecutive 8-second
segments that carry the PERCLOS labels:
  - EEG: 17 channels at 200 Hz, 50 Hz notch, band-pass 1-45 Hz (4th-order
    Butterworth, zero-phase), resampled to 100 Hz -> 800 samples per window
    (identical filtering to preprocess_clean.py, without the cleaning steps)
  - EOG: vertical and horizontal forehead EOG at 125 Hz, band-pass 0.1-15 Hz,
    resampled to 100 Hz -> 800 samples per window
Output array X has shape (n_windows, 19, 800): 17 EEG channels followed by
EOG-V and EOG-H.

Run:     python3 build_raw.py
Output:  results/seedvig/raw_windows.npy (float32, memory-mappable)
         results/seedvig/raw_meta.npz  (perclos, y3, person, session, t, channel names)
"""
import glob
import os

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(ROOT, "data", "SEED-VIG")
OUT = os.path.join(ROOT, "results", "seedvig")
N_SEG, SEG_SEC, FS_OUT = 885, 8, 100
WIN = SEG_SEC * FS_OUT


def bandpass(x, lo, hi, fs):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    sessions = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(DATA, "Raw_Data", "*.mat")))
    X = np.lib.format.open_memmap(os.path.join(OUT, "raw_windows.npy"), mode="w+", dtype=np.float32,
                                  shape=(len(sessions) * N_SEG, 19, WIN))
    perclos, person, session, t = [], [], [], []
    chn = None
    for k, s in enumerate(sessions):
        m = sio.loadmat(os.path.join(DATA, "Raw_Data", s + ".mat"))
        eeg = m["EEG"]["data"][0, 0].astype(np.float64)                 # (1,416,000, 17) at 200 Hz
        fs_eeg = int(m["EEG"]["sample_rate"][0, 0].ravel()[0])
        chn = [c[0] for c in m["EEG"]["chn"][0, 0].ravel()]
        eog = np.hstack([m["EOG"]["eog_v"][0, 0], m["EOG"]["eog_h"][0, 0]]).astype(np.float64)  # 125 Hz
        fs_eog = int(m["EOG"]["eog_config"][0, 0]["current_sample_rate"][0, 0].ravel()[0])
        assert eeg.shape[0] == N_SEG * SEG_SEC * fs_eeg, (s, eeg.shape)
        assert eog.shape[0] == N_SEG * SEG_SEC * fs_eog, (s, eog.shape)
        b, a = iirnotch(50.0, 30.0, fs_eeg)
        eeg = filtfilt(b, a, eeg, axis=0)
        eeg = resample_poly(bandpass(eeg, 1.0, 45.0, fs_eeg), FS_OUT, fs_eeg, axis=0)
        eog = resample_poly(bandpass(eog, 0.1, 15.0, fs_eog), FS_OUT, fs_eog, axis=0)
        sig = np.hstack([eeg, eog]).astype(np.float32)                   # (885*800, 19)
        X[k * N_SEG:(k + 1) * N_SEG] = sig.reshape(N_SEG, WIN, 19).transpose(0, 2, 1)
        p = sio.loadmat(os.path.join(DATA, "perclos_labels", s + ".mat"))["perclos"].ravel()
        assert len(p) == N_SEG
        perclos.append(p)
        person.extend([s.split("_")[0]] * N_SEG)
        session.extend([s] * N_SEG)
        t.append(np.arange(N_SEG))
        print(f"[{k + 1}/{len(sessions)}] {s}", flush=True)
    X.flush()
    perclos = np.concatenate(perclos).astype(np.float32)
    np.savez(os.path.join(OUT, "raw_meta.npz"), perclos=perclos, y3=np.digitize(perclos, [0.35, 0.7]),
             person=np.array(person), session=np.array(session), t=np.concatenate(t),
             channels=np.array(chn + ["EOG_V", "EOG_H"]), fs=FS_OUT, window_sec=SEG_SEC)
    print("windows", X.shape, "non-finite", int((~np.isfinite(X[:200])).sum()))


if __name__ == "__main__":
    main()
