"""
Artifact cleaning of the SEED-VIG raw recordings.

Per session (continuous recording, before windowing):
  1. EEG: 50 Hz notch, band-pass 1-45 Hz (4th-order Butterworth, zero-phase),
     resampled 200 -> 100 Hz.  EOG: band-pass 0.1-15 Hz, resampled 125 -> 100 Hz.
  2. Bad-channel detection: a channel is bad if its SD is below 0.15 or above 4
     times the median channel SD of the session, or if more than 5% of its samples
     exceed 200 uV; bad channels are interpolated with spherical splines
     (standard 10-05 positions).
  3. Ocular artifact removal: extended-Infomax ICA on the EEG channels; a component
     is removed if its time course correlates with the vertical or horizontal EOG
     (|r| >= 0.30, or |r| >= 0.20 with a z-score of |r| >= 3 among the session's
     components), at most 3 components per session. The number of ICA components
     equals the data rank minus one (16 minus the number of interpolated channels).
  4. Windowing into the 885 labeled 8-s segments (800 samples at 100 Hz).
  5. Label-blind window flagging: peak-to-peak EEG amplitude above
     median + 5 robust SDs of the session, or a flat window, is flagged.
     Flagged windows are excluded from training only; every window is still tested.

Run:     python3 preprocess_clean.py
Output:  results/seedvig/clean_windows.npy  (20355, 19, 800) float32
         results/seedvig/clean_meta.npz      flags + labels
         results/seedvig/cleaning_report.json
"""
import json
import os
import time

import mne
import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, filtfilt

mne.set_log_level("ERROR")
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(ROOT, "data", "SEED-VIG")
OUT = os.path.join(ROOT, "results", "seedvig")
N_SEG, SEG_SEC, FS = 885, 8, 100
WIN = SEG_SEC * FS
NAME_FIX = {"PZ": "Pz", "POZ": "POz", "OZ": "Oz"}


def bandpass(x, lo, hi, fs):
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=fs, output="sos"), x, axis=0)


def robust_z(v):
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-12
    return (v - med) / mad


def clean_session(path):
    m = sio.loadmat(path)
    eeg = m["EEG"]["data"][0, 0].astype(np.float64)
    fs_eeg = int(m["EEG"]["sample_rate"][0, 0].ravel()[0])
    chn = [NAME_FIX.get(c[0], c[0]) for c in m["EEG"]["chn"][0, 0].ravel()]
    eog = np.hstack([m["EOG"]["eog_v"][0, 0], m["EOG"]["eog_h"][0, 0]]).astype(np.float64)
    fs_eog = int(m["EOG"]["eog_config"][0, 0]["current_sample_rate"][0, 0].ravel()[0])

    b, a = iirnotch(50.0, 30.0, fs_eeg)
    eeg = filtfilt(b, a, eeg, axis=0)
    eeg = resample_poly(bandpass(eeg, 1.0, 45.0, fs_eeg), FS, fs_eeg, axis=0)
    eog = resample_poly(bandpass(eog, 0.1, 15.0, fs_eog), FS, fs_eog, axis=0)

    info = mne.create_info(chn + ["EOG_V", "EOG_H"], FS, ["eeg"] * len(chn) + ["eog", "eog"])
    raw = mne.io.RawArray(np.hstack([eeg, eog]).T * 1e-6, info)   # MNE uses volts
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))

    sd = eeg.std(axis=0)
    ratio = sd / np.median(sd)
    big = (np.abs(eeg) > 200).mean(axis=0)
    bads = [chn[i] for i in range(len(chn)) if ratio[i] < 0.15 or ratio[i] > 4 or big[i] > 0.05]
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True)

    ica = mne.preprocessing.ICA(n_components=len(chn) - len(bads) - 1, method="infomax", fit_params=dict(extended=True),
                                random_state=0, max_iter=500)
    ica.fit(raw, picks="eeg", decim=4)
    src = ica.get_sources(raw).get_data()
    eogd = raw.get_data(picks="eog")
    r = np.array([[abs(np.corrcoef(s, e)[0, 1]) for e in eogd] for s in src]).max(axis=1)
    rz = robust_z(r)
    cand = [i for i in np.argsort(-r) if r[i] >= 0.30 or (r[i] >= 0.20 and rz[i] >= 3)][:3]
    ica.exclude = [int(i) for i in cand]
    if cand:
        ica.apply(raw)

    x = raw.get_data() * 1e6                                        # back to uV, (19, n)
    win = x[:, :N_SEG * WIN].reshape(19, N_SEG, WIN).transpose(1, 0, 2).astype(np.float32)
    ptp = np.ptp(win[:, :17], axis=2).max(axis=1)
    flat = win[:, :17].std(axis=2).min(axis=1) < 0.1
    flag = (robust_z(np.log(ptp)) > 5) | flat
    rep = dict(bad_channels=[str(b) for b in bads], ica_removed=ica.exclude, ica_eog_corr=[round(float(r[i]), 3) for i in ica.exclude],
               windows_flagged=int(flag.sum()))
    return win, flag, rep, chn


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = np.load(os.path.join(OUT, "raw_meta.npz"))
    sessions = list(dict.fromkeys(meta["session"]))
    X = np.lib.format.open_memmap(os.path.join(OUT, "clean_windows.npy"), mode="w+", dtype=np.float32,
                                  shape=(len(sessions) * N_SEG, 19, WIN))
    flags, report = [], {}
    for k, s in enumerate(sessions):
        t0 = time.time()
        win, flag, rep, chn = clean_session(os.path.join(DATA, "Raw_Data", s + ".mat"))
        X[k * N_SEG:(k + 1) * N_SEG] = win
        flags.append(flag)
        report[s] = rep
        print(f"[{k + 1}/{len(sessions)}] {s}: bad={rep['bad_channels']} ica_removed={rep['ica_removed']} "
              f"r={rep['ica_eog_corr']} flagged={rep['windows_flagged']} ({time.time() - t0:.0f}s)", flush=True)
    X.flush()
    flags = np.concatenate(flags)
    np.savez(os.path.join(OUT, "clean_meta.npz"), flag=flags, perclos=meta["perclos"], y3=meta["y3"],
             person=meta["person"], session=meta["session"], t=meta["t"], channels=meta["channels"])
    summary = dict(sessions=len(sessions),
                   sessions_with_bad_channels=sum(bool(r["bad_channels"]) for r in report.values()),
                   bad_channels_total=sum(len(r["bad_channels"]) for r in report.values()),
                   ica_components_removed_total=sum(len(r["ica_removed"]) for r in report.values()),
                   windows_flagged=int(flags.sum()), windows_total=int(len(flags)))
    json.dump(dict(summary=summary, sessions=report), open(os.path.join(OUT, "cleaning_report.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
