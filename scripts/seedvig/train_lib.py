"""
Shared data preparation, training, and evaluation for the SEED-VIG experiments.

Inputs (results/seedvig/):
  features_causal.npz   causal DE / EOG / compact features (raw and clean)
  clean_windows.npy     cleaned raw windows (20355, 19, 800)
  clean_meta.npz        labels, person, session, t, cleaning flags

Model kinds:
  seq      LSTMSeq on sequences of 91 features (85 DE + 6 EOG) of the current and L-1 previous windows
  raw      CNNLSTM on cleaned raw windows
  compact  KANNet on the 19 compact features (causal 5-window mean)
  flat     MLP / SVR on 91 features (causal 5-window mean)
  compactmlp  MLP on the 19 compact features (control for the KAN)
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from extract_features import causal_ma
from models import CNNLSTM, KANNet, LSTMSeq, MLP

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RES = os.path.join(ROOT, "results", "seedvig")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Data:
    def __init__(self, signals="clean", load_raw=True):
        f = np.load(os.path.join(RES, "features_causal.npz"))
        self.perclos = f["perclos"].astype(np.float32)
        self.y3 = f["y3"].astype(np.int64)
        self.person, self.session, self.t, self.flag = f["person"], f["session"], f["t"], f["flag"].astype(bool)
        self.feat91 = np.hstack([f[f"de_{signals}"], f[f"eog_{signals}"]]).astype(np.float32)
        self.feat91_ma = causal_ma(self.feat91, self.session, self.t).astype(np.float32)
        self.compact_ma = causal_ma(f[f"compact_{signals}"], self.session, self.t).astype(np.float32)
        self.compact_names = list(f["compact_names"])
        self.raw = np.load(os.path.join(RES, f"{signals}_windows.npy")) if load_raw else None
        # index of the previous window of the same session (or itself at the session start)
        order = np.lexsort((self.t, self.session))
        prev = np.arange(len(self.t))
        same = self.session[order][1:] == self.session[order][:-1]
        prev[order[1:][same]] = order[:-1][same]
        self.prev = prev

    def seq_index(self, idx, L):
        out = np.empty((len(idx), L), dtype=np.int64)
        cur = np.asarray(idx)
        for k in range(L - 1, -1, -1):
            out[:, k] = cur
            cur = self.prev[cur]
        return out

    def target(self, task):
        return self.perclos if task == "reg" else self.y3


def validation_people(train_people, fold_id, n=3):
    rng = np.random.default_rng(1000 + int(fold_id))
    return sorted(rng.choice(sorted(set(train_people), key=int), size=n, replace=False).tolist(), key=int)


def build_model(kind, task, hp):
    if kind == "seq":
        return LSTMSeq(91, task, hidden=64, dropout=0.3)
    if kind == "raw":
        return CNNLSTM(task, head=hp.get("head", "linear"))
    if kind == "compact":
        return KANNet(len(hp.get("cols", range(19))), task, width=hp["width"], grid=hp["grid"])
    if kind == "flat":
        return MLP(91, task)
    if kind == "compactmlp":
        return MLP(19, task)
    raise ValueError(kind)


class Batcher:
    """Creates standardized model inputs for a set of window indices."""

    def __init__(self, data, kind, train_idx, hp):
        self.d, self.kind, self.L = data, kind, hp.get("L", 1)
        self.cols = list(hp["cols"]) if "cols" in hp else None
        if kind == "raw":
            sub = train_idx[:: max(1, len(train_idx) // 3000)]
            x = data.raw[sub]
            self.mu = x.mean(axis=(0, 2), keepdims=True)[0]
            self.sd = x.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        else:
            src = self.source()
            self.mu, self.sd = src[train_idx].mean(0), src[train_idx].std(0) + 1e-6

    def source(self):
        src = {"seq": self.d.feat91, "compact": self.d.compact_ma, "compactmlp": self.d.compact_ma,
               "flat": self.d.feat91_ma}[self.kind]
        return src[:, self.cols] if self.cols is not None else src

    def __call__(self, idx):
        if self.kind == "raw":
            x = (self.d.raw[idx] - self.mu) / self.sd
        elif self.kind == "seq":
            x = (self.d.feat91[self.d.seq_index(idx, self.L)] - self.mu) / self.sd
        else:
            x = (self.source()[idx] - self.mu) / self.sd
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    def select(self, cols):
        """Restrict a feature-based batcher to a subset of feature columns (modality ablation)."""
        self.cols = list(cols)


def predict(model, batcher, idx, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            out.append(model(batcher(idx[i:i + bs]).to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()


def train_one(data, kind, task, hp, train_idx, val_idx, seed, max_epochs=60, patience=10, bs=128, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    batcher = Batcher(data, kind, train_idx, hp)
    model = build_model(kind, task, hp).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    y_all = data.target(task)
    if task == "reg":
        loss_fn = nn.MSELoss()
    else:
        counts = np.bincount(y_all[train_idx], minlength=3).astype(np.float32)
        w = torch.tensor(len(train_idx) / (3 * np.maximum(counts, 1)), device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=w)
    yv = torch.from_numpy(y_all[val_idx])
    best, best_state, wait, hist = np.inf, None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x = batcher(b).to(DEVICE)
            y = torch.from_numpy(y_all[b]).to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = torch.from_numpy(predict(model, batcher, val_idx))
        vl = float(loss_fn(pv.to(DEVICE), yv.to(DEVICE)))
        hist.append(vl)
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose:
            print(f"    ep {ep + 1:2d} val {vl:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, batcher, dict(best_val_loss=best, epochs=len(hist), stopped_epoch=int(np.argmin(hist)) + 1)


def person_metrics(task, y, p):
    if task == "reg":
        cor = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 1e-9 and np.std(y) > 1e-9 else float("nan")
        return dict(cor=cor, rmse=float(np.sqrt(np.mean((p - y) ** 2))))
    pred = p.argmax(1) if p.ndim == 2 else p
    # macro-F1 over the classes present in this person's data (some people never reach a class)
    return dict(acc=float((pred == y).mean()), f1=float(f1_score(y, pred, average="macro", labels=np.unique(y),
                                                                    zero_division=0)))


def split_fold(data, test_person, fold_id):
    test = np.where(data.person == test_person)[0]
    train_people = sorted(set(data.person) - {test_person}, key=int)
    val_people = validation_people(train_people, fold_id)
    usable = ~data.flag
    val = np.where(np.isin(data.person, val_people) & usable)[0]
    train = np.where(~np.isin(data.person, val_people + [test_person]) & usable)[0]
    return train, val, test, val_people
