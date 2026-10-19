"""
Interpretation of the KAN learned on the compact features (PERCLOS regression).

The KAN with the configuration selected in the pilot is trained with 3 seeds on all
people, with early stopping on 3 validation people (seed-specific choice). For each
input feature, its effect curve is the predicted PERCLOS as that feature varies over
its observed range (2.5th to 97.5th percentile of the usable windows) while all other
features are held at their mean (zero after standardization); curves are averaged over seeds. Feature influence is the
range (max - min) of the curve. The first-layer edge functions phi_{j,i} are also stored.

Run:     python3 kan_interpret.py
Output:  results/seedvig/kan_interpretation.json
         figures/seedvig/fig_kan_effects.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_lib import DEVICE, RES, Data, train_one

FIG = os.path.join(os.path.dirname(RES), "..", "figures", "seedvig")
PRETTY = {
    "temporal_DE_delta": "Temporal DE, delta", "temporal_DE_theta": "Temporal DE, theta",
    "temporal_DE_alpha": "Temporal DE, alpha", "temporal_DE_beta": "Temporal DE, beta",
    "temporal_DE_gamma": "Temporal DE, gamma", "posterior_DE_delta": "Posterior DE, delta",
    "posterior_DE_theta": "Posterior DE, theta", "posterior_DE_alpha": "Posterior DE, alpha",
    "posterior_DE_beta": "Posterior DE, beta", "posterior_DE_gamma": "Posterior DE, gamma",
    "log_theta/beta": "log theta/beta", "log_alpha/beta": "log alpha/beta",
    "log_(theta+alpha)/beta": "log (theta+alpha)/beta", "blink_rate": "Blink rate (/min)",
    "blink_amplitude": "Blink amplitude (uV)", "blink_duration": "Blink duration (s)",
    "blink_time_fraction": "Blink-time fraction", "saccade_rate": "Saccade rate (/min)",
    "slow_eye_movement_fraction": "Slow-eye-movement fraction",
}


def main():
    hp = json.load(open(os.path.join(RES, "pilot_reg.json")))["selection"]["KAN"]["selected"]
    data = Data("clean", load_raw=False)
    names = data.compact_names
    people = sorted(set(data.person), key=int)
    usable_all = ~data.flag
    q = np.percentile(data.compact_ma[usable_all], [2.5, 97.5], axis=0)          # (2, 19) original units
    grid = np.linspace(q[0], q[1], 101).T                                         # (19, 101)
    curves, edges, infos = [], [], []
    for seed in range(3):
        rng = np.random.default_rng(500 + seed)
        val_people = sorted(rng.choice(people, 3, replace=False).tolist(), key=int)
        usable = ~data.flag
        val = np.where(np.isin(data.person, val_people) & usable)[0]
        train = np.where(~np.isin(data.person, val_people) & usable)[0]
        model, batcher, info = train_one(data, "compact", "reg", hp, train, val, seed=seed)
        infos.append(dict(info, val_people=val_people))
        model.eval()
        c = np.zeros((len(names), grid.shape[1]))
        with torch.no_grad():
            for i in range(len(names)):
                zi = (grid[i] - batcher.mu[i]) / batcher.sd[i]
                x = torch.zeros(grid.shape[1], len(names), device=DEVICE)
                x[:, i] = torch.tensor(zi, dtype=torch.float32, device=DEVICE)
                c[i] = model(x).cpu().numpy()
            zz = torch.linspace(-2.5, 2.5, 101, device=DEVICE)
            e = np.stack([[model.l1.edge_function(i, j, zz).cpu().numpy() for j in range(model.l1.out_dim)]
                          for i in range(len(names))])
        curves.append(c)
        edges.append(e)
        mu, sd = batcher.mu, batcher.sd
    curves = np.mean(curves, axis=0)
    influence = curves.max(1) - curves.min(1)
    order = np.argsort(-influence)
    out = dict(hp=hp, runs=infos, grid={names[i]: grid[i].tolist() for i in range(len(names))}, feature_names=names,
               influence={names[i]: float(influence[i]) for i in order},
               direction={names[i]: float(np.sign(curves[i, -1] - curves[i, 0])) for i in order},
               curves={names[i]: curves[i].tolist() for i in range(len(names))},
               feature_mean=dict(zip(names, map(float, mu))), feature_sd=dict(zip(names, map(float, sd))))
    json.dump(out, open(os.path.join(RES, "kan_interpretation.json"), "w"), indent=1)
    np.save(os.path.join(RES, "kan_edge_functions.npy"), np.mean(edges, axis=0))
    for i in order:
        print(f"{names[i]:30s} influence {influence[i]:.3f} direction {'+' if curves[i, -1] > curves[i, 0] else '-'}")

    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    top = order[:8]
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.5, 1, 1, 1, 1], wspace=0.45, hspace=0.6)
    ax = fig.add_subplot(gs[:, 0])
    ax.barh(range(len(order)), influence[order][::-1], color="#2a78d6")
    ax.set_yticks(range(len(order)), [PRETTY.get(names[i], names[i]) for i in order][::-1], fontsize=6.5)
    ax.set_xlabel("Range of predicted PERCLOS")
    ax.set_title("(a) Feature influence", loc="left", fontsize=9)
    for k, i in enumerate(top):
        ax = fig.add_subplot(gs[k // 4, 1 + k % 4])
        ax.plot(grid[i], curves[i], color="#2a78d6", lw=1.8)
        ax.set_title(PRETTY.get(names[i], names[i]), fontsize=7.5)
        ax.set_ylim(0, 1)
        if k % 4 == 0:
            ax.set_ylabel("Predicted PERCLOS")
    fig.text(0.62, 0.995, "(b) Effect of each feature (others held at their mean)", ha="center", va="top", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_kan_effects.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
