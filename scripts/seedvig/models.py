"""
Models for the SEED-VIG study (see seedvig-paper/paper/main.tex, Section III-E).

  LSTMSeq      LSTM on sequences of window features (causal)
  CNNLSTM      1-D CNN + LSTM on raw 19 x 800 windows, with a linear or KAN head
  KANNet       B-spline KAN on the compact feature set
  MLP          reference multilayer perceptron

KANLayer follows the B-spline formulation of Liu et al. (ICLR 2025):
  phi(x) = w_b * SiLU(x) + w_s * sum_k c_k B_k(x)
with cubic B-splines on a uniform grid of G intervals over [-3, 3] (inputs are
standardized), extended by the spline order on both sides.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid=5, order=3, lo=-3.0, hi=3.0):
        super().__init__()
        self.in_dim, self.out_dim, self.grid, self.order = in_dim, out_dim, grid, order
        h = (hi - lo) / grid
        knots = torch.arange(-order, grid + order + 1, dtype=torch.float32) * h + lo
        self.register_buffer("knots", knots.expand(in_dim, -1).contiguous())      # (in, G+2k+1)
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.spline_coef = nn.Parameter(torch.empty(out_dim, in_dim, grid + order))
        self.spline_scale = nn.Parameter(torch.ones(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.normal_(self.spline_coef, std=0.1 / (in_dim ** 0.5))

    def b_splines(self, x):
        """x: (batch, in) -> (batch, in, G + order) cubic B-spline basis values (Cox-de Boor)."""
        g = self.knots
        x = x.unsqueeze(-1)
        b = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.order + 1):
            b = ((x - g[:, :-(k + 1)]) / (g[:, k:-1] - g[:, :-(k + 1)]) * b[..., :-1] +
                 (g[:, k + 1:] - x) / (g[:, k + 1:] - g[:, 1:-k]) * b[..., 1:])
        return b

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        coef = self.spline_coef * self.spline_scale.unsqueeze(-1)
        spline = torch.einsum("bik,oik->bo", self.b_splines(x), coef)
        return base + spline

    @torch.no_grad()
    def edge_function(self, i, j, x):
        """Value of the learned function phi_{j,i} on a 1-D grid x (for plotting)."""
        xx = torch.zeros(len(x), self.in_dim, device=x.device)
        xx[:, i] = x
        bsp = self.b_splines(xx)[:, i]                                           # (n, G+k)
        return (self.base_weight[j, i] * F.silu(x) +
                bsp @ (self.spline_coef[j, i] * self.spline_scale[j, i]))


def output_dim(task):
    return 1 if task == "reg" else 3


def finalize(y, task):
    return torch.sigmoid(y.squeeze(-1)) if task == "reg" else y


class LSTMSeq(nn.Module):
    def __init__(self, n_feat, task, hidden=64, dropout=0.3):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, output_dim(task))

    def forward(self, x):                      # x: (batch, L, n_feat)
        h, _ = self.lstm(x)
        return finalize(self.out(self.drop(h[:, -1])), self.task)


class CNNLSTM(nn.Module):
    def __init__(self, task, head="linear", n_ch=19, hidden=64):
        super().__init__()
        self.task = task

        def block(cin, cout, k, pool):
            layers = [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool1d(4))
            return nn.Sequential(*layers)

        self.cnn = nn.Sequential(block(n_ch, 32, 7, True), block(32, 64, 7, True), block(64, 64, 5, False))
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        if head == "kan":
            self.head = nn.Sequential(KANLayer(hidden, 16), KANLayer(16, output_dim(task)))
            self.head_norm = nn.LayerNorm(hidden)   # keeps the LSTM state in the KAN grid range
        else:
            self.head = nn.Linear(hidden, output_dim(task))
            self.head_norm = nn.Identity()

    def forward(self, x):                      # x: (batch, 19, 800)
        z = self.cnn(x).transpose(1, 2)        # (batch, 50, 64)
        h, _ = self.lstm(z)
        return finalize(self.head(self.head_norm(h[:, -1])), self.task)


class KANNet(nn.Module):
    def __init__(self, n_feat, task, width=16, grid=5):
        super().__init__()
        self.task = task
        self.l1 = KANLayer(n_feat, width, grid=grid)
        self.l2 = KANLayer(width, output_dim(task), grid=grid)

    def forward(self, x):
        return finalize(self.l2(self.l1(x.clamp(-3, 3))), self.task)


class MLP(nn.Module):
    def __init__(self, n_feat, task, hidden=(64, 32), dropout=0.3):
        super().__init__()
        self.task = task
        layers, d = [], n_feat
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, output_dim(task)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return finalize(self.net(x), self.task)
