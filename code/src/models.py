"""Six architectures spanning the capacity/inductive-bias axis.

All expose the same interface:
    fit(train_arrs, val_arrs, sample_weight=None) -> history dict
    predict(arrs) -> np.array of P(choose B) in [0, 1]

LINEAR   logistic regression on flat engineered features
MLP_S    small MLP (2 x 64) on flat features
MLP_L    large MLP (4 x 256) on flat features
SET      permutation-invariant DeepSets encoder over (p, x) outcome slots
CPT_NET  differentiable rank- and sign-dependent CPT (Prelec-I weighting)
         -- a "theory-shaped" model with 6 interpretable parameters
GBT      gradient-boosted trees on flat features (non-neural control)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .problems import flat_features, set_features

DEVICE_DEFAULT = "cpu"


def _to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# torch modules
# ---------------------------------------------------------------------------

class FlatNet(nn.Module):
    def __init__(self, d_in, hidden, depth):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X).squeeze(-1)


class SetNet(nn.Module):
    """DeepSets over outcome slots -> option embeddings -> choice logit."""

    def __init__(self, d_phi=64, d_rho=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, d_phi), nn.ReLU(),
                                 nn.Linear(d_phi, d_phi), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(d_phi, d_rho), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * d_rho + 1, d_rho), nn.ReLU(),
                                  nn.Linear(d_rho, 1))

    def forward(self, slots, mask, corr):
        # slots [N,2,K,2], mask [N,2,K], corr [N,1]
        h = self.phi(slots) * mask.unsqueeze(-1)
        emb = self.rho(h.sum(dim=2))                 # [N,2,d_rho]
        z = torch.cat([emb[:, 0], emb[:, 1], corr], dim=-1)
        return self.head(z).squeeze(-1)


class CPTNet(nn.Module):
    """Differentiable CPT; torch twin of problems.cpt_value (unit-tested)."""

    BOUNDS = dict(alpha=(0.2, 2.0), beta=(0.2, 2.0), lam=(0.2, 8.0),
                  gamma_pos=(0.2, 2.5), gamma_neg=(0.2, 2.5), tau=(0.1, 30.0))

    def __init__(self):
        super().__init__()
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.zeros(()))
                                     for k in self.BOUNDS})

    def params(self):
        out = {}
        for k, (lo, hi) in self.BOUNDS.items():
            out[k] = lo + (hi - lo) * torch.sigmoid(self.raw[k])
        return out

    @staticmethod
    def _prelec(p, gamma):
        p = torch.clamp(p, 1e-12, 1.0)
        return torch.exp(-torch.pow(-torch.log(p), gamma))

    def _value(self, x, p, th):
        xs = x  # already scaled by caller
        v = torch.where(xs >= 0,
                        torch.pow(torch.abs(xs) + 1e-12, th["alpha"]),
                        -th["lam"] * torch.pow(torch.abs(xs) + 1e-12,
                                               th["beta"]))
        cum_top = torch.cumsum(p, dim=-1)
        prev_top = cum_top - p
        wg = self._prelec(cum_top, th["gamma_pos"]) - \
            self._prelec(prev_top, th["gamma_pos"])
        cum_bot = torch.flip(torch.cumsum(torch.flip(p, [-1]), dim=-1), [-1])
        prev_bot = cum_bot - p
        wl = self._prelec(cum_bot, th["gamma_neg"]) - \
            self._prelec(prev_bot, th["gamma_neg"])
        pi = torch.where(xs >= 0, wg, wl) * (p > 0)
        return (pi * v).sum(-1)

    def forward(self, slots, mask, corr):
        th = self.params()
        va = self._value(slots[:, 0, :, 1], slots[:, 0, :, 0], th)
        vb = self._value(slots[:, 1, :, 1], slots[:, 1, :, 0], th)
        return th["tau"] * (vb - va)


class TransformerNet(nn.Module):
    """Self-attention over outcome slots, then a choice head.

    Included because every other architecture here either flattens the
    gamble into a fixed vector or pools its outcomes without letting them
    interact. Attention lets an outcome's contribution depend on the other
    outcomes in the same option, which is the inductive bias most modern
    sequence models bring and the one this battery can actually distinguish.
    """

    def __init__(self, d=64, heads=4, layers=2, ff=128):
        super().__init__()
        self.proj = nn.Linear(2, d)
        self.opt_emb = nn.Parameter(torch.zeros(2, 1, d))
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff, dropout=0.0,
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(),
                                  nn.Linear(d, 1))

    def forward(self, slots, mask, corr):
        # slots [N,2,K,2] (p, x); mask [N,2,K]
        N, O, K, _ = slots.shape
        h = self.proj(slots) + self.opt_emb.unsqueeze(0)      # [N,2,K,d]
        h = h.reshape(N * O, K, -1)
        pad = ~mask.reshape(N * O, K).bool()
        # a padded-out option would make softmax undefined; keep one slot live
        pad = torch.where(pad.all(-1, keepdim=True),
                          torch.zeros_like(pad), pad)
        h = self.enc(h, src_key_padding_mask=pad)
        h = (h * mask.reshape(N * O, K, 1)).sum(1) \
            / mask.reshape(N * O, K, 1).sum(1).clamp(min=1.0)
        h = h.reshape(N, O, -1)
        return self.head(torch.cat([h[:, 0], h[:, 1], corr], -1)).squeeze(-1)


class ResMLP(nn.Module):
    """Wide residual MLP -- the scale control. Same input as the other flat
    models, roughly an order of magnitude more parameters than MLP_L, so
    'architecture barely matters' can be tested across a real size range
    rather than only among small models."""

    def __init__(self, d_in, hidden=512, blocks=4):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                          nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(blocks)])
        self.out = nn.Linear(hidden, 1)
        # Zero-init the last layer of each residual branch. Without it the
        # four branches add O(1) noise on top of the input projection before
        # any training, the activations entering `out` are several times
        # larger than the target scale, and the model never recovers -- it
        # scored worse than predicting the mean. Starting each branch at the
        # identity is the standard fix (Fixup / zero-init residual).
        for b in self.blocks:
            nn.init.zeros_(b[-1].weight)
            nn.init.zeros_(b[-1].bias)

    def forward(self, X):
        h = self.inp(X)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h).squeeze(-1)


# ---------------------------------------------------------------------------
# unified wrappers
# ---------------------------------------------------------------------------

# Per-architecture learning rate, chosen on validation loss in a calibration
# sweep over {3e-2, 1e-2, 3e-3, 1e-3, 3e-4}, scored as val MSE normalised by
# each arm's variance and averaged over a large arm (c13k_exp) and a small one
# (cpc18_exp). See scripts/09_lr_calibration.py, which regenerates the table.
#
# This matters for the headline comparison, not just for speed. A single
# global rate flatters whichever architectures happen to suit it: at the old
# shared 3e-3, LINEAR, MLP_S and SET were all under-trained by a factor of ten
# in learning rate, and RESMLP collapsed to a constant. Differences that come
# from tuning would otherwise be scored as differences between architectures.
ARCH_LR = {
    "LINEAR": 3e-3, "MLP_S": 3e-2, "MLP_L": 1e-3, "RESMLP": 3e-4,
    "SET": 1e-2, "TRANSFORMER": 1e-3, "CPT_NET": 3e-2,
}


class TorchModel:
    kind = "flat"  # or "set"

    def __init__(self, arch, seed=0, device=DEVICE_DEFAULT, lr=None,
                 max_epochs=600, batch=512, patience=40, weight_decay=1e-5):
        self.arch, self.seed, self.device = arch, seed, device
        self.lr = ARCH_LR.get(arch, 3e-3) if lr is None else lr
        self.max_epochs = max_epochs
        self.batch, self.patience = batch, patience
        self.weight_decay = weight_decay
        torch.manual_seed(seed)
        np.random.seed(seed)
        if arch == "LINEAR":
            self.kind = "flat"; self.net = FlatNet(55, 1, 0)
        elif arch == "MLP_S":
            self.kind = "flat"; self.net = FlatNet(55, 64, 2)
        elif arch == "MLP_L":
            self.kind = "flat"; self.net = FlatNet(55, 256, 4)
        elif arch == "RESMLP":
            self.kind = "flat"; self.net = ResMLP(55)
        elif arch == "SET":
            self.kind = "set"; self.net = SetNet()
        elif arch == "TRANSFORMER":
            self.kind = "set"; self.net = TransformerNet()
        elif arch == "CPT_NET":
            self.kind = "set"; self.net = CPTNet()
            self.weight_decay = 0.0   # six interpretable params; do not shrink
        else:
            raise ValueError(arch)
        self.net.to(device)

    # ---- data plumbing ----------------------------------------------------
    def _inputs(self, arrs):
        if self.kind == "flat":
            return (_to_t(flat_features(arrs), self.device),)
        slots, mask, corr = set_features(arrs)
        return (_to_t(slots, self.device), _to_t(mask, self.device),
                _to_t(corr, self.device))

    def _forward(self, inputs):
        return torch.sigmoid(self.net(*inputs))

    # ---- training ---------------------------------------------------------
    def fit(self, train_arrs, val_arrs, sample_weight=None):
        Xtr = self._inputs(train_arrs)
        ytr = _to_t(train_arrs["y"], self.device)
        wtr = (_to_t(sample_weight, self.device)
               if sample_weight is not None else torch.ones_like(ytr))
        Xva = self._inputs(val_arrs)
        yva = _to_t(val_arrs["y"], self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        n = len(ytr)
        batch = min(self.batch, n)
        best, best_state, bad, hist = np.inf, None, 0, []
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            for s in range(0, n, batch):
                sel = perm[s:s + batch]
                xb = tuple(t[sel] for t in Xtr)
                pred = self._forward(xb)
                loss = (wtr[sel] * (pred - ytr[sel]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(((self._forward(Xva) - yva) ** 2).mean())
            hist.append(v)
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().clone()
                              for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return dict(val_mse=best, epochs=len(hist))

    def predict(self, arrs, batch=4096):
        self.net.eval()
        X = self._inputs(arrs)
        out = []
        with torch.no_grad():
            for s in range(0, len(arrs["y"]), batch):
                xb = tuple(t[s:s + batch] for t in X)
                out.append(self._forward(xb).cpu().numpy())
        return np.concatenate(out)

    def extra_state(self):
        if self.arch == "CPT_NET":
            with torch.no_grad():
                return {k: float(v) for k, v in self.net.params().items()}
        return {}


class GBTModel:
    def __init__(self, seed=0, **kw):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.arch = "GBT"
        self.reg = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_depth=None,
            max_leaf_nodes=31, l2_regularization=1e-3, random_state=seed,
            early_stopping=False)

    def fit(self, train_arrs, val_arrs, sample_weight=None):
        X = flat_features(train_arrs)
        self.reg.fit(X, train_arrs["y"], sample_weight=sample_weight)
        v = float(np.mean((self.predict(val_arrs) - val_arrs["y"]) ** 2))
        return dict(val_mse=v, epochs=self.reg.n_iter_)

    def predict(self, arrs):
        return np.clip(self.reg.predict(flat_features(arrs)), 0.0, 1.0)

    def extra_state(self):
        return {}


ARCHS = ["LINEAR", "MLP_S", "MLP_L", "RESMLP", "SET", "TRANSFORMER",
         "CPT_NET", "GBT"]


def make_model(arch, seed=0, device=DEVICE_DEFAULT, **kw):
    if arch == "GBT":
        return GBTModel(seed=seed)
    return TorchModel(arch, seed=seed, device=device, **kw)
