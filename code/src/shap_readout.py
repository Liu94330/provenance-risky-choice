"""A fourth readout: which parts of a gamble a trained network actually uses.

The other three readouts all assume a form. CPT coordinates assume the network
implements cumulative prospect theory; the behavioural indices assume the
effects worth measuring are the ones the literature already names; symbolic
distillation assumes a closed form exists in the operator set. Attribution
assumes none of that -- it asks only how much the prediction moves when a part
of the input is withheld -- so it is the one readout that cannot inherit a
mis-specified theory.

Grouped exactly, not sampled
----------------------------
Off-the-shelf KernelSHAP approximates Shapley values over all 55 raw features,
which is both slow and unreadable: `A_x[7]` means nothing on its own. Here the
features are partitioned into semantic groups and the Shapley value of each
GROUP is computed exactly by enumerating all 2^G coalitions. With G = 11 that
is 2,048 evaluations -- cheap in batch, and exact rather than sampled, so two
architectures can be compared without an approximation-noise caveat.

Withheld groups are set to the training-set mean, i.e. the standard baseline
("what would the model predict if it did not know this?").

Scope: only the architectures that consume the flat 55-d encoding
(LINEAR, MLP_S, MLP_L, RESMLP, GBT). SET, TRANSFORMER and CPT_NET read
per-outcome slots instead, so the same feature groups do not exist for them.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import torch

from .problems import flat_features

# Indices follow problems.flat_features: A_p[10] A_x[10] | B_p[10] B_x[10] |
# summary_A[6] | summary_B[6] | diffs[2] | corr, where each summary is
# (EV, SD, max, min, P(loss), n_outcomes).
FEATURE_GROUPS = {
    "A outcomes": list(range(0, 20)),
    "B outcomes": list(range(20, 40)),
    "A summary": list(range(40, 46)),
    "EV of B": [46],
    "SD of B": [47],
    "extremes of B": [48, 49],
    "P(loss) of B": [50],
    "n outcomes B": [51],
    "EV difference": [52],
    "SD difference": [53],
    "correlation": [54],
}

FLAT_ARCHS = ("LINEAR", "MLP_S", "MLP_L", "RESMLP", "GBT")


def predict_flat(model, X, batch=8192):
    """Run a flat-input model on raw 55-d features, bypassing the usual
    arrs -> features step so groups can be withheld."""
    if getattr(model, "arch", None) == "GBT":
        return np.asarray(model.reg.predict(X), dtype=float)
    if getattr(model, "kind", None) != "flat":
        raise ValueError(f"{model.arch} does not consume flat features")
    model.net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            t = torch.as_tensor(X[i:i + batch], dtype=torch.float32,
                                device=model.device)
            out.append(torch.sigmoid(model.net(t)).cpu().numpy())
    return np.concatenate(out).astype(float)


def grouped_shapley(model, X, baseline, groups=None):
    """Exact Shapley value of each feature group, averaged over rows of X.

    Returns {group: value}. Values sum to mean(f(X)) - f(baseline), which the
    caller can check -- efficiency is the property that makes these comparable
    across architectures.
    """
    groups = groups or FEATURE_GROUPS
    keys = list(groups)
    G = len(keys)

    # v(S) for every coalition, evaluated once and cached
    v = {}
    for r in range(G + 1):
        for S in combinations(range(G), r):
            Z = np.tile(baseline, (len(X), 1)).astype(np.float32)
            for gi in S:
                cols = groups[keys[gi]]
                Z[:, cols] = X[:, cols]
            v[S] = float(predict_flat(model, Z).mean())

    shap = {}
    for gi, k in enumerate(keys):
        rest = [j for j in range(G) if j != gi]
        total = 0.0
        for r in range(G):
            w = factorial(r) * factorial(G - r - 1) / factorial(G)
            for S in combinations(rest, r):
                total += w * (v[tuple(sorted(S + (gi,)))] - v[S])
        shap[k] = total
    return shap


def attribution_fingerprint(model, arrs, train_arrs, groups=None,
                            n_rows=256, seed=0):
    """Normalised group attributions for one trained model.

    Absolute attributions scale with how much a model's predictions vary at
    all, which differs between arms; dividing by the total |attribution| makes
    the profile a statement about WHICH features are used rather than how
    strongly the model responds, which is what the comparison needs.
    """
    rng = np.random.default_rng(seed)
    X = flat_features(arrs)
    if len(X) > n_rows:
        X = X[rng.choice(len(X), n_rows, replace=False)]
    baseline = flat_features(train_arrs).mean(0)
    raw = grouped_shapley(model, X, baseline, groups)
    tot = sum(abs(x) for x in raw.values()) or 1.0
    return {k: val / tot for k, val in raw.items()}, raw
