"""The grouped Shapley implementation must satisfy the Shapley axioms.

This is the correctness guarantee the whole attribution readout rests on. The
values are computed by enumerating coalitions rather than by calling a library,
so nothing external checks them -- if the weighting or the coalition
enumeration were wrong the numbers would still look plausible, and every
conclusion drawn from them would be quietly wrong.

Four checks, on a model whose exact behaviour is known analytically plus a real
trained network:

  efficiency   values sum to f(x) - f(baseline), per row, not just on average
  symmetry     two groups the model treats identically get equal values
  null player  a group the model ignores gets exactly zero
  additivity   values for a sum of two models equal the sum of their values

Usage: python shapcode/test_shapley.py
"""
import _boot  # noqa

import numpy as np

from shap_readout import grouped_shapley, predict_flat  # noqa: F401


class LinearProbe:
    """f(x) = sigmoid-free linear read of chosen columns, so every Shapley
    value is known in closed form: a linear model's group value is exactly the
    sum of its weights times (x - baseline) over that group's columns."""

    arch = "TEST"
    kind = "flat"

    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=float)

    def predict_raw(self, X):
        return X @ self.w


def _shap_linear(model, X, baseline, groups):
    """grouped_shapley expects the project's model interface; this adapts a
    plain callable by monkey-patching the one function it calls."""
    import shap_readout as sr
    orig = sr.predict_flat
    sr.predict_flat = lambda m, Z, batch=8192: m.predict_raw(Z)
    try:
        # float64 so the axioms are checked against the algebra, not against
        # the float32 the torch path uses
        return sr.grouped_shapley(model, X, baseline, groups,
                                  per_sample=True, dtype=np.float64)
    finally:
        sr.predict_flat = orig


def main():
    rng = np.random.default_rng(0)
    D = 8
    groups = {"g0": [0, 1], "g1": [2, 3], "g2": [4, 5], "g3": [6, 7]}
    X = rng.normal(size=(16, D))
    baseline = np.zeros(D)

    # --- efficiency + closed form on a linear model -----------------------
    w = np.array([1.0, -2.0, 0.5, 0.5, 3.0, 0.0, 0.0, 0.0])
    m = LinearProbe(w)
    sv = _shap_linear(m, X, baseline, groups)
    M = np.stack([sv[k] for k in groups])
    fx = m.predict_raw(X)
    fb = float(m.predict_raw(baseline[None, :])[0])
    eff = np.abs(M.sum(0) - (fx - fb)).max()
    print(f"efficiency (per row, linear model): max error {eff:.2e}")
    assert eff < 1e-10, "efficiency violated"

    for gi, (k, cols) in enumerate(groups.items()):
        exact = X[:, cols] @ w[cols]
        err = np.abs(sv[k] - exact).max()
        print(f"  closed form {k}: max error {err:.2e}")
        assert err < 1e-10, f"{k} does not match the analytic value"

    # --- null player: g3 has zero weight ----------------------------------
    null = np.abs(sv["g3"]).max()
    print(f"null player (g3, zero weight): max |value| {null:.2e}")
    assert null < 1e-12, "a group the model ignores got a non-zero value"

    # --- symmetry: g1's two columns carry equal weight, and a group with the
    # same total weight must get the same value on symmetric input ---------
    w2 = np.array([2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    m2 = LinearProbe(w2)
    Xs = X.copy()
    Xs[:, 2] = Xs[:, 0]                     # make g0 and g1 identical inputs
    sv2 = _shap_linear(m2, Xs, baseline, groups)
    sym = np.abs(sv2["g0"] - sv2["g1"]).max()
    print(f"symmetry (identical groups): max difference {sym:.2e}")
    assert sym < 1e-10, "symmetric groups received different values"

    # --- additivity: shap(f+g) == shap(f) + shap(g) ------------------------
    wa = rng.normal(size=D)
    wb = rng.normal(size=D)
    sa = _shap_linear(LinearProbe(wa), X, baseline, groups)
    sb = _shap_linear(LinearProbe(wb), X, baseline, groups)
    sab = _shap_linear(LinearProbe(wa + wb), X, baseline, groups)
    add = max(np.abs(sab[k] - (sa[k] + sb[k])).max() for k in groups)
    print(f"additivity: max error {add:.2e}")
    assert add < 1e-10, "additivity violated"

    # --- efficiency on a real trained network -----------------------------
    from src.data import load_arm, split_indices, take
    from src.models import make_model
    from src.probes import build_probe_battery
    from src.problems import flat_features

    arrs = load_arm("cpc18_exp")
    tr_i, va_i, _ = split_indices(len(arrs["y"]), seed=0)
    tr, va = take(arrs, tr_i), take(arrs, va_i)
    net = make_model("MLP_S", seed=0, max_epochs=60, patience=15)
    net.fit(tr, va)
    probe, _ = build_probe_battery()
    Xp = flat_features(probe)[:64]
    bl = flat_features(tr).mean(0)
    sv3 = grouped_shapley(net, Xp, bl, per_sample=True)
    M3 = np.stack(list(sv3.values()))
    err3 = np.abs(M3.sum(0)
                  - (predict_flat(net, Xp)
                     - predict_flat(net, bl[None, :])[0])).max()
    print(f"efficiency (per row, trained MLP_S over 11 groups): "
          f"max error {err3:.2e}")
    assert err3 < 1e-6, "efficiency violated on the real model"

    print("\nShapley axiom test passed.")


if __name__ == "__main__":
    main()
