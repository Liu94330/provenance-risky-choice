"""Fit an interpretable CPT (power value + Prelec-I weighting, sign-dependent)
to a model's predictions on the probe battery. Each trained network thereby
becomes a point in 6-D theory space: (alpha, beta, lambda, gamma+, gamma-, tau).

This is the readout that lets us ask "what theory did the network learn?"
without assuming the network IS a CPT agent: the fit is to its behaviour on
the shared battery, exactly as an experimental economist would characterise a
human participant.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .problems import cpt_rate

PARAM_NAMES = ["alpha", "beta", "lam", "gamma_pos", "gamma_neg", "tau"]
BOUNDS = [(0.2, 2.0), (0.2, 2.0), (0.2, 8.0),
          (0.2, 2.5), (0.2, 2.5), (0.1, 30.0)]
X0 = np.array([0.9, 0.9, 1.5, 0.8, 0.8, 5.0])

# A network whose predictions barely move across the whole probe battery has
# no risk attitude to read out: every CPT parameter is then unconstrained and
# the fit returns whatever the optimiser touched last. Such runs are flagged
# not-identifiable and excluded from the theory-space analyses. The threshold
# is on the SD of the run's probe predictions; 0.02 corresponds to a total
# response range of roughly 0.1 choice-probability across 1,742 probes, i.e.
# below the aggregation noise of every arm. Sensitivity to this cut is
# reported in scripts/06_analyze.py.
IDENT_MIN_PROBE_SD = 0.02


def fit_cpt_to_predictions(probe_arrs, pred, n_restarts=8, seed=0,
                           ident_min_sd=IDENT_MIN_PROBE_SD):
    """Least-squares CPT fit to a probe-prediction vector.

    Returns the 6 fitted parameters plus fit diagnostics.

    Two different validity conditions are reported, because the two kinds of
    readout in this paper fail differently:

    ``identifiable``          probe predictions vary enough to describe any
                              behaviour at all. This is what the behavioural
                              indices need -- they are read straight off the
                              predictions and never touch this fit.
    ``theory_identifiable``   additionally, no fitted parameter sits on its
                              box constraint. A parameter pinned to a bound is
                              the optimiser reporting "at least this much",
                              not an estimate, so the CPT coordinates only
                              mean something when this is True.

    ``r2`` is normalised by the spread of the predictions and is therefore
    NaN when the run is not identifiable: dividing by a near-zero variance
    produces arbitrarily large negative numbers that are an artefact of the
    normalisation, not a bad fit."""
    rng = np.random.default_rng(seed)
    y = np.asarray(pred, dtype=np.float64)

    def loss(theta):
        return float(np.mean((cpt_rate(probe_arrs, theta) - y) ** 2))

    starts = [X0]
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    for _ in range(n_restarts - 1):
        starts.append(lo + rng.random(6) * (hi - lo))

    best, best_theta = np.inf, X0
    for s in starts:
        res = minimize(loss, s, method="L-BFGS-B", bounds=BOUNDS,
                       options=dict(maxiter=400))
        if res.fun < best:
            best, best_theta = res.fun, res.x
    probe_sd = float(np.std(y))
    identifiable = bool(probe_sd >= ident_min_sd)
    at_bound = bool(np.any(np.isclose(best_theta, lo, atol=1e-3)
                           | np.isclose(best_theta, hi, atol=1e-3)))
    r2 = 1.0 - best / max(np.var(y), 1e-12)
    out = {k: float(v) for k, v in zip(PARAM_NAMES, best_theta)}
    out.update(cpt_fit_mse=float(best), cpt_fit_rmse=float(np.sqrt(best)),
               cpt_fit_r2=float(r2) if identifiable else float("nan"),
               cpt_fit_r2_raw=float(r2), probe_sd=probe_sd,
               probe_range=float(np.ptp(y)),
               identifiable=identifiable,
               at_bound=at_bound,
               theory_identifiable=bool(identifiable and not at_bound))
    return out
