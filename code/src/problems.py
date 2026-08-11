"""Canonical representation of a two-option risky-choice problem.

Every dataset arm (choices13k, CPC18, synthetic) is converted into the same
padded-array format so that every architecture sees identical inputs and the
probe battery can be evaluated on any trained model.

Conventions
-----------
* Option A / Option B, target = P(choose B).
* Up to MAX_OUT outcomes per option, sorted by outcome DESCENDING, padded
  with (x=0, p=0); a slot is "real" iff p > 0.
* All outcomes are divided by SCALE (fixed across datasets) before entering
  any model, so cross-dataset transfer is well defined.
"""

from __future__ import annotations

from math import comb

import numpy as np

MAX_OUT = 10   # CPC18 allows up to 10 outcomes per option; c13k max is 9
SCALE = 100.0  # fixed global payoff scale (both datasets span [-50, 256])


# ---------------------------------------------------------------------------
# construction helpers
# ---------------------------------------------------------------------------

def pad_option(outcomes, probs, max_out: int = MAX_OUT):
    """Sort (outcome, prob) pairs by outcome descending, drop p==0, pad."""
    outcomes = np.asarray(outcomes, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    keep = probs > 1e-12
    outcomes, probs = outcomes[keep], probs[keep]
    # merge duplicate outcomes (can occur after lottery expansion)
    uniq = {}
    for x, p in zip(outcomes, probs):
        uniq[x] = uniq.get(x, 0.0) + p
    outcomes = np.array(sorted(uniq.keys(), reverse=True), dtype=np.float64)
    probs = np.array([uniq[x] for x in outcomes], dtype=np.float64)
    s = probs.sum()
    if not (0.98 < s < 1.02):
        raise ValueError(f"option probabilities sum to {s:.4f}")
    probs = probs / s
    n = len(outcomes)
    if n > max_out:
        raise ValueError(f"too many outcomes: {n} > {max_out}")
    xo = np.zeros(max_out)
    po = np.zeros(max_out)
    xo[:n], po[:n] = outcomes, probs
    return xo, po


def problem_arrays(problem_list):
    """Stack a list of dict problems into batched arrays.

    Each problem dict: {"A": (outcomes, probs), "B": (outcomes, probs),
                        "corr": int, "y": float or nan, "sem2": float,
                        "n": float, "pid": str}
    """
    N = len(problem_list)
    out = {
        "A_x": np.zeros((N, MAX_OUT)), "A_p": np.zeros((N, MAX_OUT)),
        "B_x": np.zeros((N, MAX_OUT)), "B_p": np.zeros((N, MAX_OUT)),
        "corr": np.zeros(N), "y": np.full(N, np.nan),
        "sem2": np.full(N, np.nan), "n": np.full(N, np.nan),
    }
    pids = []
    for i, pr in enumerate(problem_list):
        out["A_x"][i], out["A_p"][i] = pad_option(*pr["A"])
        out["B_x"][i], out["B_p"][i] = pad_option(*pr["B"])
        out["corr"][i] = pr.get("corr", 0)
        out["y"][i] = pr.get("y", np.nan)
        out["sem2"][i] = pr.get("sem2", np.nan)
        out["n"][i] = pr.get("n", np.nan)
        pids.append(str(pr.get("pid", i)))
    out["pid"] = np.array(pids)
    return out


# ---------------------------------------------------------------------------
# feature engineering (for LINEAR / MLP / GBT flat inputs)
# ---------------------------------------------------------------------------

def _option_summary(x, p):
    """Summary statistics of one padded option batch [N, K] -> [N, 6]."""
    xs = x / SCALE
    ev = (xs * p).sum(-1)
    var = (p * (xs - ev[..., None]) ** 2).sum(-1)
    sd = np.sqrt(np.maximum(var, 0.0))
    mask = p > 0
    big = np.where(mask, xs, -np.inf).max(-1)
    small = np.where(mask, xs, np.inf).min(-1)
    ploss = (p * (xs < 0)).sum(-1)
    nout = mask.sum(-1) / MAX_OUT
    return np.stack([ev, sd, big, small, ploss, nout], axis=-1)


def flat_features(arrs):
    """Flat input for LINEAR/MLP/GBT: padded slots + summaries + diffs + corr.

    Returns [N, D] with D = 2*MAX_OUT*2 + 6*2 + 2 + 1 = 55.
    """
    A = np.concatenate([arrs["A_p"], arrs["A_x"] / SCALE], axis=-1)
    B = np.concatenate([arrs["B_p"], arrs["B_x"] / SCALE], axis=-1)
    sa = _option_summary(arrs["A_x"], arrs["A_p"])
    sb = _option_summary(arrs["B_x"], arrs["B_p"])
    diffs = np.stack([sb[:, 0] - sa[:, 0], sb[:, 1] - sa[:, 1]], axis=-1)
    corr = arrs["corr"][:, None]
    return np.concatenate([A, B, sa, sb, diffs, corr], axis=-1).astype(np.float32)


def set_features(arrs):
    """Inputs for the permutation-invariant SET model.

    Returns slots [N, 2, K, 2] (p, x/SCALE), mask [N, 2, K], corr [N, 1].
    """
    slots = np.stack(
        [np.stack([arrs["A_p"], arrs["A_x"] / SCALE], axis=-1),
         np.stack([arrs["B_p"], arrs["B_x"] / SCALE], axis=-1)], axis=1)
    mask = np.stack([arrs["A_p"] > 0, arrs["B_p"] > 0], axis=1)
    return (slots.astype(np.float32), mask.astype(np.float32),
            arrs["corr"][:, None].astype(np.float32))


# ---------------------------------------------------------------------------
# generic CPT valuation (numpy reference; torch twin lives in models.CPTNet)
# ---------------------------------------------------------------------------

def expand_lottery(H, pH, L, shape, num):
    """Expand a CPC-format option into an explicit [(outcome, prob)] list.

    CPC15/CPC18 store each option compactly as (H, pH, L, LotShape, LotNum):
    with probability pH the payoff is a lottery centred on H whose shape is
    given by LotShape/LotNum, otherwise it is L. Only the CPC18 estimation set
    ships with the distributions already expanded (av1..av10 / bv1..bp10), so
    the competition problems and all of CPC15 need this expansion.

    The rules below were reverse-engineered from the 420 option distributions
    that DO ship expanded, and reproduce every one of them exactly (the single
    apparent mismatch is the published CSV rounding 3.90625e-05 to 3.91e-05).
    tests/test_lottery_expansion.py re-checks this on every run.

      '-'       degenerate: H with pH
      'Symm'    H + (i - (num-1)/2) for i in 0..num-1,
                probabilities pH * Binomial(num-1, 1/2)
      'R-skew'  H - (num-1) + (2^i - 2) for i in 1..num,
                probabilities pH * 2^-i with the last mass repeated
      'L-skew'  mirror image of R-skew about H
    """
    num = int(num)
    pH = float(pH)
    if shape == "-" or num == 1:
        hi = [(float(H), pH)]
    elif shape == "Symm":
        c = [comb(num - 1, i) / 2 ** (num - 1) for i in range(num)]
        hi = [(float(H) - (num - 1) / 2 + i, pH * c[i]) for i in range(num)]
    elif shape in ("R-skew", "L-skew"):
        sgn = -1 if shape == "R-skew" else 1
        pr = [2.0 ** -(i + 1) for i in range(num)]
        if num > 1:
            pr[-1] = pr[-2]
        else:
            pr[-1] = 1.0
        hi = [(float(H) + sgn * ((num - 1) - (2 ** (i + 1) - 2)), pH * pr[i])
              for i in range(num)]
    else:
        raise ValueError(f"unknown LotShape {shape!r}")

    parts = hi + ([(float(L), 1.0 - pH)] if pH < 1.0 else [])
    agg = {}
    for v, p in parts:
        if p > 1e-12:
            agg[round(v, 6)] = agg.get(round(v, 6), 0.0) + p
    xs = sorted(agg)
    return [x for x in xs], [agg[x] for x in xs]


def prelec_w(p, gamma):
    """Prelec-I probability weighting, w(p) = exp(-(-ln p)^gamma)."""
    p = np.clip(p, 1e-12, 1.0)
    return np.exp(-np.power(-np.log(p), gamma))


def cpt_value(x, p, alpha, beta, lam, gamma_pos, gamma_neg):
    """Rank- and sign-dependent CPT value of padded option batches.

    x, p: [..., K] padded (p=0 slots ignored), x in RAW units (scaled inside).
    Outcomes must be sorted descending (pad_option guarantees this).
    """
    xs = x / SCALE
    v = np.where(xs >= 0, np.power(np.abs(xs), alpha),
                 -lam * np.power(np.abs(xs), beta))
    # gains: decumulative from the top (outcomes sorted descending)
    cum_from_top = np.cumsum(p, axis=-1)
    prev_top = cum_from_top - p
    wg = prelec_w(cum_from_top, gamma_pos) - prelec_w(prev_top, gamma_pos)
    # losses: cumulative from the bottom
    cum_from_bot = np.cumsum(p[..., ::-1], axis=-1)[..., ::-1]
    prev_bot = cum_from_bot - p
    wl = prelec_w(cum_from_bot, gamma_neg) - prelec_w(prev_bot, gamma_neg)
    pi = np.where(xs >= 0, wg, wl) * (p > 0)
    return (pi * v).sum(-1)


def cpt_rate(arrs, theta):
    """P(choose B) under CPT with params theta=(alpha,beta,lam,gp,gn,tau)."""
    a, b, lam, gp, gn, tau = theta
    va = cpt_value(arrs["A_x"], arrs["A_p"], a, b, lam, gp, gn)
    vb = cpt_value(arrs["B_x"], arrs["B_p"], a, b, lam, gp, gn)
    return 1.0 / (1.0 + np.exp(-tau * (vb - va)))
