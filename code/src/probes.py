"""The shared probe battery: a fixed set of diagnostic 2-outcome problems on
which EVERY trained model is evaluated, turning each model into (a) a
behavioural fingerprint vector and (b) a set of interpretable effect indices.

Groups
------
ce_gain  risky (x, p; 0) vs sure r*p*x           certainty-equivalent grid
ce_loss  risky (-x, p; 0) vs sure -r*p*x         mirrored for losses
mixed    50-50 mixed gamble (g, .5; -l) vs sure 0   loss-aversion ladder
ladder   common-ratio (Allais) ladders, gains and losses

Convention: option A = the SURE option, option B = the RISKY option, so the
model output P(B) is always P(choose risky).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .problems import problem_arrays

XG = [5, 10, 20, 40, 80, 120, 200]
PG = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
RG = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]

MIXED_G = [10, 20, 40, 80]
MIXED_L = [5, 10, 20, 40, 80]
LA_LADDER_L = 20
LA_LADDER_G = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]

LADDERS = [(30, 40, 0.8), (45, 90, 0.5), (20, 80, 0.25)]
LADDER_Q = [1.0, 0.5, 0.25, 0.1, 0.05]


def build_probe_battery():
    rows, plist = [], []

    def add(group, meta, sure_x, sure_p, risky_x, risky_p):
        rows.append(dict(group=group, **meta))
        plist.append(dict(A=(sure_x, sure_p), B=(risky_x, risky_p),
                          corr=0, pid=f"{group}_{len(plist)}"))

    for sign, group in [(1, "ce_gain"), (-1, "ce_loss")]:
        for x in XG:
            for p in PG:
                for r in RG:
                    c = sign * r * p * x
                    add(group, dict(x=sign * x, p=p, r=r),
                        [c], [1.0], [sign * x, 0.0], [p, 1.0 - p])

    for g in MIXED_G:
        for l in MIXED_L:
            add("mixed", dict(g=g, l=l, ratio=g / l),
                [0.0], [1.0], [g, -l], [0.5, 0.5])
    for g in LA_LADDER_G:
        add("la_ladder", dict(g=g, l=LA_LADDER_L, ratio=g / LA_LADDER_L),
            [0.0], [1.0], [g, -LA_LADDER_L], [0.5, 0.5])

    for sign, group in [(1, "ladder_gain"), (-1, "ladder_loss")]:
        for li, (c, x, p) in enumerate(LADDERS):
            for q in LADDER_Q:
                add(group, dict(ladder=li, q=q, c=sign * c, x=sign * x,
                                p_base=p),
                    [sign * c, 0.0], [q, 1.0 - q],
                    [sign * x, 0.0], [p * q, 1.0 - p * q])

    meta = pd.DataFrame(rows)
    arrs = problem_arrays(plist)
    return arrs, meta


# ---------------------------------------------------------------------------
# behavioural indices from probe predictions
# ---------------------------------------------------------------------------

RARE_P = (0.01, 0.02, 0.05)
MOD_P = (0.2, 0.3, 0.5, 0.7, 0.8)


def fingerprint_indices(pred, meta):
    """pred: np.array over the battery, P(choose risky). Returns dict."""
    m = meta
    out = {}

    ev_eq = np.isclose(m["r"], 1.0)
    rare = m["p"].isin(RARE_P)
    modp = m["p"].isin(MOD_P)
    g, l = m["group"] == "ce_gain", m["group"] == "ce_loss"

    out["rare_gain_over"] = float(pred[g & ev_eq & rare].mean() - 0.5)
    out["rare_loss_over"] = float(0.5 - pred[l & ev_eq & rare].mean())
    out["rare_event_index"] = 0.5 * (out["rare_gain_over"] +
                                     out["rare_loss_over"])

    out["risk_gain"] = float(pred[g & ev_eq & modp].mean())
    out["risk_loss"] = float(pred[l & ev_eq & modp].mean())
    out["reflection_index"] = out["risk_loss"] - out["risk_gain"]

    lad = m["group"] == "ladder_gain"
    q1 = lad & np.isclose(m["q"], 1.0)
    qs = lad & (m["q"] < 1.0)
    base = {li: pred[q1 & (m["ladder"] == li)].mean()
            for li in m.loc[lad, "ladder"].unique()}
    deltas = [pred[qs & (m["ladder"] == li)].mean() - base[li] for li in base]
    out["certainty_index"] = float(np.mean(deltas))

    ll = m["group"] == "la_ladder"
    ratios = m.loc[ll, "ratio"].to_numpy()
    acc = pred[ll]
    order = np.argsort(ratios)
    ratios, acc = ratios[order], acc[order]
    kappa = np.nan
    for i in range(len(acc) - 1):
        if (acc[i] - 0.5) * (acc[i + 1] - 0.5) <= 0 and acc[i + 1] != acc[i]:
            t = (0.5 - acc[i]) / (acc[i + 1] - acc[i])
            kappa = ratios[i] + t * (ratios[i + 1] - ratios[i])
            break
    if np.isnan(kappa):
        kappa = float(ratios[-1] if acc[-1] < 0.5 else ratios[0])
    out["loss_aversion_kappa"] = float(kappa)

    ce = (g | l) & (np.abs(m["r"] - 1.0) >= 0.15)
    ev_says_risky = ((m["r"] < 1.0) & g) | ((m["r"] > 1.0) & l)
    agree = (pred[ce] > 0.5) == ev_says_risky[ce]
    out["ev_consistency"] = float(agree.mean())

    return out


def ce_surface(pred, meta, sign=1):
    """Certainty equivalents CE(x, p)/|x| from the ce grid by interpolating
    the P(risky)=0.5 crossing over the sure-amount ratio r. Returns a
    DataFrame with columns x, p, ce_ratio (relative CE, i.e. w-proxy when
    utility is near-linear)."""
    group = "ce_gain" if sign > 0 else "ce_loss"
    m = meta[meta["group"] == group]
    rows = []
    for (x, p), sub in m.groupby(["x", "p"]):
        sub = sub.sort_values("r")
        r = sub["r"].to_numpy()
        pr = pred[sub.index.to_numpy()]
        # P(risky) should fall as the sure amount rises; find the crossing
        cross = np.nan
        for i in range(len(r) - 1):
            if (pr[i] - 0.5) * (pr[i + 1] - 0.5) <= 0 and pr[i + 1] != pr[i]:
                t = (0.5 - pr[i]) / (pr[i + 1] - pr[i])
                cross = r[i] + t * (r[i + 1] - r[i])
                break
        if np.isnan(cross):
            cross = r[-1] if pr[-1] > 0.5 else r[0]
        rows.append(dict(x=x, p=p, ce_ratio=float(np.clip(cross, r[0], r[-1])),
                         ce=float(np.clip(cross, r[0], r[-1]) * p * abs(x))))
    return pd.DataFrame(rows)
