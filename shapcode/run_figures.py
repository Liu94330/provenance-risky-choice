"""The attribution figure, in the form attribution analysis is normally read.

The decomposition in 11_attribution.py collapses each model to eleven numbers,
which is what the variance analysis needs and also throws away the thing
attribution is good at: for `EV difference` the per-probe spread is over three
times the mean, so a single number says "unused" about a group the model leans
on hard in one region and against in another.

This stage keeps the per-probe values and draws them the way the method is
usually presented -- a beeswarm over every probe, a dependence panel showing
the functional form, the conditional map across probability space, and the
cross-check against the CPT readout.

Self-contained: trains four representative models (one per arm), computes
exact per-sample grouped Shapley, draws, and exits. About a minute.

Usage: python shapcode/run_figures.py [--arch MLP_L]
"""
import _boot  # noqa
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from src.data import load_arm, split_indices, take
from shap_figures import fig_attribution_rich
from src.models import make_model
from src.paths import FIGURES, TABLES
from src.probes import build_probe_battery
from src.problems import flat_features
from shap_readout import FEATURE_GROUPS, grouped_shapley

MAIN = ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="MLP_L",
                    help="representative flat-input architecture")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-probes", type=int, default=500)
    args = ap.parse_args()

    probe_arrs, probe_meta = build_probe_battery()
    # certainty-equivalent probes only: they carry the p and r that make the
    # dependence and conditional panels interpretable
    sel = probe_meta["group"].isin(["ce_gain", "ce_loss"]).to_numpy()
    rng = np.random.default_rng(args.seed)
    idx = np.where(sel)[0]
    if len(idx) > args.n_probes:
        idx = np.sort(rng.choice(idx, args.n_probes, replace=False))
    meta = probe_meta.iloc[idx].reset_index(drop=True)
    Xp = flat_features(probe_arrs)[idx]

    rows = []
    for arm in MAIN:
        arrs = load_arm(arm)
        tr_i, va_i, _ = split_indices(len(arrs["y"]), seed=0)
        tr, va = take(arrs, tr_i), take(arrs, va_i)
        model = make_model(args.arch, seed=args.seed)
        model.fit(tr, va)
        sv = grouped_shapley(model, Xp, flat_features(tr).mean(0),
                             per_sample=True)
        for g, vals in sv.items():
            rows.append(pd.DataFrame(dict(
                dataset=arm, feature_group=g, shap=vals,
                p=meta["p"].to_numpy(), r=meta["r"].to_numpy(),
                x=meta["x"].to_numpy(),
                domain=np.where(meta["group"].to_numpy() == "ce_gain",
                                "gains", "losses"))))
        print(f"  {arm}: {len(sv)} groups x {len(Xp)} probes")

    per = pd.concat(rows, ignore_index=True)
    # dEV of each probe, so the dependence panel plots against a real quantity
    ev_col = FEATURE_GROUPS["EV difference"][0]
    dev = pd.DataFrame(dict(p=meta["p"].to_numpy(),
                            r=meta["r"].to_numpy(),
                            dEV=Xp[:, ev_col]))
    per = per.merge(dev.drop_duplicates(["p", "r"]), on=["p", "r"],
                    how="left")
    os.makedirs(TABLES, exist_ok=True)
    per.to_csv(os.path.join(TABLES, "attribution_per_probe.csv"), index=False)

    cross = os.path.join(TABLES, "attribution_vs_cpt.csv")
    fp = os.path.join(TABLES, "attribution_fingerprints.csv")
    tc = os.path.join(TABLES, "theory_coords.csv")
    fig_attribution_rich(
        per,
        pd.read_csv(cross) if os.path.exists(cross) else None,
        pd.read_csv(fp) if os.path.exists(fp) else None,
        pd.read_csv(tc) if os.path.exists(tc) else None,
        FIGURES, arch=args.arch)
    print(f"\nwrote {TABLES}/attribution_per_probe.csv")
    print(f"wrote {FIGURES}/fig_attribution_rich.pdf")
