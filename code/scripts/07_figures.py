"""Generate the paper figures from saved tables and run artifacts."""
import _boot  # noqa
import glob
import json
import os

import numpy as np
import pandas as pd

from src.data import TRUE_THETA
from src.figures import (fig1_theory_space, fig2_de_flip, fig3_transfer,
                         fig4_decomposition)
from src.probes import build_probe_battery
from src.theoryfit import BOUNDS, PARAM_NAMES
from src.train import RESULTS

MAIN = ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]

if __name__ == "__main__":
    figdir = os.path.join(RESULTS, "figures")
    tabdir = os.path.join(RESULTS, "tables")

    coords = pd.read_csv(os.path.join(tabdir, "theory_coords.csv"))
    base = coords[coords["subsample_n"].isna() & coords["reweight_to"].isna()]
    bounds = {p: b for p, b in zip(PARAM_NAMES, BOUNDS)}
    fig1_theory_space(base[base["dataset"].isin(MAIN)], figdir, bounds=bounds)

    _, meta = build_probe_battery()
    run_info = base[["run_id", "dataset", "arch", "seed"]]
    pair_ds = ["c13k_pair_desc", "c13k_pair_exp"]
    have_pairs = base["dataset"].isin(pair_ds).any()
    use_ds = pair_ds if have_pairs else ["c13k_desc", "c13k_exp"]
    preds = {}
    keep_ids = base[base["identifiable"]]["run_id"]
    run_info = run_info[run_info["run_id"].isin(keep_ids)]
    for rid in run_info[run_info["dataset"].isin(use_ds)]["run_id"]:
        z = np.load(os.path.join(RESULTS, "runs", rid, "arrays.npz"),
                    allow_pickle=True)
        preds[rid] = z["probe_pred"]
    h3path = os.path.join(tabdir, "h3_de_flip.csv")
    h3 = (pd.read_csv(h3path, index_col=[0, 1])
          if os.path.exists(h3path) else None)
    fig2_de_flip(meta, preds, run_info, figdir, datasets=use_ds, h3=h3)

    tpath = os.path.join(tabdir, "transfer.csv")
    if os.path.exists(tpath):
        fig3_transfer(pd.read_csv(tpath), figdir, order=MAIN,
                      eval_only=["cpc18_comp_desc", "cpc18_comp_exp"])

    dpath = os.path.join(tabdir, "h1_variance_decomposition.csv")
    if os.path.exists(dpath):
        dec = pd.read_csv(dpath)
        syn = base[base["dataset"].isin(["syn_c13k", "syn_cpc"])
                   & base["identifiable"]]
        fig4_decomposition(dec, syn, TRUE_THETA, figdir)

    print("figures ->", figdir)
