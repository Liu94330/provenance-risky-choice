"""Regenerate the per-architecture learning rates in src.models.ARCH_LR.

Every architecture gets the same sweep and the same selection rule, so no
architecture is handicapped by a rate that happens to suit another one. The
score is validation MSE normalised by the arm's own variance, averaged over
one large arm and one small arm -- normalising matters because CPC18 has a
much higher variance than choices13k and would otherwise dominate the mean.

This is a calibration step, not part of the main pipeline: it writes a table
for the manuscript and prints the dict to paste into src/models.py. Run it
again if the architectures or the arms change.

Usage: python scripts/09_lr_calibration.py [--quick]
"""
import _boot  # noqa
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from src.data import load_arm, split_indices, take
from src.models import ARCH_LR, make_model
from src.train import RESULTS

LRS = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
ARMS = ["c13k_exp", "cpc18_exp"]          # one large, one small

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    # Calibrate under the SAME budget the real runs get. Learning rate and
    # early stopping are coupled: a smaller rate that looks worse after 80
    # epochs can win outright when it has 600 to converge in. Choosing the
    # rate under a short budget and then training long picked a rate that made
    # SET 44% worse on one arm than the un-tuned default did.
    kw = dict(max_epochs=80, patience=20) if args.quick \
        else dict(max_epochs=600, patience=40)

    data = {}
    for d in ARMS:
        a = load_arm(d)
        data[d] = [take(a, i) for i in split_indices(len(a["y"]), seed=0)]

    rows = []
    for arch in [a for a in ARCH_LR]:
        for lr in LRS:
            for seed in range(args.seeds):
                for d in ARMS:
                    tr, va, _ = data[d]
                    m = make_model(arch, seed=seed, lr=lr, **kw)
                    h = m.fit(tr, va)
                    rows.append(dict(arch=arch, lr=lr, seed=seed, arm=d,
                                     val_mse=h["val_mse"],
                                     norm=h["val_mse"] / va["y"].var(),
                                     epochs=h["epochs"]))
        print(f"  {arch} done")

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="arch", columns="lr", values="norm",
                         aggfunc="mean")
    best = piv.idxmin(axis=1)
    piv["selected_lr"] = best
    out = os.path.join(RESULTS, "tables", "lr_calibration.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    piv.to_csv(out)
    print("\nnormalised val MSE (mean over arms and seeds):")
    print(piv.round(4).to_string())
    print("\nARCH_LR = {")
    for a, lr in best.items():
        print(f'    "{a}": {lr:g},')
    print("}")
    print("\nwrote", out)
