"""Distill each model's certainty-equivalent surface into a closed-form
expression via symbolic regression. Run on a subset of seeds by default
(SR is the slowest stage)."""
import _boot  # noqa
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.probes import build_probe_battery, ce_surface
from src.sr_distill import distill_ce, distill_weighting_curve
from src.train import RESULTS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gplearn",
                    choices=["gplearn", "pysr"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    _, meta = build_probe_battery()
    rows = []
    run_dirs = sorted(glob.glob(os.path.join(RESULTS, "runs", "*")))
    for run_dir in tqdm(run_dirs):
        with open(os.path.join(run_dir, "run.json")) as f:
            rec = json.load(f)
        if rec["seed"] not in args.seeds or rec.get("subsample_n") \
                or rec.get("reweight_to"):
            continue
        z = np.load(os.path.join(run_dir, "arrays.npz"), allow_pickle=True)
        for sign, tag in [(1, "gain"), (-1, "loss")]:
            ce = ce_surface(z["probe_pred"], meta, sign=sign)
            d = distill_ce(ce, backend=args.backend, seed=rec["seed"])
            w = distill_weighting_curve(ce, backend=args.backend,
                                        seed=rec["seed"])
            rows.append(dict(rec, domain=tag,
                             ce_expression=d["expression"], ce_r2=d["r2"],
                             ce_complexity=d["complexity"],
                             w_expression=(w or {}).get("expression"),
                             backend=d["backend"]))
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    out = os.path.join(RESULTS, "tables", "sr_expressions.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, f"({len(rows)} rows)")
