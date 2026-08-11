"""Compute behavioural fingerprint indices for every saved run."""
import _boot  # noqa
import glob
import json
import os

import numpy as np
import pandas as pd

from src.probes import build_probe_battery, fingerprint_indices
from src.train import RESULTS

if __name__ == "__main__":
    _, meta = build_probe_battery()
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(RESULTS, "runs", "*"))):
        with open(os.path.join(run_dir, "run.json")) as f:
            rec = json.load(f)
        z = np.load(os.path.join(run_dir, "arrays.npz"), allow_pickle=True)
        rec.update(fingerprint_indices(z["probe_pred"], meta))
        rows.append(rec)
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    out = os.path.join(RESULTS, "tables", "indices.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, f"({len(rows)} runs)")
