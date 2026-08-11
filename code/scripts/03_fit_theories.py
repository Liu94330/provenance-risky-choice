"""Fit CPT theory coordinates to every run's probe predictions.

Each fit is an independent L-BFGS-B problem over the same fixed probe battery,
so the work parallelises exactly: with 1,760 runs the serial version takes
around two hours and this one takes minutes. The result is identical either
way -- fit_cpt_to_predictions is deterministic given (predictions, seed) and no
state is shared between fits. Use --jobs 1 to reproduce serially.
"""
import _boot  # noqa
import argparse
import glob
import json
import os

# One BLAS thread per worker. Each fit is small, so the threads inside numpy
# add nothing and instead oversubscribe the cores the worker processes are
# already using -- with the default settings the pool ran only 1.7x faster than
# serial instead of near-linearly. Must be set before numpy is imported, and
# because spawn re-imports this module in each worker, that is where it lands.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from multiprocessing import Pool  # noqa: E402

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.probes import build_probe_battery
from src.theoryfit import fit_cpt_to_predictions
from src.train import RESULTS

_PROBE = None


def _init():
    """Build the probe battery once per worker rather than per fit."""
    global _PROBE
    _PROBE, _ = build_probe_battery()


def _one(run_dir):
    with open(os.path.join(run_dir, "run.json")) as f:
        rec = json.load(f)
    z = np.load(os.path.join(run_dir, "arrays.npz"), allow_pickle=True)
    rec.update(fit_cpt_to_predictions(_PROBE, z["probe_pred"]))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes; 0 = cpu_count-2, 1 = serial")
    args = ap.parse_args()
    run_dirs = sorted(glob.glob(os.path.join(RESULTS, "runs", "*")))
    n_jobs = args.jobs or max(1, (os.cpu_count() or 2) - 2)

    if n_jobs == 1:
        _init()
        rows = [_one(d) for d in tqdm(run_dirs)]
    else:
        with Pool(n_jobs, initializer=_init) as pool:
            rows = list(tqdm(pool.imap(_one, run_dirs, chunksize=8),
                             total=len(run_dirs)))

    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    out = os.path.join(RESULTS, "tables", "theory_coords.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, f"({len(rows)} runs, {n_jobs} jobs)")
