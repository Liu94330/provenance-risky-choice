"""Attribution readout: which parts of a gamble each trained network uses.

Fourth readout in the paper, and the only one that assumes no functional form
(see src/shap_readout for why that matters). Produces, per trained model, an
exact Shapley value for each of eleven semantic feature groups, then runs the
same two questions asked of every other readout:

  1. how much of the variation in those attributions is explained by data
     provenance vs by architecture;
  2. does either effect clear the floor the readout manufactures on its own,
     measured on fixed-agent arms where the true generating process is
     identical across runs.

Models are retrained here rather than reloaded: 01_run_grid stores predictions
and not weights, and attribution needs to perturb inputs. Training is
deterministic given the seed, so these are the same models the rest of the
pipeline analysed.

Usage: python code/scripts/11_attribution.py [--seeds 5]
"""
import _boot  # noqa
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.analysis import factorial_omega2, two_way_omega2
from src.data import load_arm, split_indices, take
from src.models import make_model
from src.paths import TABLES
from src.probes import build_probe_battery
from src.shap_readout import (FEATURE_GROUPS, FLAT_ARCHS,
                              attribution_fingerprint)

MAIN = ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]
# Fixed-agent arms: the generating theory is identical across every run, so
# any attribution difference they show is manufactured by the readout.
FLOOR_ARMS = ["suf_p250_n16", "suf_p250_n128", "suf_p1000_n16",
              "suf_p1000_n128"]


def fingerprints(arms, archs, seeds, probe_arrs, label):
    rows = []
    jobs = [(d, a, s) for d in arms for a in archs for s in range(seeds)]
    for d, a, s in tqdm(jobs, desc=label):
        arrs = load_arm(d)
        tr_i, va_i, _ = split_indices(len(arrs["y"]), seed=0)
        tr, va = take(arrs, tr_i), take(arrs, va_i)
        model = make_model(a, seed=s)
        model.fit(tr, va)
        norm, raw = attribution_fingerprint(model, probe_arrs, tr, seed=s)
        rows.append(dict(dataset=d, arch=a, seed=s,
                         **{f"attr_{k}": v for k, v in norm.items()},
                         **{f"raw_{k}": v for k, v in raw.items()}))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--floor-seeds", type=int, default=3)
    args = ap.parse_args()

    probe_arrs, _ = build_probe_battery()
    archs = list(FLAT_ARCHS)
    print(f"architectures with a flat 55-d input: {archs}")
    print("(SET / TRANSFORMER / CPT_NET read per-outcome slots, so these "
          "feature groups do not exist for them)\n")

    real = fingerprints(MAIN, archs, args.seeds, probe_arrs, "main arms")
    floor = fingerprints(FLOOR_ARMS, archs, args.floor_seeds, probe_arrs,
                         "fixed-agent arms")
    os.makedirs(TABLES, exist_ok=True)
    real.to_csv(os.path.join(TABLES, "attribution_fingerprints.csv"),
                index=False)

    # variance decomposition on the observed arms, and the same quantity on
    # arms where nothing about the generating process differs
    groups = list(FEATURE_GROUPS)
    floor["n_problems"] = floor["dataset"].str.extract(r"p(\d+)").astype(int)
    floor["n_subj"] = floor["dataset"].str.extract(r"n(\d+)$").astype(int)
    rows = []
    for g in groups:
        col = f"attr_{g}"
        obs = two_way_omega2(real, col)
        fl = factorial_omega2(floor, col, ["n_problems", "n_subj"])
        rows.append(dict(
            feature_group=g,
            omega2_dataset=obs["omega2_dataset"],
            omega2_arch=obs["omega2_arch"],
            arch_artifact_floor=(fl or {}).get("omega2_arch", np.nan),
            dataset_artifact_floor=(fl or {}).get("omega2_n_problems",
                                                  np.nan)))
    dec = pd.DataFrame(rows)
    dec["arch_clears"] = dec["omega2_arch"] > dec["arch_artifact_floor"]
    dec["dataset_clears"] = (dec["omega2_dataset"]
                             > dec["dataset_artifact_floor"])
    dec.to_csv(os.path.join(TABLES, "attribution_decomposition.csv"),
               index=False)

    print("\nAttribution variance decomposition (normalised group shares):")
    print(dec.round(3).to_string(index=False))
    med = dec[["omega2_dataset", "omega2_arch"]].median()
    print(f"\nMedian across feature groups: provenance "
          f"{med['omega2_dataset']:.3f} vs architecture "
          f"{med['omega2_arch']:.3f}")
    print(f"Clearing their own artifact floor -- provenance "
          f"{int(dec['dataset_clears'].sum())}/{len(dec)}, architecture "
          f"{int(dec['arch_clears'].sum())}/{len(dec)}")
    print(f"\nwrote {TABLES}/attribution_fingerprints.csv and "
          f"attribution_decomposition.csv")
