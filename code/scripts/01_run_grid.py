"""Train the full grid. Examples:

  python scripts/01_run_grid.py                          # main 4-dataset grid
  python scripts/01_run_grid.py --suite synthetic        # ground-truth arms
  python scripts/01_run_grid.py --suite paired           # D-E matched problems
  python scripts/01_run_grid.py --suite population       # Technion vs Rehovot
  python scripts/01_run_grid.py --suite size_control     # c13k subsampled to CPC size
  python scripts/01_run_grid.py --suite reweight_control # c13k reweighted to CPC features

Add --quick for a smoke run (fewer epochs, 2 seeds). Runs are cached: an
existing results/runs/<id>/run.json is skipped, so re-running is safe.
"""
import _boot  # noqa
import argparse
import os

import pandas as pd
from tqdm import tqdm

from src.data import (CONTROLLED_FACTORS, SUFFICIENCY_PROBLEMS,
                      SUFFICIENCY_SUBJECTS)
from src.models import ARCHS
from src.train import RESULTS, run_cell

# derived from the factor table so the suite cannot drift from the arm builder
CONTROLLED_ARMS = [
    f"sim_{p}_{a}_n{n}_p{s}"
    for p in CONTROLLED_FACTORS["problems"]
    for a in CONTROLLED_FACTORS["agent"]
    for n in CONTROLLED_FACTORS["n_subj"]
    for s in CONTROLLED_FACTORS["n_problems"]]

SUFFICIENCY_ARMS = [f"suf_p{p}_n{n}" for p in SUFFICIENCY_PROBLEMS
                    for n in SUFFICIENCY_SUBJECTS]

MAIN = ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]
# Evaluation-only targets: the CPC18 competition problems (GameID 211-270).
# 48 problems is too few to train on -- an 80/10/10 split leaves a 5-problem
# test set, which makes per-arm MSE and completeness meaningless -- but they
# are the only genuinely unseen problems and subjects available, so they make
# an honest held-out transfer target instead. Nothing is ever trained on them.
EVAL_EXTRA = ["cpc18_comp_desc", "cpc18_comp_exp"]
EVAL_ON = MAIN + EVAL_EXTRA
SUITES = {
    "main": dict(datasets=MAIN, seeds=10),
    "synthetic": dict(datasets=["syn_c13k", "syn_cpc"], seeds=10),
    "paired": dict(datasets=["c13k_pair_exp", "c13k_pair_desc"], seeds=5),
    "population": dict(datasets=["cpc18_technion_exp", "cpc18_rehovot_exp"],
                       seeds=5),
    "size_control": dict(datasets=["c13k_exp", "c13k_desc"], seeds=5,
                         subsample=182),
    "reweight_control": dict(datasets=["c13k_exp"], seeds=5,
                             reweight_to="cpc18_exp"),
    # Experience as an ordered factor rather than a two-level contrast.
    # CPC18 blocks are the same 210 problems and respondents at five stages of
    # feedback (within-problem); choices13k assigns each problem to one block,
    # so its gradient is between-problem with ~2,450 problems per level. The
    # two designs are independent, so a trend that appears in both is not an
    # artefact of either.
    "experience_gradient": dict(
        datasets=[f"cpc18_blk{b}" for b in range(1, 6)]
                 + [f"c13k_blk{b}" for b in range(2, 6)], seeds=5),
    # 2^4 factorial against a known generating agent. Observational contrasts
    # between two real datasets confound problem pool, respondent count, arm
    # size and protocol at once; here each is varied on its own, so which
    # component of "provenance" actually moves the learned theory becomes a
    # causal question rather than an inference from two data points.
    "controlled": dict(datasets=CONTROLLED_ARMS, seeds=5),
    # How much data the readout itself needs. Fixed known agent, only arm size
    # and respondents per problem vary, so recovery error is a function of the
    # design and can be turned into a usable threshold.
    "sufficiency": dict(datasets=SUFFICIENCY_ARMS, seeds=3),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main", choices=list(SUITES))
    ap.add_argument("--archs", nargs="*", default=ARCHS)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = SUITES[args.suite]
    seeds = args.seeds or cfg["seeds"]
    if args.quick:
        seeds = min(seeds, 2)
    jobs = [(d, a, s) for d in cfg["datasets"] for a in args.archs
            for s in range(seeds)]
    recs = []
    for d, a, s in tqdm(jobs, desc=f"grid[{args.suite}]"):
        recs.append(run_cell(d, a, s, eval_datasets=EVAL_ON,
                             device=args.device, quick=args.quick,
                             subsample_n=cfg.get("subsample"),
                             reweight_to=cfg.get("reweight_to")))
    idx_path = os.path.join(RESULTS, f"runs_{args.suite}.csv")
    pd.DataFrame(recs).to_csv(idx_path, index=False)
    print("wrote", idx_path)
