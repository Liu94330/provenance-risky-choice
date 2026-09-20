"""Run one (dataset, architecture, seed) grid cell end to end.

Saves everything downstream stages need, so trained models never have to be
reloaded: own-test metrics, cross-dataset test MSEs, probe predictions, and
(for CPT_NET) the learned interpretable parameters.

Artifacts per run, under results/runs/{dataset}__{arch}__s{seed}[__tag]/:
  run.json    config + metrics
  arrays.npz  probe_pred [P], test_pred, test_y
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .data import load_arm, split_indices, take, subsample, importance_weights
from .models import make_model
from .probes import build_probe_battery

from .paths import PROCESSED, RESULTS  # noqa: E402  (single source of truth)


def run_cell(dataset, arch, seed, eval_datasets, device="cpu",
             quick=False, subsample_n=None, reweight_to=None,
             out_root=None, split_seed=0):
    out_root = out_root or os.path.join(RESULTS, "runs")
    tag = ""
    if subsample_n:
        tag += f"__sub{subsample_n}"
    if reweight_to:
        tag += f"__rw-{reweight_to}"
    run_id = f"{dataset}__{arch}__s{seed}{tag}"
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    done_flag = os.path.join(out_dir, "run.json")
    if os.path.exists(done_flag):
        with open(done_flag) as f:
            return json.load(f)

    arrs = load_arm(dataset)
    if subsample_n and subsample_n < len(arrs["y"]):
        arrs = subsample(arrs, subsample_n, seed=seed)
    tr_idx, va_idx, te_idx = split_indices(len(arrs["y"]), seed=split_seed)
    train, val, test = take(arrs, tr_idx), take(arrs, va_idx), take(arrs, te_idx)

    sample_weight = None
    if reweight_to:
        target = load_arm(reweight_to)
        sample_weight = importance_weights(train, target, seed=seed)

    kw = {}
    if quick:
        kw = dict(max_epochs=40, patience=10)
    model = make_model(arch, seed=seed, device=device, **kw)
    t0 = time.time()
    hist = model.fit(train, val, sample_weight=sample_weight)
    fit_s = time.time() - t0

    test_pred = model.predict(test)
    own_mse = float(np.mean((test_pred - test["y"]) ** 2))

    cross = {}
    for d in eval_datasets:
        # Eval-only arms built from data we cannot redistribute (the CPC18
        # competition problems need Zenodo 2571510) are simply absent for most
        # readers. Skip them instead of failing the whole grid; every consumer
        # of these columns selects the ones that exist.
        if not os.path.exists(os.path.join(PROCESSED, f"{d}.npz")):
            continue
        darrs = load_arm(d)
        _, _, dte = split_indices(len(darrs["y"]), seed=split_seed)
        dtest = take(darrs, dte)
        pred = model.predict(dtest)
        cross[f"mse_on_{d}"] = float(np.mean((pred - dtest["y"]) ** 2))

    probe_arrs, _ = build_probe_battery()
    probe_pred = model.predict(probe_arrs)

    rec = dict(run_id=run_id, dataset=dataset, arch=arch, seed=seed,
               subsample_n=subsample_n, reweight_to=reweight_to,
               n_train=len(train["y"]), val_mse=hist["val_mse"],
               epochs=hist["epochs"], test_mse=own_mse, fit_seconds=fit_s,
               **cross, **{f"cptnet_{k}": v
                           for k, v in model.extra_state().items()})
    np.savez_compressed(os.path.join(out_dir, "arrays.npz"),
                        probe_pred=probe_pred, test_pred=test_pred,
                        test_y=test["y"], test_pid=test["pid"])
    with open(done_flag, "w") as f:
        json.dump(rec, f, indent=2)
    return rec
