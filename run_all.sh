#!/usr/bin/env bash
# Full pipeline. Set C13K_DIR to your unzipped choices13k-main folder.
# QUICK=1 bash run_all.sh   -> 30-minute smoke version of everything
set -e

# Some systems ship only `python3`. Override with PYTHON=/path/to/python.
PY="${PYTHON:-python3}"
C13K_DIR="${C13K_DIR:-../choices13k-main}"
Q=""
if [ "${QUICK:-0}" = "1" ]; then Q="--quick"; fi

"$PY" code/scripts/00_prepare_data.py --c13k-dir "$C13K_DIR"
"$PY" code/tests/test_cpt_consistency.py
"$PY" code/tests/test_lottery_expansion.py
"$PY" shapcode/test_shapley.py

# Learning rates are calibrated per architecture and pinned in
# src/models.ARCH_LR. Re-run this only after changing the architectures or the
# arms; it rewrites results/tables/lr_calibration.csv and prints the dict to
# paste back. Left out of the default run because the pinned values are what
# the reported results use.
# "$PY" code/scripts/09_lr_calibration.py

"$PY" code/scripts/01_run_grid.py --suite main $Q
"$PY" code/scripts/01_run_grid.py --suite experience_gradient $Q
"$PY" code/scripts/01_run_grid.py --suite synthetic $Q
"$PY" code/scripts/01_run_grid.py --suite paired $Q
"$PY" code/scripts/01_run_grid.py --suite population $Q
"$PY" code/scripts/01_run_grid.py --suite size_control $Q
"$PY" code/scripts/01_run_grid.py --suite reweight_control $Q
"$PY" code/scripts/01_run_grid.py --suite controlled $Q
"$PY" code/scripts/01_run_grid.py --suite sufficiency $Q

"$PY" code/scripts/02_probe_models.py
"$PY" code/scripts/03_fit_theories.py
"$PY" code/scripts/04_transfer_matrix.py
"$PY" code/scripts/05_symbolic_distill.py
"$PY" code/scripts/06_analyze.py
"$PY" code/scripts/11_robustness.py
"$PY" code/scripts/07_figures.py
"$PY" code/scripts/08_paper_outputs.py
"$PY" shapcode/run_decomposition.py
"$PY" shapcode/run_figures.py
"$PY" code/scripts/10_main_figures.py
