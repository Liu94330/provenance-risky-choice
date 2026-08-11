#!/usr/bin/env bash
# Full pipeline. Set C13K_DIR to your unzipped choices13k-main folder.
# QUICK=1 bash run_all.sh   -> 30-minute smoke version of everything
set -e
C13K_DIR="${C13K_DIR:-../choices13k-main}"
Q=""
if [ "${QUICK:-0}" = "1" ]; then Q="--quick"; fi

python code/scripts/00_prepare_data.py --c13k-dir "$C13K_DIR"
python code/tests/test_cpt_consistency.py

# Learning rates are calibrated per architecture and pinned in
# src/models.ARCH_LR. Re-run this only after changing the architectures or the
# arms; it rewrites results/tables/lr_calibration.csv and prints the dict to
# paste back. Left out of the default run because the pinned values are what
# the reported results use.
# python code/scripts/09_lr_calibration.py

python code/scripts/01_run_grid.py --suite main $Q
python code/scripts/01_run_grid.py --suite experience_gradient $Q
python code/scripts/01_run_grid.py --suite synthetic $Q
python code/scripts/01_run_grid.py --suite paired $Q
python code/scripts/01_run_grid.py --suite population $Q
python code/scripts/01_run_grid.py --suite size_control $Q
python code/scripts/01_run_grid.py --suite reweight_control $Q
python code/scripts/01_run_grid.py --suite controlled $Q
python code/scripts/01_run_grid.py --suite sufficiency $Q

python code/scripts/02_probe_models.py
python code/scripts/03_fit_theories.py
python code/scripts/04_transfer_matrix.py
python code/scripts/05_symbolic_distill.py
python code/scripts/06_analyze.py
python code/scripts/07_figures.py
python code/scripts/08_paper_outputs.py
python code/scripts/10_main_figures.py
