"""Render the six main-text figures and lay out paper_results/.

The seventeen figures under results/figures are one per analysis -- the right
granularity while working, the wrong one for a manuscript. This stage composes
them into six main figures, each carrying one argument across three or four
panels, and files the rest as supplementary.

paper_results/ ends up as:
  figures/main/  fig1..fig6, the main text
  figures/supp/  figS1..figS8, everything a reviewer may ask to see
  tables/        every numbered table plus TABLES.md

Run after 06, 08 and shapcode/run_figures.py.
Usage, from the repository root: python3 code/scripts/10_main_figures.py
"""
import _boot  # noqa
import json
import os
import shutil

import numpy as np
import pandas as pd

from src.data import TRUE_THETA
from src.figures_main import (main1_what_is_learned, main2_artifact_floor,
                              main3_causal, main4_description_experience,
                              main5_transfer, main6_reliability,
                              supp_controlled_extra, supp_robustness)
from src.probes import build_probe_battery
from src.train import RESULTS

MAIN = ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]
EVAL_EXTRA = ["cpc18_comp_desc", "cpc18_comp_exp"]

# Main text: composed figures, in the order the argument is made.
MAIN_FIGS = [
    ("fig1_what_is_learned", "main1_what_is_learned",
     "What a network encodes, read at four levels of abstraction"),
    ("fig2_artifact_floor", "main2_artifact_floor",
     "Only provenance clears the floor the readout manufactures on its own"),
    ("fig3_causal", "main3_causal",
     "Which component of provenance does the work (manipulated, not inferred)"),
    ("fig4_description_experience", "main4_description_experience",
     "The description-experience gap as a flip and as a trajectory"),
    ("fig5_transfer", "main5_transfer",
     "Transfer, including to problems and respondents never seen"),
    ("fig6_reliability", "main6_reliability",
     "When the readout can be trusted"),
]

# Supplementary: the analyses that support a main panel but do not need to be
# in the main text.
SUPP_FIGS = [
    ("figS1_cpt_functions", "fig6_cpt_functions",
     "Full implied CPT functions: w+(p), w-(p) and v(x)"),
    ("figS2_symbolic", "fig9_symbolic",
     "Symbolic distillation of the certainty-equivalent surface"),
    ("figS3_population", "fig10_population_calibration",
     "Population arm (two labs, same problems) and predictive calibration"),
    ("figS4_probe_map", "fig11_abstraction",
     "Probe-space map of where provenance leads architecture"),
    ("figS5_sufficiency", "fig14_sufficiency",
     "Design sufficiency and the parameter-screen sensitivity"),
    ("figS6_controlled_extra", "supp_controlled_extra",
     "Interaction check, index responsiveness, per-family ordering and screen "
     "sensitivity"),
    ("figS7_attribution", "fig_attribution",
     "Attribution readout: which parts of a gamble each network relies on"),
    ("figS8_attribution_probe_level", "fig_attribution_rich",
     "Per-probe Shapley values: beeswarm, dependence, conditional map and the "
     "cross-check against the CPT readout"),
    ("figS9_robustness", "supp_robustness",
     "Robustness of the floor-clearing result: five floor definitions, "
     "family-wise correction, leave-one-architecture-out, and the distilled "
     "expressions compared as functions"),
]


def load():
    tab = os.path.join(RESULTS, "tables")
    coords = pd.read_csv(os.path.join(tab, "theory_coords.csv"))
    idx = pd.read_csv(os.path.join(tab, "indices.csv"))
    keep = [c for c in idx.columns if c not in coords.columns or c == "run_id"]
    return coords.merge(idx[keep], on="run_id"), tab


def probe_preds(run_ids):
    out = {}
    for r in run_ids:
        p = os.path.join(RESULTS, "runs", r, "arrays.npz")
        if os.path.exists(p):
            out[r] = np.load(p, allow_pickle=True)["probe_pred"]
    return out


def collect(figdir, tabdir, outdir):
    fmain = os.path.join(outdir, "figures", "main")
    fsupp = os.path.join(outdir, "figures", "supp")
    tdir = os.path.join(outdir, "tables")
    for d in (fmain, fsupp, tdir):
        os.makedirs(d, exist_ok=True)

    lines = ["# Paper results\n",
             "Figures as 600 dpi PNG with the vector PDF master alongside. "
             "Composed and filed by `python3 code/scripts/10_main_figures.py` "
             "(run from the repository root, after stage 08 and after "
             "`shapcode/run_figures.py`, which produces S7 and S8); anything "
             "edited here is overwritten.\n"]
    missing = []
    for label, items, dest in [("Main text", MAIN_FIGS, fmain),
                               ("Supplementary", SUPP_FIGS, fsupp)]:
        lines.append(f"## {label}\n")
        for i, (name, src, caption) in enumerate(items, 1):
            got = False
            for ext in (".png", ".pdf"):
                s = os.path.join(figdir, src + ext)
                if os.path.exists(s):
                    shutil.copyfile(s, os.path.join(dest, name + ext))
                    got = got or ext == ".png"
            if not got:
                missing.append(src)
            num = f"Figure {i}" if label == "Main text" else f"Figure S{i}"
            lines.append(f"- **{num}** `{name}.png` -- {caption}")
        lines.append("")

    # the numbered manuscript tables, not the intermediate artefacts in
    # results/tables (theory_coords.csv and friends are per-run records, not
    # something that goes in a paper)
    src_tab = os.path.join(tabdir, "paper")
    n_tab = 0
    if os.path.isdir(src_tab):
        for f in sorted(os.listdir(src_tab)):
            if f.startswith("table") and f.endswith(".csv") \
                    or f == "TABLES.md":
                shutil.copyfile(os.path.join(src_tab, f),
                                os.path.join(tdir, f))
                n_tab += f.endswith(".csv")
    lines.append("## Tables\n")
    lines.append(f"`TABLES.md` carries all {n_tab} tables with captions and "
                 "sample sizes; `tableNN.csv` are the same numbers "
                 "machine-readable. Per-run records "
                 "(`theory_coords.csv`, `indices.csv`, ...) stay in "
                 "`results/tables/` -- they are inputs, not manuscript "
                 "tables.\n")
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n".join(lines))
    if missing:
        print(f"  WARNING: no PNG for {missing}")
    return len(MAIN_FIGS) + len(SUPP_FIGS) - len(missing), n_tab


if __name__ == "__main__":
    df, tabdir = load()
    figdir = os.path.join(RESULTS, "figures")
    base = df[df["subsample_n"].isna() & df["reweight_to"].isna()]
    ctrl = df[df["subsample_n"].notna() | df["reweight_to"].notna()]
    main_all = base[base["dataset"].isin(MAIN)]
    _, probe_meta = build_probe_battery()

    def table(name):
        p = os.path.join(tabdir, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    print("main figures:")
    beh = main_all[main_all["identifiable"]]
    preds = probe_preds(beh["run_id"].tolist())
    mv = json.load(open(os.path.join(tabdir, "h1_multivariate.json")))
    main1_what_is_learned(preds, beh[["run_id", "dataset", "arch"]],
                          probe_meta, main_all, figdir,
                          mv=mv.get("identifiable"))
    print("  fig1 what is learned")

    bt, err = table("artifact_bootstrap.csv"), table("sufficiency_error.csv")
    lad = table("table11.csv") if os.path.exists(
        os.path.join(tabdir, "table11.csv")) else None
    if lad is None:
        lad = pd.read_csv(os.path.join(tabdir, "paper", "table11.csv"))
    # keep both screens: panel e plots them as two series
    if bt is not None and err is not None and lad is not None:
        main2_artifact_floor(bt, figdir)
        print("  fig2 artifact floor")

    fac = table("controlled_factorial.csv")
    sim = base[base["dataset"].str.startswith("sim_", na=False)].copy()
    if fac is not None and len(sim):
        f = sim["dataset"].str.extract(
            r"sim_(?P<problems>[a-z0-9]+)_(?P<agent>desc|exp)"
            r"_n(?P<n_subj>\d+)_p(?P<n_problems>\d+)")
        sim = pd.concat([sim, f], axis=1)
        main3_causal(fac, sim, base[base["theory_identifiable"]],
                     ctrl[ctrl["theory_identifiable"]], figdir,
                     true_theta=TRUE_THETA)
        print("  fig3 causal")
        supp_controlled_extra(fac, sim, err, lad,
                              table("h1_threshold_sensitivity.csv"), figdir)
        print("  figS6 cut panels")

    h3 = pd.read_csv(os.path.join(tabdir, "h3_de_flip.csv"), index_col=[0, 1])
    grad = base[base["dataset"].str.contains("_blk", na=False)].copy()
    grad["source"] = np.where(grad["dataset"].str.startswith("cpc18"),
                              "cpc18", "c13k")
    grad["block"] = grad["dataset"].str.extract(r"blk(\d)").astype(int)
    pair_ids = base[base["dataset"].isin(["c13k_pair_desc", "c13k_pair_exp"])
                    & base["identifiable"]]
    ppreds = probe_preds(pair_ids["run_id"].tolist())
    main4_description_experience(
        probe_meta, ppreds, pair_ids[["run_id", "dataset", "arch"]], h3,
        grad[grad["identifiable"]], table("experience_trend.csv"), figdir)
    print("  fig4 description-experience")

    # Attribution figures are produced by shapcode/run_figures.py; this stage
    # only files them. Keeping the generation out here means the main pipeline
    # does not import the attribution module.

    tr = table("transfer.csv")
    if tr is not None:
        main5_transfer(tr, figdir, order=MAIN, eval_only=EVAL_EXTRA)
        print("  fig5 transfer")

    # Robustness of the headline (11_robustness.py); filed as S9.
    fd_, fs_, js_ = (table("robustness_floor_definitions.csv"),
                     table("robustness_floor_summary.csv"),
                     table("robustness_jackknife_summary.csv"))
    sym_, pca_ = (table("symbolic_clustering.csv"),
                  table("symbolic_fingerprint_pca.csv"))
    if all(x is not None for x in (fd_, fs_, js_, sym_, pca_)):
        pmeta = os.path.join(tabdir, "symbolic_fingerprint_pca.json")
        pmeta = json.load(open(pmeta)) if os.path.exists(pmeta) else None
        supp_robustness(fd_, fs_, js_, sym_, pca_, figdir, pca_meta=pmeta)
        print("  figS9 robustness")

    if err is not None:
        main6_reliability(main_all, err, figdir)
        print("  fig6 reliability")

    out = os.path.abspath(os.path.join(RESULTS, "..", "paper_results"))
    n_fig, n_tab = collect(figdir, tabdir, out)
    print(f"\ncollected {n_fig} figures ({len(MAIN_FIGS)} main + "
          f"{len(SUPP_FIGS)} supplementary) + {n_tab} tables -> {out}")
