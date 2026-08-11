"""Paper-ready outputs: figures 5-10 and the ten numbered tables.

Figures 1-4 come from scripts/07_figures.py; this stage adds the rest and
writes every table twice -- a machine-readable CSV under results/tables/paper/
and a rendered block in results/tables/paper/TABLES.md carrying the caption
and the sample sizes, so the numbers that go into the manuscript and the
numbers on disk cannot drift apart.

Usage: python scripts/08_paper_outputs.py
"""
import _boot  # noqa
import json
import os

import numpy as np
import pandas as pd

from src.data import TRUE_THETA
from src.analysis import abstraction_ladder, probe_level_omega2
from src.figures_paper import (fig5_fingerprints, fig6_cpt_functions,
                               fig7_controls, fig8_identifiability,
                               fig9_symbolic, fig10_population_calibration,
                               fig11_abstraction,
                               fig12_experience_gradient,
                               fig13_controlled, fig14_sufficiency,
                               fig15_artifact_floor)
from src.probes import build_probe_battery
from src.theoryfit import IDENT_MIN_PROBE_SD, PARAM_NAMES
from src.train import RESULTS

MAIN = ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]
INDICES = ["rare_event_index", "certainty_index", "reflection_index",
           "loss_aversion_kappa", "ev_consistency"]
ARM_NOTE = {
    "c13k_exp": "MTurk, feedback-based, n~16/problem",
    "c13k_desc": "MTurk, described, n~16/problem",
    "c13k_pair_exp": "matched subset, experience protocol",
    "c13k_pair_desc": "matched subset, description protocol",
    "cpc18_exp": "two Israeli labs, experience, n~97-125",
    "cpc18_desc": "two Israeli labs, described, n~97-125",
    "cpc18_technion_exp": "Technion sample only",
    "cpc18_rehovot_exp": "Rehovot sample only",
    "syn_c13k": "simulated CPT agent on c13k problems",
    "syn_cpc": "simulated CPT agent on CPC18 problems",
}


def load_all():
    tab = os.path.join(RESULTS, "tables")
    coords = pd.read_csv(os.path.join(tab, "theory_coords.csv"))
    idx = pd.read_csv(os.path.join(tab, "indices.csv"))
    keep = [c for c in idx.columns if c not in coords.columns or c == "run_id"]
    df = coords.merge(idx[keep], on="run_id")
    return df, tab


def load_probe_preds(run_ids):
    out = {}
    for r in run_ids:
        p = os.path.join(RESULTS, "runs", r, "arrays.npz")
        if os.path.exists(p):
            out[r] = np.load(p, allow_pickle=True)["probe_pred"]
    return out


def load_test_preds(run_ids):
    out = {}
    for r in run_ids:
        p = os.path.join(RESULTS, "runs", r, "arrays.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            out[r] = dict(pred=z["test_pred"], y=z["test_y"])
    return out


PAPER_DIR = os.path.abspath(os.path.join(RESULTS, "..", "paper_results"))

# Manuscript figure order -> the file each one is built from. Renaming happens
# here and nowhere else, so figure numbers can be reshuffled for the paper
# without touching any plotting code.
FIGURE_ORDER = [
    ("fig01_theory_space", "fig1_theory_space",
     "Theory space: what each network learned, as CPT coordinates"),
    ("fig02_description_experience", "fig2_de_flip",
     "The description-experience gap and its sign flip"),
    ("fig03_transfer", "fig3_transfer",
     "Cross-provenance transfer and its asymmetry"),
    ("fig04_variance_decomposition", "fig4_decomposition",
     "Provenance vs architecture, and recovery of a known agent"),
    ("fig05_fingerprints", "fig5_fingerprints",
     "Behavioural fingerprints across the probe battery"),
    ("fig06_cpt_functions", "fig6_cpt_functions",
     "The weighting and value functions the networks imply"),
    ("fig07_controls", "fig7_controls",
     "Provenance or sample size: size-matched and reweighted controls"),
    ("fig08_identifiability", "fig8_identifiability",
     "Identifiability diagnostics for the coordinate fits"),
    ("fig09_symbolic", "fig9_symbolic",
     "Symbolic distillation of the certainty-equivalent surface"),
    ("fig10_population_calibration", "fig10_population_calibration",
     "Population arm and predictive calibration"),
    ("fig11_abstraction", "fig11_abstraction",
     "Where provenance wins: abstraction level and probe-space location"),
    ("fig12_experience_gradient", "fig12_experience_gradient",
     "Experience as an ordered factor, replicated in two independent designs"),
    ("fig13_controlled", "fig13_controlled",
     "Controlled factorial: which component of provenance moves the theory"),
    ("fig14_sufficiency", "fig14_sufficiency",
     "Design sufficiency: how much data the readout needs to be trusted"),
    ("fig15_artifact_floor", "fig15_artifact_floor",
     "Which effects survive the floor the readout manufactures on its own"),
]


def collect_paper_dir(figdir, tabledir, outdir=PAPER_DIR):
    """Assemble the manuscript deliverables in one place: numbered PNGs plus
    the captioned tables. Everything here is a copy -- regenerating the
    pipeline overwrites it, and nothing else reads from it."""
    import shutil

    fdir = os.path.join(outdir, "figures")
    tdir = os.path.join(outdir, "tables")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)

    lines = ["# Paper results\n",
             "Figures as 600 dpi PNG (vector PDF masters alongside) and the "
             "ten numbered tables. Regenerated by "
             "`python scripts/08_paper_outputs.py`; edits here are "
             "overwritten.\n", "## Figures\n"]
    missing = []
    for i, (dest, src, caption) in enumerate(FIGURE_ORDER, 1):
        got = []
        for ext in (".png", ".pdf"):
            s = os.path.join(figdir, src + ext)
            if os.path.exists(s):
                shutil.copyfile(s, os.path.join(fdir, dest + ext))
                got.append(ext[1:])
        if "png" not in got:
            missing.append(src)
        lines.append(f"- **Figure {i}** `{dest}.png` -- {caption}")
    lines.append("\n## Tables\n")

    n_tab = 0
    for f in sorted(os.listdir(tabledir)):
        if f.endswith(".csv") or f == "TABLES.md":
            shutil.copyfile(os.path.join(tabledir, f),
                            os.path.join(tdir, f))
            n_tab += f.endswith(".csv")
    lines.append("`TABLES.md` carries every table with its caption and "
                 f"sample sizes; `table01..{n_tab:02d}.csv` are the same "
                 "numbers machine-readable.\n")

    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n".join(lines))
    if missing:
        print(f"  WARNING: no PNG for {missing} -- run scripts/07_figures.py "
              "first so figures 1-4 are rebuilt with PNG output")
    return len(FIGURE_ORDER) - len(missing), n_tab, outdir


class Tables:
    """Collects numbered tables, writing CSV + a captioned markdown block."""

    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.blocks = []
        self.n = 0

    def add(self, frame, caption, note="", index=True, floatfmt="%.3f"):
        self.n += 1
        name = f"table{self.n:02d}"
        frame.to_csv(os.path.join(self.outdir, f"{name}.csv"), index=index,
                     float_format=floatfmt.replace("%", "%"))
        md = frame.round(4).to_markdown(index=index)
        self.blocks.append(
            f"## Table {self.n}. {caption}\n\n"
            + (f"_{note}_\n\n" if note else "") + md + "\n")
        print(f"  Table {self.n:2d}  {caption}")
        return name

    def write(self, path, title):
        with open(path, "w") as f:
            f.write(f"# {title}\n\n" + "\n".join(self.blocks))


if __name__ == "__main__":
    df, tabdir = load_all()
    figdir = os.path.join(RESULTS, "figures")
    paperdir = os.path.join(tabdir, "paper")
    base = df[df["subsample_n"].isna() & df["reweight_to"].isna()]
    ctrl = df[df["subsample_n"].notna() | df["reweight_to"].notna()]
    main_all = base[base["dataset"].isin(MAIN)]
    main_beh = main_all[main_all["identifiable"]]
    main_thy = main_all[main_all["theory_identifiable"]]
    meta = json.load(open(os.path.join(RESULTS, "..", "data", "processed",
                                       "meta.json")))
    probe_arrs, probe_meta = build_probe_battery()

    # ---------------- figures 5-10 -------------------------------------
    print("figures:")
    preds = load_probe_preds(main_beh["run_id"].tolist())
    mv = json.load(open(os.path.join(tabdir, "h1_multivariate.json")))
    fig5_fingerprints(preds, main_beh[["run_id", "dataset", "arch"]],
                      probe_meta, figdir, mv=mv.get("identifiable"))
    print("  fig5 fingerprints")

    fig6_cpt_functions(main_thy, figdir)
    print("  fig6 CPT functions")

    fig7_controls(base[base["theory_identifiable"]],
                  ctrl[ctrl["theory_identifiable"]], figdir)
    print("  fig7 controls")

    fig8_identifiability(main_all, figdir, thr=IDENT_MIN_PROBE_SD)
    print("  fig8 identifiability")

    sr = pd.read_csv(os.path.join(tabdir, "sr_expressions.csv"))
    fig9_symbolic(sr, figdir)
    print("  fig9 symbolic")

    # ---- abstraction ladder + probe-level map (figure 11, tables 11-12) ----
    F_raw = np.stack([preds[r] for r in main_beh["run_id"] if r in preds])
    ok = main_beh[main_beh["run_id"].isin(preds)]
    thy = ok[ok["theory_identifiable"]]
    F_thy = np.stack([preds[r] for r in thy["run_id"]])
    # All three levels are read off the SAME runs, so the trend cannot come
    # from each level being allowed a different subset. That has to be the
    # strict screen, since the CPT coordinates only exist there.
    ladder = abstraction_ladder(
        [("raw probe\nresponses", F_thy),
         ("behavioural\nindices", thy[INDICES].to_numpy()),
         ("CPT\ncoordinates", thy[list(PARAM_NAMES)].to_numpy())],
        thy["dataset"].to_numpy(), thy["arch"].to_numpy())
    # sensitivity: the two levels that survive the looser screen, on all of it
    ladder_loose = abstraction_ladder(
        [("raw probe\nresponses", F_raw),
         ("behavioural\nindices", ok[INDICES].to_numpy())],
        ok["dataset"].to_numpy(), ok["arch"].to_numpy())

    w_ds, w_ar = probe_level_omega2(F_raw, ok["dataset"].to_numpy(),
                                    ok["arch"].to_numpy())
    pomega = probe_meta.copy()
    pomega["omega2_dataset"], pomega["omega2_arch"] = w_ds, w_ar
    pomega["diff"] = w_ds - w_ar
    pomega.to_csv(os.path.join(tabdir, "probe_level_omega2.csv"), index=False)
    fig11_abstraction(ladder, pomega, figdir)
    print("  fig11 abstraction ladder + probe map")

    apath = os.path.join(tabdir, "artifact_corrected.csv")
    if os.path.exists(apath):
        fig15_artifact_floor(pd.read_csv(apath), figdir)
        print("  fig15 artifact floor")

    spath = os.path.join(tabdir, "sufficiency_error.csv")
    if os.path.exists(spath):
        h1p = os.path.join(tabdir, "h1_variance_decomposition.csv")
        fig14_sufficiency(pd.read_csv(spath), figdir,
                          h1=pd.read_csv(h1p) if os.path.exists(h1p) else None)
        print("  fig14 design sufficiency")

    fpath = os.path.join(tabdir, "controlled_factorial.csv")
    sim = base[base["dataset"].str.startswith("sim_", na=False)].copy()
    if os.path.exists(fpath) and len(sim):
        f = sim["dataset"].str.extract(
            r"sim_(?P<problems>[a-z0-9]+)_(?P<agent>desc|exp)"
            r"_n(?P<n_subj>\d+)_p(?P<n_problems>\d+)")
        sim = pd.concat([sim, f], axis=1)
        fig13_controlled(pd.read_csv(fpath), sim, figdir,
                         true_theta=TRUE_THETA)
        print("  fig13 controlled factorial")

    grad = base[base["dataset"].str.contains("_blk", na=False)].copy()
    if len(grad):
        grad["source"] = np.where(grad["dataset"].str.startswith("cpc18"),
                                  "cpc18", "c13k")
        grad["block"] = grad["dataset"].str.extract(r"blk(\d)").astype(int)
        tpath = os.path.join(tabdir, "experience_trend.csv")
        trend = pd.read_csv(tpath) if os.path.exists(tpath) else None
        fig12_experience_gradient(grad[grad["theory_identifiable"]], trend,
                                  figdir)
        print("  fig12 experience gradient")

    pop = base[base["dataset"].isin(["cpc18_technion_exp",
                                     "cpc18_rehovot_exp"])
               & base["theory_identifiable"]]
    tp = load_test_preds(main_beh[main_beh["dataset"]
                                  .isin(["c13k_exp", "cpc18_exp"])]["run_id"])
    fig10_population_calibration(pop, tp,
                                 main_beh[["run_id", "dataset", "arch"]],
                                 figdir)
    print("  fig10 population + calibration")

    # ---------------- tables 1-10 ---------------------------------------
    print("tables:")
    T = Tables(paperdir)

    t1 = pd.DataFrame([
        dict(arm=k, problems=v["n_problems"], mean_rate=v["mean_rate"],
             variance=v["var_y"], noise_floor_mse=v["noise_floor_mse"],
             ceiling_r2=1 - v["noise_floor_mse"] / v["var_y"],
             description=ARM_NOTE.get(k, ""))
        for k, v in meta.items()]).set_index("arm")
    T.add(t1, "Provenance arms.",
          "Aggregate choose-B rate per problem after dropping ambiguous "
          "items. The noise floor is the expected MSE of a perfect model "
          "given finite respondents per problem; ceiling r2 is the most any "
          "model can explain.")

    perf = main_all.groupby(["dataset", "arch"]).agg(
        test_mse=("test_mse", "mean"), test_mse_sd=("test_mse", "std"),
        epochs=("epochs", "mean"), fit_s=("fit_seconds", "mean")).round(4)
    perf["completeness"] = [
        (1 - r.test_mse / meta[d]["var_y"])
        / (1 - meta[d]["noise_floor_mse"] / meta[d]["var_y"])
        for (d, _), r in perf.iterrows()]
    T.add(perf, "Fit quality of every (arm x architecture) cell.",
          "Mean over 10 seeds on a held-out 15% split. Completeness "
          "normalises by that arm's noise ceiling.")

    dec = pd.read_csv(os.path.join(tabdir, "h1_variance_decomposition.csv"))
    T.add(dec.set_index("value"),
          "H1. Two-factor variance decomposition of every readout.",
          "Partial omega^2 from an unweighted-means ANOVA, seeds as "
          "replicates. CPT coordinates use the theory screen, behavioural "
          "indices the looser behaviour screen; dropped_archs lists "
          "architectures with fewer than three surviving runs in some arm.",
          index=True)

    ident = pd.read_csv(os.path.join(tabdir, "identifiability.csv"),
                        index_col=[0, 1])
    T.add(ident, "Identifiability screens, runs kept per cell.",
          f"behaviour screen: probe SD >= {IDENT_MIN_PROBE_SD}. theory "
          "screen: that, plus no CPT parameter pinned to a fitting bound.")

    T.add(pd.read_csv(os.path.join(tabdir, "h1_threshold_sensitivity.csv")),
          "Sensitivity of H1 to the identifiability threshold.",
          "Behavioural indices only, so the comparison keeps all six "
          "architectures at every threshold.", index=False)

    syn = base[base["dataset"].isin(["syn_c13k", "syn_cpc"])
               & base["theory_identifiable"]]
    rec = syn.groupby("dataset")[PARAM_NAMES].agg(["mean", "std"])
    truth = pd.Series({(p, "truth"): TRUE_THETA[p] for p in PARAM_NAMES})
    rec.loc["generating agent"] = truth.reindex(rec.columns)
    T.add(rec, "H2. Recovery of a known agent from two problem distributions.",
          "One simulated CPT agent, identical parameters, simulated on the "
          "c13k and CPC18 problem sets; every network then read back through "
          "the same probe battery.")

    h3 = pd.read_csv(os.path.join(tabdir, "h3_de_flip.csv"),
                     index_col=[0, 1]).unstack(0)
    T.add(h3, "H3. Rare-event index on the 1,266 matched problems.",
          "Same problems, same respondents pool, protocol swapped. A sign "
          "flip is description > 0 and experience < 0.")

    tr = pd.read_csv(os.path.join(tabdir, "transfer.csv"))
    T.add(tr.pivot_table(index="train", columns=["test"],
                         values="completeness").reindex(index=MAIN,
                                                        columns=MAIN),
          "H4. Cross-provenance transfer, noise-ceiling normalised.",
          "Averaged over architectures and seeds. Rows train, columns "
          "evaluate; 1.0 would mean everything explainable was explained.")

    cc = ctrl[ctrl["theory_identifiable"]].groupby(
        ["dataset", "subsample_n", "reweight_to"], dropna=False)[
            list(PARAM_NAMES) + ["rare_event_index"]].mean()
    T.add(cc, "Controls. Size-matched and feature-reweighted c13k.",
          "c13k sub-sampled to CPC18's 182 problems, and reweighted to "
          "CPC18's feature distribution, to separate provenance from sheer "
          "sample size.")

    pp = pop.groupby("dataset")[list(PARAM_NAMES) + INDICES].agg(
        ["mean", "std"])
    T.add(pp, "Population arm. Two labs, same problems, same protocol.",
          "An upper bound on how far respondent population alone moves the "
          "readout, for comparison against the provenance gaps above.")

    lad = ladder.copy()
    lad["level"] = lad["level"].str.replace("\n", " ")
    lad["subset"] = f"theory screen (n={len(thy)})"
    lad2 = ladder_loose.copy()
    lad2["level"] = lad2["level"].str.replace("\n", " ")
    lad2["subset"] = f"behaviour screen (n={len(ok)})"
    T.add(pd.concat([lad, lad2], ignore_index=True)
          .set_index(["subset", "level"]),
          "Where the contest is decided: variance shares by abstraction "
          "level.",
          "Distance-based R^2 for each factor, read at three levels of "
          "abstraction: the raw 1,742 probe responses, the five behavioural "
          "indices, and the six CPT coordinates. Every level is standardised "
          "before the distances, so shares are comparable across very "
          "different dimensionalities, and all three are read off the same "
          "runs. The second block repeats the two levels that survive the "
          "looser screen on all of it.")

    fam = (pomega.groupby("group")[["omega2_dataset", "omega2_arch", "diff"]]
           .agg(["mean", "count"]))
    fam.columns = ["_".join(c) for c in fam.columns]
    fam = fam.drop(columns=[c for c in fam.columns
                            if c.endswith("_count")][1:])
    T.add(fam, "Probe-level decomposition by problem family.",
          "The two-way decomposition run separately on each of the 1,742 "
          "probes, then averaged within family. Positive diff means "
          "provenance explains more of that family's variance than "
          "architecture does.")

    tpath = os.path.join(tabdir, "experience_trend.csv")
    if os.path.exists(tpath):
        T.add(pd.read_csv(tpath).set_index(["source", "value"]),
              "Experience gradient: is each readout monotone in block, and "
              "does the trend replicate?",
              "Spearman rho between block index and readout, computed within "
              "architecture and averaged; p from permuting block labels "
              "within architecture, so each architecture keeps its own level "
              "and spread and only the ordering is tested. CPC18 blocks are "
              "within-problem, choices13k blocks are between-problem; the two "
              "share no problems, respondents or country.")

    lpath = os.path.join(tabdir, "lr_calibration.csv")
    if os.path.exists(lpath):
        T.add(pd.read_csv(lpath).set_index("arch"),
              "Learning-rate calibration.",
              "Validation MSE normalised by each arm's variance, averaged "
              "over a large arm and a small one and over two seeds, under the "
              "same 600-epoch budget the reported runs get. Calibrating every "
              "architecture the same way keeps tuning differences from being "
              "scored as architecture differences: a single shared rate left "
              "SET and MLP_L short of their own optimum and made RESMLP "
              "collapse to a constant.")

    if os.path.exists(fpath):
        T.add(pd.read_csv(fpath).set_index("value"),
              "Controlled factorial: effect of each component of provenance.",
              "2^4 design against a known generating agent -- problem pool x "
              "agent weighting function x respondents per problem x arm size "
              "-- with architecture entered as a fifth factor so the "
              "comparison sits on one scale. Every factor is fully crossed, so "
              "unlike the observational arms each share is free of "
              "confounding. The two agents differ only in their probability "
              "weighting (Prelec gamma 0.65 vs 1.0), which is the one property "
              "the rare-event index is built to detect.")

    if os.path.exists(spath):
        T.add(pd.read_csv(spath).set_index(["n_subj", "n_problems"]),
              "Design sufficiency: recovery error by arm size and respondents "
              "per problem.",
              "Known generating agent, only the design varies. Error is "
              "|recovered - true| / |true| averaged over the six CPT "
              "coordinates so parameters on different scales are comparable. "
              "This is the prior question the rest of the paper depends on: "
              "without it, 'CPC18 models encode a different theory' cannot be "
              "told apart from '182 problems is too few to tell'.")
        tp = os.path.join(tabdir, "sufficiency_threshold.csv")
        if os.path.exists(tp):
            T.add(pd.read_csv(tp), "Smallest arm size reaching 10% recovery "
                  "error.", "NaN means no swept size reached the tolerance at "
                  "that respondent count -- which is the answer, not a gap.",
                  index=False)

    if os.path.exists(apath):
        T.add(pd.read_csv(apath).set_index("value"),
              "Artifact floor: observed effects against what the readout "
              "manufactures when nothing differs.",
              "On the fixed-agent arms every run shares the same true "
              "coordinates, so any omega^2 still reported is produced by the "
              "readout itself. Each observed effect is compared with the floor "
              "for its own factor, because architecture perturbs the "
              "prediction surface directly and has a much higher floor than "
              "swapping the problem pool does.", index=True)

    bpath = os.path.join(tabdir, "artifact_bootstrap.csv")
    if os.path.exists(bpath):
        T.add(pd.read_csv(bpath).set_index("value"),
              "Bootstrap 95% intervals for each effect and its artifact "
              "floor.",
              "400 resamples within design cells, so every replicate keeps a "
              "balanced design. `clears` is True only when the observed "
              "interval lies strictly above the floor interval -- comparing "
              "point estimates alone would not support the claim.", index=True)

    adp = os.path.join(tabdir, "attribution_decomposition.csv")
    if os.path.exists(adp):
        T.add(pd.read_csv(adp).set_index("feature_group"),
              "Attribution readout: variance decomposition of exact grouped "
              "Shapley values.",
              "A fourth readout that assumes no functional form -- it asks "
              "only how much the prediction moves when a group of input "
              "features is withheld. Shapley values are exact over eleven "
              "semantic groups (all 2^11 coalitions enumerated), not sampled, "
              "so architectures are comparable without approximation noise. "
              "Covers the five architectures that consume the flat 55-d "
              "encoding; SET, TRANSFORMER and CPT_NET read per-outcome slots, "
              "for which these groups do not exist. Note the artifact floors "
              "are far higher here than for the other readouts.")

    T.write(os.path.join(paperdir, "TABLES.md"),
            "Paper tables -- provenance vs architecture")
    print(f"\nwrote {T.n} tables -> {paperdir}")
    print("figures ->", figdir)

    print("\nrun scripts/10_main_figures.py to compose the main-text "
          "figures and lay out paper_results/")
