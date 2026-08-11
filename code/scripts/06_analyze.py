"""Headline statistics -> results/tables + results/summary.md

H1  variance decomposition (dataset vs architecture) on theory coordinates,
    behavioural indices, and fingerprint PCs; PERMANOVA with restricted
    permutations; silhouette by dataset vs by architecture.
H2  ground-truth recovery bias in the synthetic arms.
H3  description-experience sign flip on the matched problem pairs.
Controls: size-matched and feature-reweighted comparisons.
"""
import _boot  # noqa
import glob
import json
import os

import numpy as np
import pandas as pd

from src.analysis import (artifact_corrected, bootstrap_omega2,
                          experience_trend,
                          factorial_omega2, permanova, recovery_error,
                          silhouette_by, sufficiency_threshold,
                          two_way_omega2)
from src.data import TRUE_THETA
from src.theoryfit import IDENT_MIN_PROBE_SD, PARAM_NAMES
from src.train import RESULTS

MAIN = ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]
COORDS = PARAM_NAMES
INDICES = ["rare_event_index", "certainty_index", "reflection_index",
           "loss_aversion_kappa", "ev_consistency"]


def load_tables():
    coords = pd.read_csv(os.path.join(RESULTS, "tables", "theory_coords.csv"))
    idx = pd.read_csv(os.path.join(RESULTS, "tables", "indices.csv"))
    keep = [c for c in idx.columns if c not in coords.columns
            or c == "run_id"]
    df = coords.merge(idx[keep], on="run_id")
    base = df[df["subsample_n"].isna() & df["reweight_to"].isna()]
    return df, base


MIN_CELL = 3


def screened_omega2(sub, value, screen, min_cell=MIN_CELL):
    """omega^2 for one readout on the runs that pass `screen`.

    An architecture that is wiped out on some arm (fewer than `min_cell`
    surviving runs) cannot enter a two-way design at all, so it is dropped and
    named in the result. That is itself a finding, not bookkeeping: the
    architectures that vanish are the ones that learn nothing on the small
    arms, and dropping them removes the most extreme architecture effects, so
    the reported architecture omega^2 is if anything conservative."""
    d = sub[sub[screen]]
    counts = d.groupby(["dataset", "arch"]).size().unstack(fill_value=0)
    dropped = sorted(counts.columns[(counts < min_cell).any()].tolist())
    if dropped:
        d = d[~d["arch"].isin(dropped)]
    if d["arch"].nunique() < 2 or d["dataset"].nunique() < 2:
        return None
    out = two_way_omega2(d, value)
    out["screen"] = screen
    out["dropped_archs"] = ",".join(dropped)
    return out


def ident_report(sub, screen):
    """Per-cell 'kept/total' table for one screen, as a (dataset x arch)
    grid."""
    tot = sub.groupby(["dataset", "arch"]).size().unstack()
    kept = (sub[sub[screen]].groupby(["dataset", "arch"]).size().unstack()
            .reindex_like(tot).fillna(0).astype(int))
    return (kept.astype(str) + "/" + tot.astype(str)), kept, tot


def load_fingerprints(run_ids):
    F = []
    for rid in run_ids:
        z = np.load(os.path.join(RESULTS, "runs", rid, "arrays.npz"),
                    allow_pickle=True)
        F.append(z["probe_pred"])
    return np.stack(F)


if __name__ == "__main__":
    df, base = load_tables()
    main_all = base[base["dataset"].isin(MAIN)].copy()
    main = main_all[main_all["identifiable"]].copy()
    tabdir = os.path.join(RESULTS, "tables")
    lines = ["# Provenance vs architecture - summary\n"]

    # ----- identifiability: which runs have a theory to read at all ------
    lines.append("## Identifiability screens (runs kept / total)\n")
    lines.append(
        "Two screens, because the two kinds of readout fail differently.\n"
        f"  behaviour  probe SD >= {IDENT_MIN_PROBE_SD}: the network responds "
        "to the battery at all. Used for the behavioural indices.\n"
        "  theory     the above AND no CPT parameter pinned to its box "
        "constraint. Used for the CPT coordinates.\n")
    parts = {}
    for screen in ["identifiable", "theory_identifiable"]:
        tab, kept, tot = ident_report(main_all, screen)
        parts[screen] = tab
        lines.append(f"### {screen}")
        lines.append(tab.to_string())
        lines.append(f"kept {int(kept.to_numpy().sum())}/"
                     f"{int(tot.to_numpy().sum())} main-grid runs\n")
    pd.concat(parts, names=["screen"]).to_csv(
        os.path.join(tabdir, "identifiability.csv"))

    # ----- H1: omega^2 decomposition ------------------------------------
    # CPT coordinates need a fit that did not run into its bounds; the
    # behavioural indices are read straight off the probe predictions and only
    # need the network to do something, so they use the looser screen and keep
    # all six architectures.
    rows = [screened_omega2(main_all, v, "theory_identifiable")
            for v in COORDS if v in main_all.columns]
    rows += [screened_omega2(main_all, v, "identifiable")
             for v in INDICES if v in main_all.columns]
    dec = pd.DataFrame([r for r in rows if r is not None])
    dec.to_csv(os.path.join(tabdir, "h1_variance_decomposition.csv"),
               index=False)
    lines.append("## H1: partial omega^2 (dataset vs architecture)\n")
    show = ["value", "screen", "n", "dropped_archs", "omega2_dataset",
            "omega2_arch", "omega2_interaction"]
    lines.append(dec[show].round(3).to_string(index=False))
    for grp, sub in dec.groupby("screen", sort=False):
        med = sub[["omega2_dataset", "omega2_arch"]].median()
        n_ds = int((sub["omega2_dataset"] > sub["omega2_arch"]).sum())
        kind = ("CPT coordinates" if grp == "theory_identifiable"
                else "behavioural indices")
        lines.append(f"\n{kind} ({grp}, {len(sub)} readouts): median "
                     f"dataset {med['omega2_dataset']:.3f} vs architecture "
                     f"{med['omega2_arch']:.3f}; dataset wins "
                     f"{n_ds}/{len(sub)}")
    med = dec[["omega2_dataset", "omega2_arch"]].median()
    n_ds = int((dec["omega2_dataset"] > dec["omega2_arch"]).sum())
    lines.append(f"\nAll {len(dec)} readouts pooled: median dataset "
                 f"{med['omega2_dataset']:.3f} vs architecture "
                 f"{med['omega2_arch']:.3f}; dataset wins {n_ds}/{len(dec)}\n")

    # sensitivity of the headline to where the identifiability cut is drawn
    sens = []
    for thr in [0.0, 0.01, 0.02, 0.05, 0.10]:
        tmp = main_all.assign(_scr=main_all["probe_sd"] >= thr)
        rs = [screened_omega2(tmp, v, "_scr") for v in INDICES
              if v in tmp.columns]
        rs = [r for r in rs if r is not None]
        if not rs:
            continue
        d = pd.DataFrame(rs)
        sens.append(dict(
            probe_sd_threshold=thr, n_runs=int(tmp["_scr"].sum()),
            median_omega2_dataset=d["omega2_dataset"].median(),
            median_omega2_arch=d["omega2_arch"].median(),
            n_readouts_dataset_wins=int(
                (d["omega2_dataset"] > d["omega2_arch"]).sum()),
            dropped_archs=d["dropped_archs"].iloc[0]))
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(tabdir, "h1_threshold_sensitivity.csv"),
                index=False)
    lines.append("### Sensitivity of the behavioural indices to the "
                 "identifiability threshold\n")
    lines.append(sens.round(3).to_string(index=False) + "\n")

    # ----- H1 multivariate: PERMANOVA + silhouettes ---------------------
    mv = {}
    for label, sub in [("identifiable", main), ("all_runs", main_all)]:
        F = load_fingerprints(sub["run_id"].tolist())
        mv[label] = dict(
            permanova_dataset=permanova(F, sub["dataset"].to_numpy(),
                                        strata=sub["arch"].to_numpy()),
            permanova_arch=permanova(F, sub["arch"].to_numpy(),
                                     strata=sub["dataset"].to_numpy()),
            silhouette_by_dataset=silhouette_by(F, sub["dataset"]),
            silhouette_by_arch=silhouette_by(F, sub["arch"]),
            n=len(sub))
    with open(os.path.join(tabdir, "h1_multivariate.json"), "w") as f:
        json.dump(mv, f, indent=2)
    lines.append("## H1 multivariate (behavioural fingerprints)\n")
    for label, m in mv.items():
        lines.append(
            f"[{label}, n={m['n']}] PERMANOVA dataset: "
            f"F={m['permanova_dataset']['pseudo_F']:.1f}, "
            f"p={m['permanova_dataset']['p_value']:.4f} | architecture: "
            f"F={m['permanova_arch']['pseudo_F']:.1f}, "
            f"p={m['permanova_arch']['p_value']:.4f}")
        lines.append(
            f"[{label}] Silhouette by dataset "
            f"{m['silhouette_by_dataset']:.3f} vs by architecture "
            f"{m['silhouette_by_arch']:.3f}\n")

    # ----- H2: synthetic ground-truth recovery ---------------------------
    syn = base[base["dataset"].isin(["syn_c13k", "syn_cpc"])
               & base["theory_identifiable"]]
    if len(syn):
        n_syn_all = int(base["dataset"].isin(["syn_c13k", "syn_cpc"]).sum())
        rec = syn.groupby("dataset")[COORDS].agg(["mean", "std"]).round(3)
        rec.to_csv(os.path.join(tabdir, "h2_synthetic_recovery.csv"))
        lines.append("## H2: same agent, different problem distributions\n")
        lines.append(f"(identifiable runs only: {len(syn)}/{n_syn_all})")
        lines.append("True theta: " + json.dumps(TRUE_THETA))
        lines.append(rec.to_string() + "\n")
        gap = (syn[syn.dataset == "syn_c13k"][COORDS].mean()
               - syn[syn.dataset == "syn_cpc"][COORDS].mean())
        lines.append("Recovered-parameter gap (syn_c13k - syn_cpc):")
        lines.append(gap.round(3).to_string() + "\n")

    # ----- H3: D-E flip on matched problems ------------------------------
    pair_all = base[base["dataset"].isin(["c13k_pair_exp", "c13k_pair_desc"])]
    pair = pair_all[pair_all["identifiable"]]
    if len(pair):
        h3 = pair.groupby(["dataset", "arch"])["rare_event_index"] \
                 .agg(["mean", "std"]).round(4)
        h3.to_csv(os.path.join(tabdir, "h3_de_flip.csv"))
        lines.append("## H3: rare-event index, matched problems "
                     "(desc vs exp)\n")
        lines.append(f"(identifiable runs only: {len(pair)}/{len(pair_all)})")
        lines.append(h3.to_string())
        flip = pair.pivot_table(index="arch", columns="dataset",
                                values="rare_event_index")
        n_flip = int(((flip["c13k_pair_desc"] > 0) &
                      (flip["c13k_pair_exp"] < 0)).sum())
        lines.append(f"\nArchitectures showing the description>0 / "
                     f"experience<0 sign flip: {n_flip}/{len(flip)}\n")

    # ----- experience gradient (ordered factor) ---------------------------
    grad = base[base["dataset"].str.contains("_blk", na=False)].copy()
    if len(grad):
        grad["source"] = np.where(grad["dataset"].str.startswith("cpc18"),
                                  "cpc18", "c13k")
        grad["block"] = grad["dataset"].str.extract(r"blk(\d)").astype(int)
        rows = []
        for src, sub in grad.groupby("source"):
            for v in COORDS + INDICES:
                if v not in sub.columns:
                    continue
                screen = ("theory_identifiable" if v in COORDS
                          else "identifiable")
                s = sub[sub[screen]]
                r = experience_trend(s, v)
                if r:
                    rows.append(dict(source=src, **r))
        if rows:
            tr = pd.DataFrame(rows)
            tr.to_csv(os.path.join(tabdir, "experience_trend.csv"),
                      index=False)
            lines.append("## Experience gradient (blocks as an ordered "
                         "factor)\n")
            lines.append(
                "Spearman rho between block index and each readout, computed "
                "within architecture and averaged; p from permuting block "
                "labels within architecture.\n")
            lines.append(tr.round(3).to_string(index=False))
            piv = tr.pivot(index="value", columns="source",
                           values="mean_rho")
            if piv.shape[1] == 2:
                agree = int((np.sign(piv.iloc[:, 0])
                             == np.sign(piv.iloc[:, 1])).sum())
                lines.append(f"\nReadouts whose trend has the same sign in "
                             f"both sources: {agree}/{len(piv)}\n")

    # ----- controlled factorial (causal, ground truth known) --------------
    sim = base[base["dataset"].str.startswith("sim_", na=False)].copy()
    if len(sim):
        f = sim["dataset"].str.extract(
            r"sim_(?P<problems>[a-z0-9]+)_(?P<agent>desc|exp)"
            r"_n(?P<n_subj>\d+)_p(?P<n_problems>\d+)")
        sim = pd.concat([sim, f], axis=1)
        FACTORS = ["problems", "agent", "n_subj", "n_problems"]
        rows = []
        for v in COORDS + INDICES:
            if v not in sim.columns:
                continue
            screen = "theory_identifiable" if v in COORDS else "identifiable"
            r = factorial_omega2(sim[sim[screen]], v, FACTORS)
            if r:
                rows.append(r)
        if rows:
            fac = pd.DataFrame(rows)
            fac.to_csv(os.path.join(tabdir, "controlled_factorial.csv"),
                       index=False)
            lines.append("## Controlled factorial: which component of "
                         "provenance moves the learned theory?\n")
            lines.append(
                "2^4 design against a known generating agent (problem pool x "
                "agent weighting x respondents per problem x arm size), with "
                "architecture entered as a fifth factor. Every factor is "
                "crossed, so each share is free of the confounding that limits "
                "the observational arms.\n")
            show = ["value", "n"] + [c for c in fac.columns
                                     if c.startswith("omega2_")]
            lines.append(fac[show].round(3).to_string(index=False))
            med = fac[[c for c in fac.columns
                       if c.startswith("omega2_")]].median()
            lines.append("\nMedian across readouts:")
            lines.append(med.round(3).sort_values(ascending=False).to_string())
            top = med.idxmax().replace("omega2_", "")
            lines.append(f"\nLargest median effect: {top}\n")

    # ----- design sufficiency: how much data does the readout need? -------
    suf = base[base["dataset"].str.startswith("suf_", na=False)].copy()
    if len(suf):
        g = suf["dataset"].str.extract(r"suf_p(?P<n_problems>\d+)_n(?P<n_subj>\d+)")
        suf["n_problems"] = g["n_problems"].astype(int)
        suf["n_subj"] = g["n_subj"].astype(int)
        ok = suf[suf["theory_identifiable"]]
        err = recovery_error(ok, TRUE_THETA, COORDS,
                             ["n_subj", "n_problems"])
        if len(err):
            err = err.sort_values(["n_subj", "n_problems"])
            err.to_csv(os.path.join(tabdir, "sufficiency_error.csv"),
                       index=False)
            thr = sufficiency_threshold(err, "n_problems", tol=0.10)
            thr.to_csv(os.path.join(tabdir, "sufficiency_threshold.csv"),
                       index=False)
            lines.append("## Design sufficiency: how much data before the "
                         "readout can be trusted?\n")
            lines.append(
                "Known generating agent; only arm size and respondents per "
                "problem vary. Error is |recovered - true| / |true| averaged "
                "over the six CPT coordinates, so parameters on different "
                "scales are comparable. Identifiable runs only.\n")
            piv = err.pivot(index="n_subj", columns="n_problems",
                            values="mean_rel_err")
            lines.append("Mean relative recovery error:")
            lines.append(piv.round(3).to_string())
            lines.append("\nSmallest swept arm size reaching 10% error "
                         "(NaN = unreachable at that respondent count):")
            lines.append(thr.round(3).to_string(index=False) + "\n")

    # ----- artifact floor: what the readout reports when nothing differs ---
    if len(suf) and len(sim):
        # Both families get a floor. The behavioural indices never touch the
        # CPT fit, so a reviewer could reasonably assume they are immune --
        # they are not, and including them turns the check from 6 readouts
        # into 11 and removes that escape route.
        floor_arch, floor_pool = {}, {}
        for v in COORDS + INDICES:
            scr = "theory_identifiable" if v in COORDS else "identifiable"
            ra = factorial_omega2(suf[suf[scr]], v,
                                  ["n_problems", "n_subj"])
            rp = factorial_omega2(
                sim[sim[scr]], v,
                ["problems", "agent", "n_subj", "n_problems"])
            if ra:
                floor_arch[v] = ra["omega2_arch"]
            if rp:
                floor_pool[v] = rp["omega2_problems"]
        real = dec.set_index("value")
        ac = artifact_corrected(real, floor_arch, floor_pool,
                               COORDS + INDICES)
        if len(ac):
            ac.to_csv(os.path.join(tabdir, "artifact_corrected.csv"),
                      index=False)
            lines.append("## Artifact floor: does either effect survive it?\n")
            lines.append(
                "On the fixed-agent arms every run shares the same true "
                "coordinates, so any omega^2 the readout still reports is "
                "manufactured by the readout. Architecture perturbs the "
                "prediction surface directly and has a high floor; swapping "
                "the problem pool barely moves it, so each observed effect is "
                "compared with the floor for its own factor.\n")
            lines.append(ac.round(3).to_string(index=False))
            lines.append(
                f"\nArchitecture effects clearing their floor: "
                f"{int(ac['arch_survives'].sum())}/{len(ac)}")
            lines.append(
                f"Provenance effects clearing their floor: "
                f"{int(ac['dataset_survives'].sum())}/{len(ac)}\n")

            # Point estimates alone cannot support the comparison; bootstrap
            # both sides within design cells and report whether the intervals
            # separate.
            rr = main_all[main_all["theory_identifiable"]]
            cnt = rr.groupby(["dataset", "arch"]).size().unstack(fill_value=0)
            drop = cnt.columns[(cnt < MIN_CELL).any()].tolist()
            rr = rr[~rr["arch"].isin(drop)]
            ss = suf[suf["theory_identifiable"]]
            ss = ss[~ss["arch"].isin(drop)]
            sm = sim[sim["theory_identifiable"]]
            sm = sm[~sm["arch"].isin(drop)]
            rr_b = main_all[main_all["identifiable"]]
            rr_b = rr_b[~rr_b["arch"].isin(drop)]
            ss_b = suf[suf["identifiable"]]
            ss_b = ss_b[~ss_b["arch"].isin(drop)]
            sm_b = sim[sim["identifiable"]]
            sm_b = sm_b[~sm_b["arch"].isin(drop)]
            rows = []
            for v in COORDS + INDICES:
                R, S, M = ((rr, ss, sm) if v in COORDS
                           else (rr_b, ss_b, sm_b))
                oa = bootstrap_omega2(
                    R, v, "omega2_arch", ["dataset", "arch"],
                    lambda s, v=v: two_way_omega2(s, v))
                fa = bootstrap_omega2(
                    S, v, "omega2_arch", ["n_problems", "n_subj", "arch"],
                    lambda s, v=v: factorial_omega2(
                        s, v, ["n_problems", "n_subj"]))
                od = bootstrap_omega2(
                    R, v, "omega2_dataset", ["dataset", "arch"],
                    lambda s, v=v: two_way_omega2(s, v))
                fd = bootstrap_omega2(
                    M, v, "omega2_problems",
                    ["problems", "agent", "n_subj", "n_problems", "arch"],
                    lambda s, v=v: factorial_omega2(
                        s, v, ["problems", "agent", "n_subj", "n_problems"]))
                rows.append(dict(
                    value=v, family=("CPT coordinate" if v in COORDS
                                     else "behavioural index"),
                    arch_lo=oa["lo"], arch_hi=oa["hi"],
                    arch_floor_lo=fa["lo"], arch_floor_hi=fa["hi"],
                    arch_clears=bool(oa["lo"] > fa["hi"]),
                    ds_lo=od["lo"], ds_hi=od["hi"],
                    ds_floor_lo=fd["lo"], ds_floor_hi=fd["hi"],
                    ds_clears=bool(od["lo"] > fd["hi"])))
            bt = pd.DataFrame(rows)
            bt.to_csv(os.path.join(tabdir, "artifact_bootstrap.csv"),
                      index=False)
            lines.append("### Bootstrap 95% intervals (400 resamples)\n")
            lines.append(bt.round(3).to_string(index=False))
            lines.append(
                f"\nArchitecture: interval strictly above its floor in "
                f"{int(bt['arch_clears'].sum())}/{len(bt)} coordinates")
            lines.append(
                f"Provenance:   interval strictly above its floor in "
                f"{int(bt['ds_clears'].sum())}/{len(bt)} coordinates\n")

    # ----- controls -------------------------------------------------------
    ctrl_all = df[df["subsample_n"].notna() | df["reweight_to"].notna()]
    ctrl = ctrl_all[ctrl_all["theory_identifiable"]]
    if len(ctrl):
        cc = ctrl.groupby(["dataset", "subsample_n", "reweight_to"],
                          dropna=False)[COORDS + ["rare_event_index"]] \
                 .mean().round(3)
        cc.to_csv(os.path.join(tabdir, "controls_summary.csv"))
        lines.append("## Controls (size-matched / feature-reweighted)\n")
        lines.append(f"(identifiable runs only: {len(ctrl)}/{len(ctrl_all)})")
        lines.append(cc.to_string() + "\n")

    # ----- population arm -------------------------------------------------
    pop_all = base[base["dataset"].isin(["cpc18_technion_exp",
                                         "cpc18_rehovot_exp"])]
    pop = pop_all[pop_all["theory_identifiable"]]
    if len(pop):
        pp = pop.groupby("dataset")[COORDS + ["rare_event_index"]] \
                .mean().round(3)
        pp.to_csv(os.path.join(tabdir, "population_arm.csv"))
        lines.append("## Population arm (two labs, same problems/protocol)\n")
        lines.append(f"(identifiable runs only: {len(pop)}/{len(pop_all)})")
        lines.append(pp.to_string() + "\n")

    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nwrote results/summary.md")
