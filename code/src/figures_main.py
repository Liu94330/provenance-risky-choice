"""The six main-text figures.

Reorganised from the fifteen exploratory panels that came before: those were
one figure per analysis, which is the right granularity while working and the
wrong one for a manuscript. Here each figure carries one argument across three
or four panels, in the order the paper makes them:

  main1  what a network ends up encoding, from raw behaviour to fitted theory
  main2  the headline: only provenance clears the floor the readout invents
  main3  the causal version -- which component of provenance does the work
  main4  description vs experience, as a flip and as a trajectory
  main5  transfer, including to problems and people never seen
  main6  when the readout can be trusted at all

Everything is drawn from `results/tables/`, so the figures cannot disagree with
the numbers. Panels are small builders (`_p_*`) that take an axis, which keeps
each panel defined once and lets the composition change without touching the
drawing code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .figures import (ARCH_MARKERS, DATASET_COLORS, DATASET_LABELS,
                      DATASET_SHORT, FAINT, INK, MUTED, RC, _ellipse,
                      _panel_tag, _pname, _save)
from .figures_paper import (BLOCK_COLORS, GROUP_ORDER, PROBE_GROUP_LABELS,
                            _pca)
from .problems import prelec_w

DS_C, AR_C = "#0072B2", "#D55E00"


def _tag(ax, t):
    _panel_tag(ax, t)


# ---------------------------------------------------------------------------
# panel builders
# ---------------------------------------------------------------------------

def _p_probe_profile(ax, preds, info, probe_meta):
    gmask = {g: (probe_meta["group"] == g).to_numpy() for g in GROUP_ORDER}
    rid = [r for r in info["run_id"] if r in preds]
    inf = info.set_index("run_id").loc[rid]
    F = np.stack([preds[r] for r in rid])
    prof = np.stack([[F[i][gmask[g]].mean() for g in GROUP_ORDER]
                     for i in range(len(F))])
    xs = np.arange(len(GROUP_ORDER))
    for ds in inf["dataset"].unique():
        sel = (inf["dataset"] == ds).to_numpy()
        c = DATASET_COLORS.get(ds, "gray")
        for row in prof[sel]:
            ax.plot(xs, row, color=c, lw=0.6, alpha=0.22, zorder=2)
        ax.plot(xs, prof[sel].mean(0), color=c, lw=2.1, zorder=4,
                label=DATASET_LABELS.get(ds, ds))
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--", zorder=1)
    ax.set_xticks(xs, [PROBE_GROUP_LABELS[g].replace("com. ratio", "com.\nratio")
                       for g in GROUP_ORDER], fontsize=5.4)
    ax.set_ylabel("mean P(choose risky)")
    ax.legend(fontsize=5.8, loc="upper right")
    return F, inf


def _p_fingerprint_pca(ax, F, inf, mv=None):
    sc, var = _pca(F)
    for ds in inf["dataset"].unique():
        c = DATASET_COLORS.get(ds, "gray")
        for arch in inf["arch"].unique():
            s = ((inf["dataset"] == ds) & (inf["arch"] == arch)).to_numpy()
            if s.any():
                ax.scatter(sc[s, 0], sc[s, 1], s=20, c=c,
                           marker=ARCH_MARKERS.get(arch, "o"), alpha=0.85,
                           linewidths=0.35, edgecolors="white", zorder=3)
    ax.set_xlabel(f"PC1 ({var[0] * 100:.0f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.0f}%)")
    if mv:
        ax.set_title(
            f"PERMANOVA F: arch {mv['permanova_arch']['pseudo_F']:.0f} vs "
            f"source {mv['permanova_dataset']['pseudo_F']:.0f}",
            fontsize=6.2, color=MUTED, pad=5)


def _p_theory_space(ax, coords, xv="gamma_pos", yv="gamma_neg"):
    good = coords[coords["theory_identifiable"]]
    for ds, sub in good.groupby("dataset"):
        c = DATASET_COLORS.get(ds, "gray")
        _ellipse(ax, sub[xv], sub[yv], c)
        for arch, ss in sub.groupby("arch"):
            ax.scatter(ss[xv], ss[yv], s=20, c=c,
                       marker=ARCH_MARKERS.get(arch, "o"), alpha=0.9,
                       linewidths=0.4, edgecolors="white", zorder=3)
    ax.set_xlabel(_pname(xv))
    ax.set_ylabel(_pname(yv))


def _p_weighting(ax, coords, col="gamma_pos"):
    p = np.linspace(0.001, 0.999, 200)
    ax.axhline(0, color=FAINT, lw=0.8, ls=":", zorder=1)
    for ds, sub in coords[coords["theory_identifiable"]].groupby("dataset"):
        c = DATASET_COLORS.get(ds, "gray")
        W = np.stack([prelec_w(p, g) - p for g in sub[col]])
        ax.fill_between(p, np.percentile(W, 25, axis=0),
                        np.percentile(W, 75, axis=0), color=c, alpha=0.15,
                        lw=0, zorder=2)
        ax.plot(p, np.median(W, axis=0), color=c, lw=1.7, zorder=3,
                label=DATASET_SHORT.get(ds, ds).replace("\n", " "))
    ax.legend(fontsize=5.4, loc="lower left")
    ax.set_xlabel("stated probability $p$")
    ax.set_ylabel(r"$w^{+}(p)-p$")
    ax.set_xlim(0, 1)
    ax.text(0.03, 0.95, "overweights rare", transform=ax.transAxes,
            fontsize=5.8, color=MUTED, va="top")


def _p_floor(ax, bt, which):
    obs_lo, obs_hi = (f"{which}_lo", f"{which}_hi")
    flo_lo, flo_hi = (f"{which}_floor_lo", f"{which}_floor_hi")
    col = AR_C if which == "arch" else DS_C
    lab = [_pname(v).split(" (")[0] for v in bt["value"]]
    y = np.arange(len(bt))
    for yi, r in zip(y, bt.itertuples()):
        f0, f1 = getattr(r, flo_lo), getattr(r, flo_hi)
        o0, o1 = getattr(r, obs_lo), getattr(r, obs_hi)
        # a real floor interval can be as narrow as 0.007 (tau); drawn to
        # scale it vanishes, so keep a minimum visible width centred on it
        # and outline it, and never let the widening cross the observed bar
        w = max(f1 - f0, 0.018)
        c0 = (f0 + f1) / 2 - w / 2
        ax.barh(yi, w, left=c0, height=0.62, color=FAINT,
                edgecolor=MUTED, linewidth=0.6, zorder=2)
        ax.plot([o0, o1], [yi, yi], color=col, lw=2.4, zorder=4,
                solid_capstyle="butt")
        ax.plot([(o0 + o1) / 2], [yi], marker="D", ms=5, color=col,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax.set_yticks(y, lab)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(r"$\omega^2$")
    n = int(bt[f"{which}_clears"].sum())
    name = "architecture" if which == "arch" else "provenance"
    ax.set_title(f"{name}: clears floor {n}/{len(bt)}", fontsize=6.6,
                 color=MUTED, pad=5)


def _p_recoverability(ax, err):
    params = [c.replace("err_", "") for c in err.columns
              if c.startswith("err_")]
    med = {p: err[f"err_{p}"].median() for p in params}
    order = sorted(params, key=lambda p: med[p])
    y = np.arange(len(order))
    cols = ["#D55E00" if med[p] > 0.30 else "#0072B2" for p in order]
    ax.barh(y, [med[p] for p in order], color=cols, height=0.6, zorder=3)
    for yi, p in zip(y, order):
        ax.plot([err[f"err_{p}"].min(), err[f"err_{p}"].max()], [yi, yi],
                color=INK, lw=0.9, alpha=0.5, zorder=4)
    ax.axvline(0.30, color=INK, lw=1.0, ls="--", zorder=5)
    ax.set_yticks(y, [_pname(p).split(" (")[0] for p in order])
    ax.invert_yaxis()
    ax.set_xlabel("relative recovery error")
    ax.set_title("orange: unrecoverable at any size", fontsize=6.2,
                 color=MUTED, pad=5)


def _p_ladder(ax, ladder):
    xs = np.arange(len(ladder))
    ax.plot(xs, ladder["r2_arch"], "o-", ms=5, lw=1.8, color=AR_C,
            label="architecture", zorder=3)
    ax.plot(xs, ladder["r2_dataset"], "o-", ms=5, lw=1.8, color=DS_C,
            label="provenance", zorder=3)
    ax.set_xticks(xs, [n.replace("\n", " ") for n in ladder["level"]],
                  fontsize=5.6, rotation=18, ha="right")
    ax.set_ylabel(r"share of variance ($R^2$)")
    ax.set_ylim(0, max(ladder[["r2_arch", "r2_dataset"]].to_numpy().max()
                       * 1.3, 0.1))
    ax.legend(fontsize=6.0, loc="lower left")


# ---------------------------------------------------------------------------
# main figures
# ---------------------------------------------------------------------------

def main1_what_is_learned(preds, run_info, probe_meta, coords, figdir,
                          mv=None):
    """From raw responses to fitted theory: the same networks read at four
    increasing levels of abstraction, which is the ladder the rest of the
    paper depends on."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.9))
        F, inf = _p_probe_profile(axes[0], preds, run_info, probe_meta)
        _tag(axes[0], "a")
        _p_fingerprint_pca(axes[1], F, inf, mv)
        _tag(axes[1], "b")
        _p_theory_space(axes[2], coords)
        _tag(axes[2], "c")
        _p_weighting(axes[3], coords)
        _tag(axes[3], "d")
        fig.tight_layout(w_pad=2.0)
        _save(fig, figdir, "main1_what_is_learned.pdf")


def _p_floor_wide(ax, bt, which, show_y=True):
    """Observed interval against the floor, for every readout, grouped by
    family. Both families are shown because the behavioural indices never
    touch the CPT fit and a reader will assume they are immune -- they are
    not."""
    obs = (f"{which}_lo", f"{which}_hi")
    flo = (f"{which}_floor_lo", f"{which}_floor_hi")
    col = AR_C if which == "arch" else DS_C
    d = bt.sort_values(["family", "value"], ascending=[False, True])
    rows, ticks, seps = [], [], []
    for fam, sub in d.groupby("family", sort=False):
        if rows:
            seps.append(len(rows))
            rows.append(None); ticks.append("")
        for r in sub.itertuples():
            rows.append(r)
            ticks.append(_pname(r.value).split(" (")[0].replace("_", " "))
    for yi, r in enumerate(rows):
        if r is None:
            continue
        f0, f1 = getattr(r, flo[0]), getattr(r, flo[1])
        o0, o1 = getattr(r, obs[0]), getattr(r, obs[1])
        w = max(f1 - f0, 0.018)
        ax.barh(yi, w, left=(f0 + f1) / 2 - w / 2, height=0.62, color=FAINT,
                edgecolor=MUTED, linewidth=0.5, zorder=2)
        ax.plot([o0, o1], [yi, yi], color=col, lw=2.2, zorder=4,
                solid_capstyle="butt")
        ax.plot([(o0 + o1) / 2], [yi], marker="D", ms=4.2, color=col,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    for sp in seps:
        ax.axhline(sp, color=FAINT, lw=0.7, ls=":", zorder=1)
    ax.set_yticks(range(len(rows)), ticks if show_y else [""] * len(rows),
                  fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(r"$\omega^2$")
    n = int(bt[f"{which}_clears"].sum())
    nm = "architecture" if which == "arch" else "provenance"
    ax.set_title(f"{nm}: clears floor {n}/{len(bt)}", fontsize=6.4,
                 color=MUTED, pad=4)


def main2_artifact_floor(bt, figdir):
    """The headline, in three panels.

    Grey spans are what the readout reports when every run shares the same
    true coordinates -- manufactured, not found. An observed interval has to
    sit clear of that to mean anything. All eleven readouts appear, including
    the behavioural indices, which never touch the CPT fit and which a reader
    would otherwise assume are immune.
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
        _p_floor_wide(axes[0], bt, "arch")
        _tag(axes[0], "a")
        _p_floor_wide(axes[1], bt, "ds", show_y=False)
        _tag(axes[1], "b")

        ax = axes[2]
        d = bt.sort_values(["family", "value"], ascending=[False, True])
        y = np.arange(len(d))
        ax.axvline(0, color=INK, lw=1.0, zorder=4)
        ax.barh(y - 0.2, d["arch_lo"] - d["arch_floor_hi"], height=0.36,
                color=AR_C, zorder=3, label="architecture")
        ax.barh(y + 0.2, d["ds_lo"] - d["ds_floor_hi"], height=0.36,
                color=DS_C, zorder=3, label="provenance")
        ax.set_yticks(y, [_pname(v).split(" (")[0].replace("_", " ")
                          for v in d["value"]], fontsize=5.6)
        ax.invert_yaxis()
        ax.set_xlabel("observed low  $-$  floor high")
        ax.set_title("positive = clears its own floor", fontsize=6.4,
                     color=MUTED, pad=4)
        ax.legend(fontsize=5.8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13), ncol=2)
        _tag(ax, "c")
        fig.text(0.5, -0.02,
                 "Grey: bootstrap 95% interval of what the readout reports "
                 "when the true coordinates are identical across runs. "
                 "Coloured: observed 95% interval.",
                 ha="center", va="top", fontsize=6.2, color=MUTED)
        fig.tight_layout(w_pad=1.8)
        _save(fig, figdir, "main2_artifact_floor.pdf")


def main3_causal(fac, sim, ctrl_base, ctrl, figdir, true_theta=None):
    """The causal version: each component of provenance manipulated on its own
    against a known agent, rather than inferred from two datasets that differ
    in every respect at once. Panel c is the observational control arriving at
    the same answer by a route that shares none of these assumptions.
    """
    from .figures_paper import FACTOR_LABELS
    COORD = ["alpha", "beta", "lam", "gamma_pos", "gamma_neg", "tau"]
    cols = [c for c in fac.columns if c.startswith("omega2_")]
    med = fac[cols].median().sort_values(ascending=False)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))

        ax = axes[0]
        y = np.arange(len(med))
        ax.barh(y, med.to_numpy(), height=0.58, zorder=2,
                color=[AR_C if k == "omega2_arch" else DS_C
                       for k in med.index], alpha=0.45)
        for yi, k in enumerate(med.index):
            v = fac[k].to_numpy()
            ax.scatter(v, np.full(len(v), yi)
                       + np.linspace(-0.18, 0.18, len(v)), s=10, c=INK,
                       alpha=0.75, linewidths=0, zorder=4)
        ax.set_yticks(y, [FACTOR_LABELS.get(k, k) for k in med.index],
                      fontsize=5.8)
        ax.invert_yaxis()
        ax.set_xlabel(r"$\omega^2$   (bar = median, dots = 11 readouts)")
        ax.set_title("blue: a component of provenance", fontsize=6.4,
                     color=MUTED, pad=4)
        _tag(ax, "a")

        ax = axes[1]
        sub = sim[sim["theory_identifiable"]] if "theory_identifiable" \
            in sim.columns else sim
        prm = [p for p in COORD if p in sub.columns]
        xs = np.arange(len(prm))
        for k, a2 in enumerate(["desc", "exp"]):
            s2 = sub[sub["agent"] == a2]
            if not len(s2):
                continue
            ratio, se = [], []
            for p in prm:
                truth = (true_theta[p] if a2 == "desc"
                         else (1.0 if p.startswith("gamma")
                               else true_theta[p]))
                ratio.append(s2[p].mean() / truth)
                se.append(s2[p].std() / max(np.sqrt(len(s2)), 1) / abs(truth))
            ax.errorbar(xs + (k - 0.5) * 0.18, ratio, yerr=se, fmt="o", ms=5.5,
                        lw=1.3, capsize=2.5, color=["#7a8b99", "#c98b4b"][k],
                        markeredgecolor="white", markeredgewidth=0.5,
                        label=f"{a2} agent")
        ax.axhline(1.0, color=INK, lw=1.1, ls="--", zorder=4)
        ax.set_xticks(xs, [_pname(p).split(" (")[0] for p in prm],
                      fontsize=6.0)
        ax.set_ylabel("recovered / generating")
        ax.legend(fontsize=5.8, loc="best")
        ax.set_title("dashed = perfect recovery of a known agent",
                     fontsize=6.2, color=MUTED, pad=4)
        _tag(ax, "b")

        ax = axes[2]
        prm3 = [p for p in ("gamma_pos", "lam", "tau") if p in ctrl_base]
        specs = [("c13k full", ctrl_base[ctrl_base["dataset"] == "c13k_exp"],
                  DATASET_COLORS["c13k_exp"]),
                 ("c13k @182", ctrl[(ctrl["dataset"] == "c13k_exp")
                                    & ctrl["subsample_n"].notna()], "#9c6644"),
                 ("reweighted", ctrl[(ctrl["dataset"] == "c13k_exp")
                                     & ctrl["reweight_to"].notna()],
                  "#7a8b99"),
                 ("CPC18 (182)",
                  ctrl_base[ctrl_base["dataset"] == "cpc18_exp"],
                  DATASET_COLORS["cpc18_exp"])]
        for j, p in enumerate(prm3):
            vals = [d[p].dropna() for _, d, _ in specs]
            if any(not len(v2) for v2 in vals):
                continue
            r0, r1 = vals[0].mean(), vals[-1].mean()
            span = abs(r1 - r0) or 1.0
            ax.axhspan(0, 1, color=FAINT, alpha=0.16, zorder=0)
            for i, ((nm, _, c), v2) in enumerate(zip(specs, vals)):
                ax.errorbar(j + (i - 1.5) * 0.17, (v2.mean() - r0) / span,
                            yerr=v2.std() / max(np.sqrt(len(v2)), 1) / span,
                            fmt="o", ms=5, color=c, capsize=2, lw=1.2,
                            markeredgecolor="white", markeredgewidth=0.5,
                            label=nm if j == 0 else None)
        ax.set_xticks(range(len(prm3)),
                      [_pname(p).split(" (")[0] for p in prm3], fontsize=6.0)
        ax.set_ylabel("position between the arms\n(0 = c13k, 1 = CPC18)")
        ax.legend(fontsize=5.4, loc="best", ncol=2)
        ax.set_title("shaded = the gap the controls must explain",
                     fontsize=6.2, color=MUTED, pad=4)
        _tag(ax, "c")
        fig.tight_layout(w_pad=2.0)
        _save(fig, figdir, "main3_causal.pdf")


def supp_controlled_extra(fac, sim, err, ladder, sens, figdir):
    """Panels cut from figures 2 and 3 to keep those to three each: the
    interaction check, which behavioural indices respond to the agent, the
    per-family factor ordering, and how little the screens matter."""
    from .figures_paper import FACTOR_LABELS
    IDX = ["rare_event_index", "certainty_index", "reflection_index",
           "loss_aversion_kappa", "ev_consistency"]
    COORD = ["alpha", "beta", "lam", "gamma_pos", "gamma_neg", "tau"]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.0))

        ax = axes[0, 0]
        idx = [c for c in IDX if c in sim.columns]
        eff, lab = [], []
        for v in idx:
            a = sim[sim["agent"] == "desc"][v].dropna()
            b = sim[sim["agent"] == "exp"][v].dropna()
            if len(a) and len(b):
                sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-9
                eff.append((b.mean() - a.mean()) / sd)
                lab.append(v.replace("_index", "").replace("_", " "))
        o = np.argsort(eff)
        y = np.arange(len(o))
        ax.barh(y, [eff[k] for k in o], height=0.6, zorder=3,
                color=[DS_C if abs(eff[k]) > 1 else FAINT for k in o])
        ax.axvline(0, color=INK, lw=1.0, zorder=4)
        for t in (-1, 1):
            ax.axvline(t, color=MUTED, lw=0.8, ls=":", zorder=2)
        ax.set_yticks(y, [lab[k] for k in o], fontsize=6.0)
        ax.set_xlabel("standardised change, exp $-$ desc agent")
        ax.set_title("which indices respond to the agent", fontsize=6.4,
                     color=MUTED, pad=4)
        _tag(ax, "a")

        ax = axes[0, 1]
        v = "rare_event_index"
        g = sim.groupby(["agent", "problems"])[v].agg(["mean", "std", "size"])
        xs = np.arange(2)
        for k, pool in enumerate(sorted(sim["problems"].unique())):
            sb = g.xs(pool, level="problems").reindex(["desc", "exp"])
            ax.errorbar(xs + (k - 0.5) * 0.12, sb["mean"],
                        yerr=sb["std"] / np.sqrt(sb["size"].clip(lower=1)),
                        fmt="o-", ms=5, lw=1.7, capsize=2.5,
                        color=BLOCK_COLORS["c13k" if pool == "c13k"
                                           else "cpc18"],
                        markeredgecolor="white", markeredgewidth=0.5,
                        label=f"{pool} problems")
        ax.axhline(0, color=FAINT, lw=0.8, ls="--", zorder=1)
        ax.set_xticks(xs, ["desc agent", "exp agent"], fontsize=6.0)
        ax.set_ylabel("rare-event index")
        ax.legend(fontsize=5.8)
        ax.margins(x=0.3)
        ax.set_title("parallel = pool does not modulate the agent effect",
                     fontsize=6.2, color=MUTED, pad=4)
        _tag(ax, "b")

        ax = axes[1, 0]
        fam = np.where(fac["value"].isin(COORD), "CPT coordinate",
                       "behavioural index")
        keys = [c for c in fac.columns if c.startswith("omega2_")]
        keys = list(fac[keys].median().sort_values(ascending=False).index)
        w = 0.38
        for k, nm in enumerate(["CPT coordinate", "behavioural index"]):
            ax.barh(np.arange(len(keys)) + (k - 0.5) * w,
                    [fac.loc[fam == nm, c].median() for c in keys],
                    height=w * 0.9, color=[DS_C, "#8fbcd9"][k], zorder=3,
                    label=nm)
        ax.set_yticks(range(len(keys)),
                      [FACTOR_LABELS.get(k, k).replace("\n", " ")
                       for k in keys], fontsize=5.6)
        ax.invert_yaxis()
        ax.set_xlabel(r"median $\omega^2$")
        ax.legend(fontsize=5.6, loc="lower right")
        ax.set_title("same ordering in both readout families", fontsize=6.2,
                     color=MUTED, pad=4)
        _tag(ax, "c")

        ax = axes[1, 1]
        if sens is not None and len(sens):
            ax.plot(sens["probe_sd_threshold"], sens["median_omega2_arch"],
                    "o-", ms=5, lw=1.7, color=AR_C, label="architecture")
            ax.plot(sens["probe_sd_threshold"], sens["median_omega2_dataset"],
                    "o-", ms=5, lw=1.7, color=DS_C, label="provenance")
            for _, r in sens.iterrows():
                ax.annotate(f"{int(r['n_runs'])}",
                            (r["probe_sd_threshold"],
                             max(r["median_omega2_arch"],
                                 r["median_omega2_dataset"])),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=5.4, color=MUTED)
            ax.set_xlabel("identifiability screen (probe SD)")
            ax.set_ylabel(r"median $\omega^2$")
            ax.legend(fontsize=5.8, loc="best")
            ax.set_title("labels: runs surviving the screen", fontsize=6.2,
                         color=MUTED, pad=4)
        _tag(ax, "d")
        fig.tight_layout(w_pad=2.0, h_pad=2.4)
        _save(fig, figdir, "supp_controlled_extra.pdf")


def main4_description_experience(probe_meta, preds, run_info, h3, grad, trend,
                                 figdir, datasets=("c13k_pair_desc",
                                                   "c13k_pair_exp")):
    """The description-experience gap as a sign flip and as the trajectory
    that produces it, the latter replicated in two designs sharing no
    problems, respondents or country."""
    m = probe_meta
    sel = (m["group"] == "ce_gain") & np.isclose(m["r"], 1.0)
    ps = sorted(m.loc[sel, "p"].unique())

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0))

        ax = axes[0]
        for ds in datasets:
            rid = [r for r in run_info[run_info["dataset"] == ds]["run_id"]
                   if r in preds]
            if not rid:
                continue
            cur = np.array([[preds[r][(sel & (m["p"] == p)).to_numpy()].mean()
                             for p in ps] for r in rid])
            mu, sd = cur.mean(0), cur.std(0)
            c = DATASET_COLORS.get(ds)
            ax.plot(ps, mu, color=c, lw=1.8, ls="--" if ds.endswith("exp")
                    else "-", zorder=3, label=DATASET_LABELS.get(ds, ds))
            ax.fill_between(ps, mu - sd, mu + sd, color=c, alpha=0.16, lw=0,
                            zorder=2)
        ax.axhline(0.5, color=FAINT, lw=0.8, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("probability of the risky gain")
        ax.set_ylabel("P(choose risky)")
        ax.legend(fontsize=5.8, loc="upper right")
        _tag(ax, "a")

        ax = axes[1]
        piv = h3.reset_index().pivot(index="arch", columns="dataset",
                                     values="mean")
        dcol = [c for c in piv.columns if c.endswith("desc")][0]
        ecol = [c for c in piv.columns if c.endswith("exp")][0]
        piv = piv.sort_values(dcol)
        y = np.arange(len(piv))
        for yi, (_, r) in zip(y, piv.iterrows()):
            ax.plot([r[ecol], r[dcol]], [yi, yi], color=FAINT, lw=1.2,
                    zorder=1)
        ax.scatter(piv[ecol], y, s=42, marker="D",
                   c=DATASET_COLORS.get(ecol), edgecolors="white",
                   linewidths=0.6, zorder=3)
        ax.scatter(piv[dcol], y, s=46, c=DATASET_COLORS.get(dcol),
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.axvline(0, color=INK, lw=0.8, zorder=2)
        ax.set_yticks(y, piv.index, fontsize=5.8)
        ax.set_xlabel("rare-event index")
        n = int(((piv[dcol] > 0) & (piv[ecol] < 0)).sum())
        ax.set_title(f"sign flip in {n}/{len(piv)} architectures",
                     fontsize=6.2, color=MUTED, pad=5)
        ax.margins(y=0.12, x=0.16)
        _tag(ax, "b")

        ax = axes[2]
        for src, sub in grad.groupby("source"):
            g = sub.groupby("block")["rare_event_index"].agg(
                ["mean", "std", "size"])
            ax.errorbar(g.index, g["mean"],
                        yerr=g["std"] / np.sqrt(g["size"].clip(lower=1)),
                        fmt="o-", ms=4.5, lw=1.6, capsize=2,
                        color=BLOCK_COLORS.get(src, "gray"),
                        markeredgecolor="white", markeredgewidth=0.5,
                        label=("CPC18 within-problem" if src == "cpc18"
                               else "c13k between-problem"))
        ax.set_xlabel("experience block")
        ax.set_ylabel("rare-event index")
        ax.set_xticks(sorted(grad["block"].unique()))
        ax.legend(fontsize=5.6, loc="best")
        _tag(ax, "c")

        ax = axes[3]
        if trend is not None and len(trend):
            piv = trend.pivot(index="value", columns="source",
                              values="mean_rho").dropna()
            sig = trend.pivot(index="value", columns="source",
                              values="p_value").reindex(piv.index)
            both = (sig["c13k"] < 0.05) & (sig["cpc18"] < 0.05)
            ax.axhline(0, color=FAINT, lw=0.8, zorder=1)
            ax.axvline(0, color=FAINT, lw=0.8, zorder=1)
            ax.plot([-1, 1], [-1, 1], color=FAINT, lw=0.8, ls=":", zorder=1)
            ax.scatter(piv["cpc18"][~both], piv["c13k"][~both], s=30,
                       facecolors="none", edgecolors=MUTED, linewidths=0.9,
                       zorder=3)
            ax.scatter(piv["cpc18"][both], piv["c13k"][both], s=40, c=INK,
                       zorder=4)
            for k, v in enumerate(sorted(piv.index[both],
                                         key=lambda z: piv.loc[z, "cpc18"])):
                ax.annotate(_pname(v).split(" (")[0].replace("_", " "),
                            (piv.loc[v, "cpc18"], piv.loc[v, "c13k"]),
                            textcoords="offset points",
                            xytext=(6, 8 if k % 2 == 0 else -13),
                            fontsize=5.4, color=INK)
            agree = int((np.sign(piv["cpc18"]) == np.sign(piv["c13k"])).sum())
            ax.set_title(f"same sign in {agree}/{len(piv)}", fontsize=6.2,
                         color=MUTED, pad=5)
            ax.set_xlabel(r"trend $\rho$, CPC18")
            ax.set_ylabel(r"trend $\rho$, c13k")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
        _tag(ax, "d")
        fig.tight_layout(w_pad=2.1)
        _save(fig, figdir, "main4_description_experience.pdf")


def main5_transfer(transfer_df, figdir, order=None, eval_only=()):
    """Transfer, and what happens on 48 problems with 240 respondents that no
    model ever trained on."""
    piv = transfer_df.groupby(["train", "test"])["completeness"] \
                     .mean().unstack()
    cols = list(order) + [c for c in eval_only if c in piv.columns]
    piv = piv.reindex(index=order, columns=cols)
    M = piv.to_numpy()
    n_sq = len(order)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3),
                                 gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
        ax = axes[0]
        im = ax.imshow(np.where(M > 0, M, np.nan), cmap="Blues", vmin=0,
                       vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isfinite(v):
                    continue
                if v <= 0:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               facecolor="#e8e8e8",
                                               edgecolor="white", lw=1,
                                               hatch="///", zorder=2))
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", zorder=3,
                        fontsize=6.6, color="white" if v > 0.62 else INK,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(piv.columns)),
                      [DATASET_SHORT.get(c, c) for c in piv.columns],
                      fontsize=5.6)
        ax.set_yticks(range(len(piv.index)),
                      [DATASET_LABELS.get(i, i) for i in piv.index],
                      fontsize=5.8)
        ax.axvline(n_sq - 0.5, color=INK, lw=1.2, zorder=5)
        ax.set_xlabel("evaluated on")
        ax.set_ylabel("trained on")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("completeness", fontsize=6.2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=2)
        ax.set_title("right of line: never trained on\nhatched: worse than "
                     "the mean", fontsize=6.0, color=MUTED, pad=4)
        _tag(ax, "a")

        ax = axes[1]
        names = list(piv.index)
        pairs, gaps, cols2 = [], [], []
        for i in range(n_sq):
            for j in range(i + 1, n_sq):
                d = M[i, j] - M[j, i]
                lead = names[i] if d > 0 else names[j]
                pairs.append(f"{DATASET_SHORT.get(names[i], names[i])}\nvs "
                             f"{DATASET_SHORT.get(names[j], names[j])}"
                             .replace("\n", " ", 1))
                gaps.append(abs(d))
                cols2.append(DATASET_COLORS.get(lead, "gray"))
        o = np.argsort(gaps)
        y = np.arange(len(o))
        ax.barh(y, [gaps[k] for k in o], color=[cols2[k] for k in o],
                height=0.62, zorder=3)
        ax.set_yticks(y, [pairs[k].replace("\n", " ") for k in o],
                      fontsize=5.2)
        ax.set_xlabel("transfer asymmetry")
        ax.set_title("coloured by the arm that exports better", fontsize=6.0,
                     color=MUTED, pad=5)
        _tag(ax, "b")

        ax = axes[2]
        own = np.array([M[i, i] for i in range(n_sq)])
        new = np.nanmean(M[:, n_sq:], axis=1)
        y = np.arange(n_sq)
        for yi, (o_, n_) in enumerate(zip(own, new)):
            ax.plot([o_, n_], [yi, yi], color=FAINT, lw=1.3, zorder=1)
        ax.scatter(own, y, s=44, c=[DATASET_COLORS.get(n, "gray")
                                    for n in names], zorder=3,
                   edgecolors="white", linewidths=0.6, label="own test set")
        ax.scatter(new, y, s=44, marker="X",
                   c=[DATASET_COLORS.get(n, "gray") for n in names], zorder=3,
                   edgecolors="white", linewidths=0.6, label="unseen problems")
        ax.axvline(0, color=INK, lw=0.9, zorder=2)
        ax.set_yticks(y, [DATASET_SHORT.get(n, n).replace("\n", " ")
                          for n in names], fontsize=5.8)
        ax.invert_yaxis()
        ax.set_xlabel("completeness")
        ax.legend(fontsize=5.8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=2)
        _tag(ax, "c")
        fig.tight_layout(w_pad=2.1)
        _save(fig, figdir, "main5_transfer.pdf")


def main6_reliability(coords, err, figdir, thr=0.02):
    """When the readout can be trusted: how flat the behaviour is, how often a
    parameter is pinned to a bound, and how the error falls (or does not) as
    the arm grows."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))

        ax = axes[0]
        archs = sorted(coords["arch"].unique())
        dss = [d for d in DATASET_LABELS if d in coords["dataset"].unique()]
        for i, ds in enumerate(dss):
            for j, a in enumerate(archs):
                v = coords[(coords["dataset"] == ds)
                           & (coords["arch"] == a)]["probe_sd"]
                if len(v):
                    ax.scatter(np.full(len(v),
                                       j + (i - (len(dss) - 1) / 2) * 0.17),
                               v, s=6, c=DATASET_COLORS.get(ds, "gray"),
                               alpha=0.7, linewidths=0, zorder=3)
        ax.axhline(thr, color=INK, lw=1.0, ls="--", zorder=4)
        ax.set_yscale("log")
        ax.set_xticks(range(len(archs)), archs, rotation=35, ha="right",
                      fontsize=5.8)
        ax.set_ylabel("probe-prediction SD (log)")
        ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=4,
                                      markerfacecolor=DATASET_COLORS[d],
                                      markeredgecolor="none",
                                      label=DATASET_LABELS.get(d, d))
                           for d in dss], fontsize=5.4, loc="lower left")
        _tag(ax, "a")

        ax = axes[1]
        piv = (coords.groupby(["dataset", "arch"])["at_bound"].mean()
               .unstack() * 100).reindex(dss)
        im = ax.imshow(piv.to_numpy(), cmap="Reds", vmin=0, vmax=100)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.to_numpy()[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6.2, color="white" if v > 55 else INK)
        ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=35,
                      ha="right", fontsize=5.8)
        ax.set_yticks(range(len(piv.index)),
                      [DATASET_SHORT.get(i, i).replace("\n", " ")
                       for i in piv.index], fontsize=5.8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("% of fits on a bound", fontsize=6.0)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=2)
        _tag(ax, "b")

        ax = axes[2]
        params = [c.replace("err_", "") for c in err.columns
                  if c.startswith("err_")]
        big = err[err["n_subj"] == err["n_subj"].max()]
        cmap = plt.get_cmap("viridis")
        med = {p: err[f"err_{p}"].median() for p in params}
        order = sorted(params, key=lambda p: med[p])
        for k, p in enumerate(order):
            ax.plot(big["n_problems"], big[f"err_{p}"], "o-", ms=3.5, lw=1.4,
                    color=cmap(k / max(len(order) - 1, 1)),
                    label=_pname(p).split(" (")[0], zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("problems in the arm")
        ax.set_ylabel("relative recovery error")
        ax.legend(fontsize=5.6, ncol=2, loc="best")
        ax.set_title("two parameters never improve", fontsize=6.2,
                     color=MUTED, pad=5)
        _tag(ax, "c")
        fig.tight_layout(w_pad=2.1)
        _save(fig, figdir, "main6_reliability.pdf")


# ---------------------------------------------------------------------------
# Supplementary: robustness of the floor-clearing result (11_robustness.py)
# ---------------------------------------------------------------------------

FLOOR_DEF_SHORT = {
    "problem pool": "pool",
    "arm size": "size",
    "respondents": "respondents",
    "largest component": "max of 3",
    "composite (pool x size x respondents)": "composite",
    "sufficiency suite": "sufficiency",
    "controlled suite": "controlled",
}


def supp_robustness(fd, fs, js, sym, pca, figdir, pca_meta=None):
    """Four panels a referee would ask for, from results/tables/robustness_*
    and symbolic_*:
      a  provenance: observed interval against every floor definition
      b  how many read-outs clear, per floor definition and test (both factors)
      c  leave-one-architecture-out
      d  the distilled expressions as functions: PCA + PERMANOVA
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.7),
                                 gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))

        # --- a: provenance observed vs each floor definition --------------
        ax = axes[0]
        prov = fd[fd["factor"] == "provenance"]
        defs = [d for d in FLOOR_DEF_SHORT if d in prov["floor_def"].unique()
                and d not in ("sufficiency suite", "controlled suite")]
        order = (prov[prov["floor_def"] == defs[0]]
                 .sort_values(["family", "value"], ascending=[False, True]))
        vals = order["value"].tolist()
        greys = ["#d9d9d9", "#bdbdbd", "#a0a0a0", "#7a7a7a", "#4d4d4d"]
        for yi, v in enumerate(vals):
            r0 = order[order["value"] == v].iloc[0]
            for k, dname in enumerate(defs):
                r = prov[(prov["floor_def"] == dname)
                         & (prov["value"] == v)].iloc[0]
                off = (k - (len(defs) - 1) / 2) * 0.13
                ax.plot([r["floor_lo"], r["floor_hi"]], [yi + off, yi + off],
                        color=greys[k % len(greys)], lw=1.6, zorder=2,
                        solid_capstyle="butt",
                        label=FLOOR_DEF_SHORT[dname] if yi == 0 else None)
            ax.plot([r0["obs_lo"], r0["obs_hi"]], [yi, yi], color=DS_C,
                    lw=2.2, zorder=4, solid_capstyle="butt",
                    label="observed (provenance)" if yi == 0 else None)
            ax.plot([r0["obs"]], [yi], marker="D", ms=4, color=DS_C,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        ax.set_yticks(range(len(vals)),
                      [_pname(v).split(" (")[0].replace("_", " ")
                       for v in vals], fontsize=5.4)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(r"$\omega^2$")
        ax.set_title("provenance against five floor definitions",
                     fontsize=6.4, color=MUTED, pad=4)
        ax.legend(fontsize=5.2, loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, -0.16), handlelength=1.6,
                  columnspacing=1.0)
        _tag(ax, "a")

        # --- b: counts per definition and test ----------------------------
        ax = axes[1]
        tests = [("clears_interval", "interval rule"),
                 ("clears_holm", "Holm-adjusted p < .05")]
        rows = []
        for fac in ["provenance", "architecture"]:
            for _, r in fs[fs["factor"] == fac].iterrows():
                rows.append((fac, FLOOR_DEF_SHORT.get(r["floor_def"],
                                                      r["floor_def"]),
                             r[tests[0][0]], r[tests[1][0]], r["n_readouts"]))
        y = np.arange(len(rows))
        for yi, (fac, dname, c_int, c_holm, n) in enumerate(rows):
            col = DS_C if fac == "provenance" else AR_C
            ax.barh(yi - 0.18, c_int, height=0.34, color=col, alpha=0.9,
                    zorder=3)
            ax.barh(yi + 0.18, c_holm, height=0.34, color=col, alpha=0.45,
                    zorder=3)
        ax.set_yticks(y, [f"{'prov.' if f == 'provenance' else 'arch.'}: {d}"
                          for f, d, *_ in rows], fontsize=5.6)
        for lab, (f, *_) in zip(ax.get_yticklabels(), rows):
            lab.set_color(DS_C if f == "provenance" else AR_C)
        n_tot = int(fs["n_readouts"].max())
        ax.axvline(n_tot, color=INK, lw=0.8, ls=":", zorder=4)
        ax.set_xlim(0, n_tot + 0.6)
        ax.set_xlabel(f"read-outs clearing their floor (of {n_tot})")
        ax.invert_yaxis()
        # family separators / labels
        n_prov = sum(1 for r in rows if r[0] == "provenance")
        ax.axhline(n_prov - 0.5, color=FAINT, lw=0.7, ls=":")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=INK, alpha=0.9, label=tests[0][1]),
                           Patch(color=INK, alpha=0.4, label=tests[1][1])],
                  fontsize=5.4, loc="lower right", frameon=False)
        ax.set_title("floor definition x test", fontsize=6.4, color=MUTED,
                     pad=4)
        _tag(ax, "b")

        # --- c: leave-one-architecture-out --------------------------------
        ax = axes[2]
        j = js.copy()
        labels = ["all" if l == "none" else l for l in j["left_out"]]
        x = np.arange(len(j))
        ax.plot(x, j["provenance_clears"], marker="o", ms=4.5, color=DS_C,
                lw=1.2, zorder=4, label="provenance, pool floor")
        ax.plot(x, j["provenance_clears_composite"], marker="o", ms=4.5,
                color=DS_C, lw=1.2, ls="--", mfc="white", zorder=4,
                label="provenance, composite floor")
        ax.plot(x, j["architecture_clears"], marker="s", ms=4.5, color=AR_C,
                lw=1.2, zorder=4, label="architecture")
        ax.axhline(j["n_readouts"].max(), color=INK, lw=0.8, ls=":")
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=5.4)
        ax.set_ylim(-0.4, j["n_readouts"].max() + 0.8)
        ax.set_ylabel("read-outs clearing their floor")
        ax.set_xlabel("architecture left out")
        ax.legend(fontsize=5.3, loc="center right", frameon=False)
        ax.set_title("no single architecture carries the result",
                     fontsize=6.4, color=MUTED, pad=4)
        _tag(ax, "c")

        # --- d: symbolic expressions as functions -------------------------
        ax = axes[3]
        for ds, sub in pca.groupby("dataset"):
            for ar, s2 in sub.groupby("arch"):
                ax.scatter(s2["pc1"], s2["pc2"], s=22,
                           c=DATASET_COLORS.get(ds, MUTED),
                           marker=ARCH_MARKERS.get(ar, "o"), alpha=0.85,
                           linewidths=0.4, edgecolors="white", zorder=3)
        for ds in pca["dataset"].unique():
            sub = pca[pca["dataset"] == ds]
            if len(sub) > 2:
                _ellipse(ax, sub["pc1"].to_numpy(), sub["pc2"].to_numpy(),
                         DATASET_COLORS.get(ds, MUTED))
        ev = (pca_meta or {}).get("explained", [np.nan, np.nan])
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({100 * ev[1]:.0f}%)")
        fsym = sym[sym["view"].str.startswith("functional")].iloc[0]
        ax.set_title(
            "distilled CE(x, p) as functions:  PERMANOVA "
            f"$F$ provenance {fsym['permanova_F_dataset']:.1f}, "
            f"architecture {fsym['permanova_F_arch']:.1f}",
            fontsize=6.0, color=MUTED, pad=4)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color=DATASET_COLORS[d], marker="o", ls="",
                    ms=4, label=DATASET_SHORT.get(d, d))
             for d in ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]
             if d in pca["dataset"].unique()]
        ax.legend(handles=h, fontsize=5.2, loc="best", frameon=False)
        _tag(ax, "d")

        fig.tight_layout(w_pad=1.6)
        _save(fig, figdir, "supp_robustness.pdf")
