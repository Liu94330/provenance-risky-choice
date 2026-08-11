"""Figures 5-10: the panels that need more than the headline tables.

fig5   behavioural fingerprints -- probe-group profiles + PCA embedding
fig6   the CPT functions the networks imply -- w(p) and v(x)
fig7   provenance or sample size? the size-matched / reweighted controls
fig8   identifiability diagnostics -- why 44/240 coordinate fits are dropped
fig9   symbolic distillation -- accuracy/complexity front and operator usage
fig10  population arm (two labs, same problems) + predictive calibration

Shares palette, rc block and helpers with src.figures so the whole set reads
as one system.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures import (ARCH_MARKERS, DATASET_COLORS, DATASET_LABELS,
                      DATASET_SHORT, FAINT, INK, MUTED, RC, _panel_tag,
                      _pname, _save)
from .problems import SCALE, prelec_w

PROBE_GROUP_LABELS = {
    "ce_gain": "CE\ngains",
    "ce_loss": "CE\nlosses",
    "mixed": "mixed\ngambles",
    "la_ladder": "loss-av.\nladder",
    "ladder_gain": "com. ratio\ngains",
    "ladder_loss": "com. ratio\nlosses",
}
GROUP_ORDER = ["ce_gain", "ce_loss", "mixed", "la_ladder",
               "ladder_gain", "ladder_loss"]


def _pca(X, k=2):
    """Plain PCA via SVD on column-standardised X. Returns scores and the
    fraction of variance each retained component carries."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    return U[:, :k] * S[:k], var[:k]


def _legend_ds_arch(ax, datasets, archs, **kw):
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5,
                          markerfacecolor=DATASET_COLORS.get(d, "gray"),
                          markeredgecolor=INK, markeredgewidth=0.4,
                          label=DATASET_LABELS.get(d, d)) for d in datasets]
    handles += [plt.Line2D([], [], ls="", marker=ARCH_MARKERS.get(a, "o"),
                           ms=5, markerfacecolor="none", markeredgecolor=INK,
                           markeredgewidth=0.8, label=a) for a in archs]
    ax.legend(handles=handles, **kw)


# ---------------------------------------------------------------------------

def fig5_fingerprints(preds, run_info, probe_meta, figdir, mv=None):
    """(a) mean response per probe family, one line per run, coloured by
    dataset -- the raw material the multivariate tests run on.
    (b) PCA of the full 1,742-d fingerprint, which is what PERMANOVA and the
    silhouettes actually see."""
    rid = [r for r in run_info["run_id"] if r in preds]
    info = run_info.set_index("run_id").loc[rid]
    F = np.stack([preds[r] for r in rid])

    gmask = {g: (probe_meta["group"] == g).to_numpy() for g in GROUP_ORDER}
    prof = np.stack([[F[i][gmask[g]].mean() for g in GROUP_ORDER]
                     for i in range(len(F))])

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3),
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
        ax = axes[0]
        xs = np.arange(len(GROUP_ORDER))
        for ds in info["dataset"].unique():
            sel = (info["dataset"] == ds).to_numpy()
            col = DATASET_COLORS.get(ds, "gray")
            for row in prof[sel]:
                ax.plot(xs, row, color=col, lw=0.7, alpha=0.28, zorder=2)
            ax.plot(xs, prof[sel].mean(0), color=col, lw=2.2, zorder=4,
                    label=DATASET_LABELS.get(ds, ds))
        ax.axhline(0.5, color=FAINT, lw=0.8, ls="--", zorder=1)
        ax.set_xticks(xs, [PROBE_GROUP_LABELS[g] for g in GROUP_ORDER],
                      fontsize=6.2)
        ax.set_ylabel("mean P(choose risky)")
        ax.legend(fontsize=6.5, loc="upper right")
        _panel_tag(ax, "a")

        ax = axes[1]
        sc, var = _pca(F)
        for ds in info["dataset"].unique():
            col = DATASET_COLORS.get(ds, "gray")
            for arch in info["arch"].unique():
                s = ((info["dataset"] == ds) & (info["arch"] == arch)).to_numpy()
                if not s.any():
                    continue
                ax.scatter(sc[s, 0], sc[s, 1], s=24, c=col,
                           marker=ARCH_MARKERS.get(arch, "o"), alpha=0.85,
                           linewidths=0.4, edgecolors="white", zorder=3)
        ax.set_xlabel(f"PC1 ({var[0] * 100:.0f}% var)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.0f}% var)")
        if mv:
            ax.set_title(
                f"PERMANOVA  dataset F={mv['permanova_dataset']['pseudo_F']:.1f}"
                f"   arch F={mv['permanova_arch']['pseudo_F']:.1f}\n"
                f"silhouette  dataset {mv['silhouette_by_dataset']:.3f}"
                f"   arch {mv['silhouette_by_arch']:.3f}",
                fontsize=6.8, color=MUTED, pad=6)
        _legend_ds_arch(ax, [], list(info["arch"].unique()), fontsize=6,
                        loc="best", ncol=2, handletextpad=0.3,
                        columnspacing=0.8)
        _panel_tag(ax, "b")
        fig.tight_layout(w_pad=2.2)
        _save(fig, figdir, "fig5_fingerprints.pdf")


# ---------------------------------------------------------------------------

def fig6_cpt_functions(coords, figdir):
    """The weighting and value functions the fitted coordinates imply -- how
    these parameters are actually read in the decision literature.

    The weighting panels plot w(p) - p rather than w(p). Plotted raw, all four
    arms sit within a line width of each other and of the diagonal, and the
    panel says nothing; against the diagonal the inverse-S shape and the gap
    between arms are both legible. No ground truth is drawn here: these
    networks are trained on human data, and the synthetic agent's parameters
    are not a target they should hit (that comparison lives in fig4b)."""
    p = np.linspace(0.001, 0.999, 200)
    xs = np.linspace(-1.0, 1.0, 200)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))

        for ax, (col, lab, tag) in zip(
                axes[:2],
                [("gamma_pos", r"$w^{+}(p)-p$   gains", "a"),
                 ("gamma_neg", r"$w^{-}(p)-p$   losses", "b")]):
            ax.axhline(0, color=FAINT, lw=0.8, ls=":", zorder=1)
            for ds, sub in coords.groupby("dataset"):
                c = DATASET_COLORS.get(ds, "gray")
                W = np.stack([prelec_w(p, g) - p for g in sub[col]])
                ax.fill_between(p, np.percentile(W, 25, axis=0),
                                np.percentile(W, 75, axis=0), color=c,
                                alpha=0.15, lw=0, zorder=2)
                ax.plot(p, np.median(W, axis=0), color=c, lw=1.8, zorder=3,
                        label=DATASET_LABELS.get(ds, ds))
            ax.set_xlabel("stated probability $p$")
            ax.set_ylabel(lab)
            ax.set_xlim(0, 1)
            _panel_tag(ax, tag)
        axes[0].text(0.03, 0.96, "overweighting", transform=axes[0].transAxes,
                     fontsize=6.2, color=MUTED, va="top")
        axes[0].text(0.03, 0.04, "underweighting",
                     transform=axes[0].transAxes, fontsize=6.2, color=MUTED,
                     va="bottom")

        ax = axes[2]
        ax.axhline(0, color=FAINT, lw=0.7, zorder=1)
        ax.axvline(0, color=FAINT, lw=0.7, zorder=1)
        for ds, sub in coords.groupby("dataset"):
            c = DATASET_COLORS.get(ds, "gray")
            V = np.stack([np.where(xs >= 0, np.abs(xs) ** a,
                                   -lm * np.abs(xs) ** b)
                          for a, b, lm in zip(sub["alpha"], sub["beta"],
                                              sub["lam"])])
            ax.fill_between(xs, np.percentile(V, 25, axis=0),
                            np.percentile(V, 75, axis=0), color=c, alpha=0.15,
                            lw=0, zorder=2)
            ax.plot(xs, np.median(V, axis=0), color=c, lw=1.8, zorder=3)
        ax.set_xlabel(f"outcome (units of {SCALE:.0f})")
        ax.set_ylabel(r"$v(x)$")
        ax.set_title("bands are IQR across runs", fontsize=6.5, color=MUTED,
                     pad=5)
        _panel_tag(ax, "c")

        h, lb = axes[0].get_legend_handles_labels()
        fig.legend(h, lb, loc="upper center", ncol=len(lb),
                   bbox_to_anchor=(0.5, 1.09), fontsize=6.8,
                   columnspacing=1.2, handletextpad=0.45)
        fig.tight_layout(w_pad=2.0)
        _save(fig, figdir, "fig6_cpt_functions.pdf")


# ---------------------------------------------------------------------------

def fig7_controls(base, ctrl, figdir, params=("gamma_pos", "lam", "tau")):
    """Is the c13k/CPC18 gap provenance or just 9,831 vs 182 problems?
    Sub-sampling c13k down to CPC18's size, and reweighting it to CPC18's
    feature distribution, move the coordinates part of the way -- how far is
    the answer."""
    rows = []
    for prm in params:
        def grab(df, dataset, tag):
            v = df[prm].dropna()
            if len(v):
                rows.append(dict(param=prm, arm=tag, dataset=dataset, n=len(v),
                                 mean=v.mean(),
                                 sem=v.std() / max(np.sqrt(len(v)), 1)))
        grab(base[base["dataset"] == "c13k_exp"], "c13k_exp", "c13k full")
        grab(ctrl[(ctrl["dataset"] == "c13k_exp")
                  & ctrl["subsample_n"].notna()], "c13k_exp", "c13k @182")
        grab(ctrl[(ctrl["dataset"] == "c13k_exp")
                  & ctrl["reweight_to"].notna()], "c13k_exp", "c13k reweighted")
        grab(base[base["dataset"] == "cpc18_exp"], "cpc18_exp", "CPC18 (182)")
    d = pd.DataFrame(rows)
    arms = ["c13k full", "c13k @182", "c13k reweighted", "CPC18 (182)"]
    cols = {"c13k full": DATASET_COLORS["c13k_exp"],
            "c13k @182": "#9c6644", "c13k reweighted": "#7a8b99",
            "CPC18 (182)": DATASET_COLORS["cpc18_exp"]}

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(params), figsize=(7.8, 3.0))
        for ax, prm, tag in zip(np.atleast_1d(axes), params, "abcdef"):
            sub = d[d["param"] == prm].set_index("arm").reindex(arms)
            xpos = np.arange(len(arms))
            ax.errorbar(xpos, sub["mean"], yerr=sub["sem"], fmt="none",
                        ecolor=MUTED, lw=1.1, capsize=2.5, zorder=2)
            ax.scatter(xpos, sub["mean"], s=60,
                       c=[cols[a] for a in arms], edgecolors="white",
                       linewidths=0.7, zorder=3)
            lo, hi = sub["mean"].iloc[0], sub["mean"].iloc[-1]
            ax.axhspan(min(lo, hi), max(lo, hi), color=FAINT, alpha=0.16,
                       zorder=0)
            ax.set_xticks(xpos, [f"{a}\nn={int(n)}" for a, n
                                 in zip(arms, sub["n"].fillna(0))],
                          rotation=35, ha="right", fontsize=6.0)
            ax.set_ylabel(_pname(prm))
            ax.margins(x=0.16)
            _panel_tag(ax, tag)
        fig.text(0.5, -0.04,
                 "Band spans the two full-size endpoints. A control landing "
                 "inside it is the share of the gap explained by sample size "
                 "or feature mix;\nlanding outside means that factor does not "
                 "account for the gap. Error bars are SEM over surviving runs.",
                 ha="center", va="top", fontsize=6.4, color=MUTED)
        fig.tight_layout(w_pad=2.4)
        _save(fig, figdir, "fig7_controls.pdf")


# ---------------------------------------------------------------------------

def fig8_identifiability(coords, figdir, thr=0.02):
    """Why 44 of 240 coordinate fits are excluded, in three views: how flat
    the behaviour is, how often a parameter lands on its bound, and what that
    does to the nominal fit statistic."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0),
                                 gridspec_kw=dict(width_ratios=[1.15, 1, 1]))

        ax = axes[0]
        archs = sorted(coords["arch"].unique())
        dss = [d for d in DATASET_LABELS if d in coords["dataset"].unique()]
        for i, ds in enumerate(dss):
            for j, a in enumerate(archs):
                v = coords[(coords["dataset"] == ds)
                           & (coords["arch"] == a)]["probe_sd"]
                if not len(v):
                    continue
                xj = j + (i - (len(dss) - 1) / 2) * 0.17
                ax.scatter(np.full(len(v), xj), v, s=7,
                           c=DATASET_COLORS.get(ds, "gray"), alpha=0.7,
                           linewidths=0, zorder=3)
        ax.axhline(thr, color=INK, lw=1.0, ls="--", zorder=4)
        ax.text(len(archs) - 0.4, thr * 1.3, f"screen  {thr}", fontsize=6.2,
                color=INK, ha="right")
        ax.set_yscale("log")
        ax.set_xticks(range(len(archs)), archs, rotation=35, ha="right",
                      fontsize=6.3)
        ax.set_ylabel("probe-prediction SD  (log)")
        ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=4,
                                      markerfacecolor=DATASET_COLORS[d],
                                      markeredgecolor="none",
                                      label=DATASET_LABELS.get(d, d))
                           for d in dss],
                  fontsize=5.8, loc="lower left", handletextpad=0.3,
                  borderaxespad=0.2)
        _panel_tag(ax, "a")

        ax = axes[1]
        piv = (coords.groupby(["dataset", "arch"])["at_bound"].mean()
               .unstack() * 100).reindex(dss)
        im = ax.imshow(piv.to_numpy(), cmap="Reds", vmin=0, vmax=100)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.to_numpy()[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 55 else INK)
        ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=35,
                      ha="right", fontsize=6.3)
        ax.set_yticks(range(len(piv.index)),
                      [DATASET_LABELS.get(i, i) for i in piv.index],
                      fontsize=6.3)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("% of fits on a bound", fontsize=6.5)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=2)
        _panel_tag(ax, "b")

        ax = axes[2]
        ok = coords["identifiable"]
        ax.scatter(coords.loc[~ok, "probe_sd"], coords.loc[~ok, "cpt_fit_r2_raw"],
                   s=12, facecolors="none", edgecolors=FAINT, linewidths=0.6,
                   zorder=2, label="screened out")
        ax.scatter(coords.loc[ok, "probe_sd"], coords.loc[ok, "cpt_fit_r2_raw"],
                   s=12, c=[DATASET_COLORS.get(d, "gray")
                            for d in coords.loc[ok, "dataset"]],
                   alpha=0.8, linewidths=0, zorder=3)
        ax.axvline(thr, color=INK, lw=1.0, ls="--", zorder=4)
        ax.set_xscale("log")
        ax.set_ylim(-20, 1.3)
        ax.set_xlabel("probe-prediction SD  (log)")
        ax.set_ylabel(r"nominal fit $r^2$")
        ax.legend(fontsize=6.2, loc="lower right")
        ax.set_title(r"$r^2=1-\mathrm{mse}/\mathrm{var}$ explodes as var$\to$0",
                     fontsize=6.5, color=MUTED, pad=5)
        _panel_tag(ax, "c")
        fig.tight_layout(w_pad=2.0)
        _save(fig, figdir, "fig8_identifiability.pdf")


# ---------------------------------------------------------------------------

def fig9_symbolic(sr, figdir):
    """Symbolic distillation of the certainty-equivalent surface: how well a
    closed form recovers each network, and whether the recovered expressions
    group by provenance or by architecture."""
    sr = sr[np.isfinite(sr["ce_r2"])].copy()

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))

        ax = axes[0]
        for ds, sub in sr.groupby("dataset"):
            if ds not in DATASET_COLORS:
                continue
            ax.scatter(sub["ce_complexity"], sub["ce_r2"], s=20,
                       c=DATASET_COLORS[ds], alpha=0.7, linewidths=0.3,
                       edgecolors="white", zorder=3)
        ax.set_xlabel("expression complexity (nodes)")
        ax.set_ylabel(r"CE-surface $r^2$")
        ax.set_ylim(min(0.0, sr["ce_r2"].quantile(0.02)), 1.02)
        _panel_tag(ax, "a")

        ax = axes[1]
        # medians are ~0.99 everywhere, so bars of the median are four full
        # bars and say nothing. The arms differ in their lower tail -- how
        # often the closed form fails to recover the network at all -- so
        # plot median with a whisker down to the 10th percentile.
        order = [d for d in DATASET_LABELS if d in sr["dataset"].unique()]
        ypos = np.arange(len(order))
        for k, dom in enumerate(["gain", "loss"]):
            off = (k - 0.5) * 0.30
            med, lo = [], []
            for ds in order:
                v = sr[(sr["dataset"] == ds) & (sr["domain"] == dom)]["ce_r2"]
                med.append(v.median() if len(v) else np.nan)
                lo.append(v.quantile(0.10) if len(v) else np.nan)
            col = ["#0072B2", "#D55E00"][k]
            for y, m, l in zip(ypos + off, med, lo):
                ax.plot([l, m], [y, y], color=col, lw=1.4, alpha=0.55,
                        solid_capstyle="round", zorder=2)
            ax.scatter(med, ypos + off, s=26, c=col, zorder=3,
                       edgecolors="white", linewidths=0.5, label=dom)
            ax.scatter(lo, ypos + off, s=9, c=col, zorder=3, alpha=0.75)
        ax.set_yticks(ypos, [DATASET_LABELS.get(i, i) for i in order],
                      fontsize=6.3)
        ax.invert_yaxis()
        ax.set_xlabel(r"$r^2$:  median $\bullet$  and 10th pct")
        ax.set_xlim(right=1.04)
        ax.legend(fontsize=6.3, loc="upper left", ncol=2,
                  borderaxespad=0.3)
        _panel_tag(ax, "b")

        ax = axes[2]
        ops = ["sqrt", "mul", "div", "add", "sub"]
        rows = []
        for ds, sub in sr.groupby("dataset"):
            if ds not in DATASET_COLORS:
                continue
            txt = " ".join(sub["ce_expression"].astype(str))
            n = max(len(sub), 1)
            rows.append(pd.Series({o: txt.count(o + "(") / n for o in ops},
                                  name=ds))
        use = pd.DataFrame(rows).reindex([d for d in DATASET_LABELS
                                          if d in [r.name for r in rows]])
        im = ax.imshow(use.to_numpy(), cmap="Blues", aspect="auto")
        for i in range(use.shape[0]):
            for j in range(use.shape[1]):
                v = use.to_numpy()[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.4,
                        color="white" if v > use.to_numpy().max() * 0.6
                        else INK)
        ax.set_xticks(range(len(ops)), ops, fontsize=6.4)
        ax.set_yticks(range(len(use)),
                      [DATASET_LABELS.get(i, i) for i in use.index],
                      fontsize=6.3)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title("operators per expression", fontsize=6.8, color=MUTED,
                     pad=5)
        _panel_tag(ax, "c")
        fig.tight_layout(w_pad=2.2)
        _save(fig, figdir, "fig9_symbolic.pdf")


# ---------------------------------------------------------------------------

BLOCK_COLORS = {"cpc18": "#0072B2", "c13k": "#D55E00"}

FACTOR_LABELS = {
    "omega2_agent": "generating agent\n(real behaviour)",
    "omega2_problems": "problem pool",
    "omega2_n_problems": "arm size",
    "omega2_n_subj": "respondents\nper problem",
    "omega2_arch": "architecture",
}


def fig14_sufficiency(err, figdir, h1=None, tol_grid=(0.40, 0.30, 0.20, 0.15)):
    """Can this readout recover a theory it is known to be looking at?

    Every other result assumes the fitted CPT coordinates mean something. This
    panel set tests that assumption directly, against a KNOWN agent with only
    the design varying, and the answer changes how the rest should be read:

    (a) error per parameter as the arm grows. Two parameters never improve.
    (b) recoverability ranking. tau and beta sit near 40% error regardless of
        how much data they are given, which is a structural identifiability
        limit (tau trades off against the value scale, beta against lambda on
        the loss side), not a sampling problem.
    (c) the consequence: restricting H1 to the parameters this method can
        actually recover -- a screen measured on synthetic arms and therefore
        independent of the real-data analysis -- flips which factor leads,
        twice. That is the honest conclusion: on the CPT coordinates the
        evidence does not resolve the question either way.
    """
    ecols = [c for c in err.columns if c.startswith("err_")]
    params = [c.replace("err_", "") for c in ecols]
    med = {p: err[f"err_{p}"].median() for p in params}
    order = sorted(params, key=lambda p: med[p])

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0),
                                 gridspec_kw=dict(width_ratios=[1.1, 1, 1.05]))

        ax = axes[0]
        big = err[err["n_subj"] == err["n_subj"].max()]
        cmap = plt.get_cmap("viridis")
        for k, p in enumerate(order):
            ax.plot(big["n_problems"], big[f"err_{p}"], "o-", ms=4, lw=1.5,
                    color=cmap(k / max(len(order) - 1, 1)),
                    label=_pname(p).split(" (")[0], zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("problems in the arm")
        ax.set_ylabel("relative recovery error")
        ax.set_title(f"at {int(err['n_subj'].max())} respondents/problem",
                     fontsize=6.5, color=MUTED, pad=5)
        ax.legend(fontsize=6.0, ncol=2, loc="best")
        _panel_tag(ax, "a")

        ax = axes[1]
        ypos = np.arange(len(order))
        vals = [med[p] for p in order]
        cols = ["#D55E00" if v > 0.30 else "#0072B2" for v in vals]
        ax.barh(ypos, vals, color=cols, height=0.6, zorder=3)
        for y, p in zip(ypos, order):
            lo, hi = err[f"err_{p}"].min(), err[f"err_{p}"].max()
            ax.plot([lo, hi], [y, y], color=INK, lw=0.9, alpha=0.5, zorder=4)
        ax.axvline(0.30, color=INK, lw=1.0, ls="--", zorder=5)
        ax.set_yticks(ypos, [_pname(p).split(" (")[0] for p in order])
        ax.invert_yaxis()
        ax.set_xlabel("median relative error")
        ax.set_title("orange: not recoverable at any size tested",
                     fontsize=6.3, color=MUTED, pad=5)
        _panel_tag(ax, "b")

        ax = axes[2]
        if h1 is not None and len(h1):
            h = h1.copy()
            h["rec"] = h["value"].map(med)
            h = h[h["rec"].notna()]
            xs, ds, ar, ns = [], [], [], []
            for t in tol_grid:
                sub = h[h["rec"] <= t]
                if len(sub) < 2:
                    continue
                xs.append(t)
                ds.append(sub["omega2_dataset"].median())
                ar.append(sub["omega2_arch"].median())
                ns.append(len(sub))
            if xs:
                ax.plot(xs, ds, "o-", ms=5, lw=1.8, color="#0072B2",
                        label="dataset provenance", zorder=3)
                ax.plot(xs, ar, "o-", ms=5, lw=1.8, color="#D55E00",
                        label="architecture", zorder=3)
                for x, d_, a_, n in zip(xs, ds, ar, ns):
                    ax.annotate(f"{n}", (x, max(d_, a_)),
                                textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=5.8, color=MUTED)
                ax.invert_xaxis()
                ax.set_xlabel("recovery-error screen (stricter to the right)")
                ax.set_ylabel(r"median $\omega^2$")
                ax.legend(fontsize=6.2, loc="best")
                ax.set_title("labels: parameters surviving the screen",
                             fontsize=6.3, color=MUTED, pad=5)
        _panel_tag(ax, "c")
        fig.tight_layout(w_pad=2.0)
        _save(fig, figdir, "fig14_sufficiency.pdf")


def fig13_controlled(fac, sim, figdir, true_theta=None):
    """The causal counterpart to the observational decomposition.

    Comparing two real datasets confounds problem pool, arm size, aggregation
    noise and actual respondent behaviour, so an observational split cannot
    say which one makes a network encode a different theory. Here each is
    manipulated independently against a known agent.

    (a) effect size of every manipulated factor, architecture included, so the
        comparison is on one scale.
    (b) the readout the design targets: the generating agent's weighting
        function differs between the two agents by construction, and the
        rare-event index should track it.
    (c) whether the fitted coordinate recovers the agent that generated it.
    """
    cols = [c for c in fac.columns if c.startswith("omega2_")]
    med = fac[cols].median().sort_values(ascending=False)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.1),
                                 gridspec_kw=dict(width_ratios=[1.35, 1, 1]))

        ax = axes[0]
        ypos = np.arange(len(med))
        colours = ["#D55E00" if k == "omega2_arch" else "#0072B2"
                   for k in med.index]
        ax.barh(ypos, med.to_numpy(), color=colours, height=0.6, zorder=3)
        for y, k in zip(ypos, med.index):
            lo, hi = fac[k].min(), fac[k].max()
            ax.plot([lo, hi], [y, y], color=INK, lw=0.9, alpha=0.55, zorder=4)
        ax.set_yticks(ypos, [FACTOR_LABELS.get(k, k) for k in med.index],
                      fontsize=6.3)
        ax.invert_yaxis()
        ax.set_xlabel(r"median $\omega^2$ across readouts")
        ax.set_title("blue: a component of provenance", fontsize=6.5,
                     color=MUTED, pad=5)
        _panel_tag(ax, "a")

        ax = axes[1]
        if "rare_event_index" in sim.columns:
            g = (sim.groupby(["agent", "problems"])["rare_event_index"]
                 .agg(["mean", "std", "size"]))
            xs = np.arange(2)
            for k, pool in enumerate(sorted(sim["problems"].unique())):
                sub = g.xs(pool, level="problems").reindex(["desc", "exp"])
                se = sub["std"] / np.sqrt(sub["size"].clip(lower=1))
                ax.errorbar(xs + (k - 0.5) * 0.16, sub["mean"], yerr=se,
                            fmt="o-", ms=5, lw=1.6, capsize=2.5,
                            color=BLOCK_COLORS.get(
                                "c13k" if pool == "c13k" else "cpc18"),
                            markeredgecolor="white", markeredgewidth=0.5,
                            label=f"{pool} problems")
            ax.axhline(0, color=FAINT, lw=0.8, ls="--", zorder=1)
            ax.set_xticks(xs, ["overweights\nrare (desc)",
                               "linear\n(exp)"], fontsize=6.3)
            ax.set_xlabel("generating agent")
            ax.set_ylabel("rare-event index")
            ax.legend(fontsize=6.2, loc="best")
            ax.margins(x=0.25)
        _panel_tag(ax, "b")

        ax = axes[2]
        if "gamma_pos" in sim.columns and true_theta:
            sub = sim[sim.get("theory_identifiable", True)] \
                if "theory_identifiable" in sim.columns else sim
            g = sub.groupby("agent")["gamma_pos"].agg(["mean", "std", "size"])
            xs = np.arange(len(g))
            ax.bar(xs, g["mean"], 0.5,
                   yerr=g["std"] / np.sqrt(g["size"].clip(lower=1)),
                   color="#7a8b99", capsize=3, zorder=3)
            for x, a in zip(xs, g.index):
                truth = (true_theta["gamma_pos"] if a == "desc" else 1.0)
                ax.plot([x - 0.3, x + 0.3], [truth] * 2, color=INK, lw=1.4,
                        ls="--", zorder=4)
            ax.set_xticks(xs, [f"{a}\nagent" for a in g.index], fontsize=6.3)
            ax.set_ylabel(r"recovered $\gamma^{+}$")
            ax.set_title("dashed = value used to generate", fontsize=6.5,
                         color=MUTED, pad=5)
        _panel_tag(ax, "c")
        fig.tight_layout(w_pad=2.2)
        _save(fig, figdir, "fig13_controlled.pdf")


def fig12_experience_gradient(grad, trend, figdir,
                              params=("gamma_pos", "lam", "rare_event_index")):
    """Experience as an ordered factor, in two independent designs.

    CPC18 blocks are the same 210 problems and respondents at five feedback
    stages (within-problem); choices13k assigns each problem to one block, so
    its blocks are ~2,450 different problems each (between-problem). The two
    designs share no respondents, no problems and no country, so a trend that
    appears in both is a property of accumulating experience rather than of
    either sample.
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(params) + 1,
                                 figsize=(2.15 * (len(params) + 1), 3.0))
        for ax, prm, tag in zip(axes, params, "abcdef"):
            for src, sub in grad.groupby("source"):
                col = BLOCK_COLORS.get(src, "gray")
                g = sub.groupby("block")[prm].agg(["mean", "std", "size"])
                se = g["std"] / np.sqrt(g["size"].clip(lower=1))
                ax.errorbar(g.index, g["mean"], yerr=se, fmt="o-", ms=4.5,
                            lw=1.6, capsize=2, color=col,
                            markeredgecolor="white", markeredgewidth=0.5,
                            label=("CPC18 (within-problem)" if src == "cpc18"
                                   else "choices13k (between-problem)"))
            ax.set_xlabel("experience block")
            ax.set_ylabel(_pname(prm).split(" (")[0])
            ax.set_xticks(sorted(grad["block"].unique()))
            _panel_tag(ax, tag)
        axes[0].legend(fontsize=6.0, loc="best")

        ax = axes[-1]
        if trend is not None and len(trend):
            piv = trend.pivot(index="value", columns="source",
                              values="mean_rho").dropna()
            if {"c13k", "cpc18"} <= set(piv.columns) and len(piv):
                sig = trend.pivot(index="value", columns="source",
                                  values="p_value").reindex(piv.index)
                both = ((sig["c13k"] < 0.05) & (sig["cpc18"] < 0.05))
                ax.axhline(0, color=FAINT, lw=0.8, zorder=1)
                ax.axvline(0, color=FAINT, lw=0.8, zorder=1)
                ax.plot([-1, 1], [-1, 1], color=FAINT, lw=0.8, ls=":",
                        zorder=1)
                ax.scatter(piv["cpc18"][~both], piv["c13k"][~both], s=34,
                           facecolors="none", edgecolors=MUTED,
                           linewidths=0.9, zorder=3)
                ax.scatter(piv["cpc18"][both], piv["c13k"][both], s=42,
                           c=INK, zorder=4)
                # label only the readouts that are significant in both
                # sources; labelling all eleven turns the panel into a pile of
                # overlapping text and the interesting points are exactly
                # these ones
                lab = piv.index[both]
                # alternate the offset up/down: the significant points cluster
                # tightly, so a fixed offset overlaps them
                for k, v in enumerate(sorted(lab, key=lambda z: piv.loc[z, "cpc18"])):
                    dy = 6 if k % 2 == 0 else -11
                    ax.annotate(_pname(v).split(" (")[0].replace("_", " "),
                                (piv.loc[v, "cpc18"], piv.loc[v, "c13k"]),
                                textcoords="offset points", xytext=(5, dy),
                                fontsize=5.8, color=INK)
                agree = int((np.sign(piv["cpc18"]) == np.sign(piv["c13k"]))
                            .sum())
                ax.set_title(f"same sign in {agree}/{len(piv)} readouts",
                             fontsize=6.6, color=MUTED, pad=6)
                ax.set_xlabel(r"trend $\rho$, CPC18")
                ax.set_ylabel(r"trend $\rho$, choices13k")
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
        _panel_tag(ax, "abcdef"[len(params)])
        fig.tight_layout(w_pad=2.0)
        _save(fig, figdir, "fig12_experience_gradient.pdf")


def fig11_abstraction(ladder, probe_omega, figdir):
    """Why the answer depends on where you read it.

    (a) Both factors' share of variance at three levels of abstraction, from
    the raw 1,742 probe responses up to the six fitted CPT coordinates.
    (b) The same contest resolved probe by probe across the certainty-
    equivalent grid, so the global average in (a) can be located.
    (c) Collapsed by probe family.
    """
    from matplotlib.colors import LinearSegmentedColormap
    # diverging: architecture (warm) <- neutral -> provenance (cool)
    cmap = LinearSegmentedColormap.from_list(
        "prov_arch", ["#D55E00", "#EBD9CC", "#efefef", "#CCD9E6", "#0072B2"])

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0),
                                 gridspec_kw=dict(width_ratios=[1, 1.1, 1]))

        ax = axes[0]
        xs = np.arange(len(ladder))
        ax.plot(xs, ladder["r2_arch"], "o-", ms=6, lw=2, color="#D55E00",
                label="architecture", zorder=3)
        ax.plot(xs, ladder["r2_dataset"], "o-", ms=6, lw=2, color="#0072B2",
                label="dataset provenance", zorder=3)
        for x, r in zip(xs, ladder.itertuples()):
            ax.annotate(f"{r.ratio:.2f}", (x, max(r.r2_arch, r.r2_dataset)),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=6.2, color=MUTED)
        ax.set_xticks(xs, [f"{n}\n({d}d)" for n, d
                           in zip(ladder["level"], ladder["dim"])],
                      fontsize=6.3)
        ax.set_ylabel(r"share of variance ($R^2$)")
        ax.set_ylim(0, max(ladder[["r2_arch", "r2_dataset"]].to_numpy().max()
                           * 1.35, 0.1))
        ax.legend(fontsize=6.4, loc="lower left")
        ax.set_title("numbers: provenance / architecture", fontsize=6.5,
                     color=MUTED, pad=6)
        _panel_tag(ax, "a")

        ax = axes[1]
        ce = probe_omega[probe_omega["group"] == "ce_gain"]
        piv = ce.pivot_table(index="r", columns="p", values="diff",
                             aggfunc="mean")
        M = piv.to_numpy()
        # zero must stay the neutral colour, but the data run far further
        # negative than positive; a symmetric range would spend half the ramp
        # on values that do not occur
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vcenter=0.0,
                            vmin=min(float(np.nanmin(M)), -1e-6),
                            vmax=max(float(np.nanmax(M)), 1e-6))
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto",
                       origin="lower")
        ax.set_xticks(range(len(piv.columns)),
                      [f"{c:g}" for c in piv.columns], fontsize=5.6,
                      rotation=90)
        ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index],
                      fontsize=5.6)
        ax.set_xlabel("probability of the risky gain")
        ax.set_ylabel("certainty-equivalent ratio")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$\omega^2$ provenance $-$ architecture", fontsize=6.2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        _panel_tag(ax, "b")

        ax = axes[2]
        g = (probe_omega.groupby("group")["diff"].mean()
             .reindex(GROUP_ORDER).dropna().sort_values())
        cols = ["#0072B2" if v > 0 else "#D55E00" for v in g]
        ax.barh(np.arange(len(g)), g.to_numpy(), color=cols, height=0.62,
                zorder=3)
        ax.axvline(0, color=INK, lw=0.8, zorder=4)
        ax.set_yticks(range(len(g)),
                      [PROBE_GROUP_LABELS.get(i, i).replace("\n", " ")
                       for i in g.index], fontsize=6.2)
        ax.set_xlabel(r"mean $\omega^2$ gap")
        ax.set_title("blue: provenance wins the family", fontsize=6.5,
                     color=MUTED, pad=6)
        _panel_tag(ax, "c")
        fig.tight_layout(w_pad=2.2)
        _save(fig, figdir, "fig11_abstraction.pdf")


def fig10_population_calibration(pop, preds, run_info, figdir,
                                 params=("gamma_pos", "lam", "tau")):
    """(a-b) two labs running the same 182 problems under the same protocol:
    an upper bound on how much 'population' alone can move the readout, to be
    read against the provenance gaps elsewhere.
    (c) reliability of the fitted choice probabilities on held-out problems."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))

        ax = axes[0]
        labs = ["cpc18_technion_exp", "cpc18_rehovot_exp"]
        xpos = np.arange(len(params))
        # tau lives around 15 while gamma sits below 1, so a shared raw axis
        # renders the other parameters as a flat line; each is normalised by
        # its own two-lab mean, putting every comparison on a 1.0 baseline
        scale = {p: pop[p].mean() for p in params}
        for k, ds in enumerate(labs):
            sub = pop[pop["dataset"] == ds]
            n = max(len(sub), 1)
            mu = [sub[p].mean() / scale[p] for p in params]
            se = [sub[p].std() / np.sqrt(n) / abs(scale[p]) for p in params]
            ax.errorbar(xpos + (k - 0.5) * 0.18, mu, yerr=se, fmt="o", ms=5,
                        lw=1.2, capsize=2.5, color=DATASET_COLORS.get(ds),
                        markeredgecolor="white", markeredgewidth=0.5,
                        label=f"{ds.replace('cpc18_', '').replace('_exp', '')}"
                              f"  (n={len(sub)})")
        ax.axhline(1.0, color=INK, lw=0.9, ls="--", zorder=1)
        ax.set_xticks(xpos, [_pname(p).split(" (")[0] for p in params])
        ax.set_ylabel("coordinate / two-lab mean")
        ax.legend(fontsize=6.3, title="CPC18 lab", title_fontsize=6.3,
                  loc="best")
        _panel_tag(ax, "a")

        ax = axes[1]
        both = pop.groupby("dataset")[list(params)].mean()
        if len(both) == 2:
            gap = (both.loc[labs[0]] - both.loc[labs[1]]).abs()
            rel = gap / both.mean(axis=0).abs()
            ax.barh(np.arange(len(params)), rel.to_numpy() * 100,
                    color="#7a8b99", zorder=3)
            ax.set_yticks(range(len(params)),
                          [_pname(p).split(" (")[0] for p in params])
            ax.invert_yaxis()
            ax.set_xlabel("between-lab gap (% of mean)")
        _panel_tag(ax, "b")

        ax = axes[2]
        bins = np.linspace(0.2, 0.8, 13)
        ctr = 0.5 * (bins[1:] + bins[:-1])
        for ds in ["c13k_exp", "cpc18_exp"]:
            rid = [r for r in run_info[run_info["dataset"] == ds]["run_id"]
                   if r in preds]
            if not rid:
                continue
            P = np.concatenate([preds[r]["pred"] for r in rid])
            Y = np.concatenate([preds[r]["y"] for r in rid])
            idx = np.digitize(P, bins) - 1
            m = [(Y[idx == b].mean() if (idx == b).sum() > 5 else np.nan)
                 for b in range(len(ctr))]
            ax.plot(ctr, m, "o-", ms=4, lw=1.5,
                    color=DATASET_COLORS.get(ds), zorder=3,
                    label=DATASET_LABELS.get(ds, ds))
        ax.plot([0.2, 0.8], [0.2, 0.8], color=FAINT, lw=0.9, ls=":", zorder=1)
        ax.set_xlabel("predicted P(choose B)")
        ax.set_ylabel("observed rate")
        ax.legend(fontsize=6.3, loc="upper left")
        _panel_tag(ax, "c")
        fig.tight_layout(w_pad=2.2)
        _save(fig, figdir, "fig10_population_calibration.pdf")


def fig15_artifact_floor(ac, figdir):
    """Does either effect survive what the readout invents on its own?

    On arms generated by a single fixed agent every run shares the same true
    coordinates, so any omega^2 the readout still reports is manufactured by
    the readout rather than found in the data. That level is the floor an
    observed effect must clear before it can be read as a difference in
    encoded theory -- and it is not the same floor for both factors:
    architecture perturbs the prediction surface directly, swapping the
    problem pool barely does.
    """
    ac = ac.copy()
    ac["lab"] = [_pname(v).split(" (")[0] for v in ac["value"]]
    y = np.arange(len(ac))

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
        for ax, obs, flo, col, title in [
                (axes[0], "arch_observed", "arch_artifact_floor", "#D55E00",
                 "architecture"),
                (axes[1], "dataset_observed", "dataset_artifact_floor",
                 "#0072B2", "dataset provenance")]:
            ax.barh(y, ac[flo], height=0.66, color=FAINT, zorder=2,
                    label="artifact floor")
            ax.scatter(ac[obs], y, s=58, marker="D", c=col, zorder=4,
                       edgecolors="white", linewidths=0.7, label="observed")
            for yi, (o, f) in enumerate(zip(ac[obs], ac[flo])):
                ax.plot([f, o], [yi, yi],
                        color=col if o > f else MUTED, lw=1.3,
                        ls="-" if o > f else ":", zorder=3)
            n_ok = int((ac[obs] > ac[flo]).sum())
            ax.set_title(f"{title}  --  clears floor {n_ok}/{len(ac)}",
                         fontsize=7, color=MUTED, pad=6)
            ax.set_xlabel(r"$\omega^2$")
            ax.set_xlim(0, 1.02)
        # one legend for both panels; inside panel a it lands on the tau row
        axes[1].legend(fontsize=6.2, loc="lower right")
        axes[0].set_yticks(y, ac["lab"])
        axes[0].invert_yaxis()
        _panel_tag(axes[0], "a")
        _panel_tag(axes[1], "b")
        fig.text(0.5, -0.04,
                 "Grey bar: what the readout reports when the true "
                 "coordinates are identical across runs. A diamond to the "
                 "right of its bar is an effect that cannot be explained by "
                 "the readout alone.",
                 ha="center", va="top", fontsize=6.4, color=MUTED)
        fig.tight_layout(w_pad=2.0)
        _save(fig, figdir, "fig15_artifact_floor.pdf")
