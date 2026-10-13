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


def _strip(ax, x, vals, color, marker="o", s=5.5, alpha=0.55, rng=None,
           width=0.26):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return
    rng = rng or np.random.default_rng(0)
    jit = rng.uniform(-width, width, len(vals))
    ax.scatter(x + jit, vals, s=s, c=color, marker=marker, alpha=alpha,
               linewidths=0, zorder=3)
    ax.plot([x - 0.36, x + 0.36], [np.median(vals)] * 2, color=INK, lw=1.1,
            zorder=4)


def _p_concept_row(ax, base, sim, v, screen, ac, show_x, ylab):
    """One read-out, three blocks: fixed-agent runs by architecture (where the
    architecture floor comes from), fixed-agent runs by problem pool (where
    the provenance floor comes from), real corpora by training arm (the
    observed effects)."""
    rng = np.random.default_rng(1)
    suf = base[base["dataset"].str.startswith("suf_", na=False)
               & base[screen]]
    sm = sim[(sim["agent"] == "desc") & sim[screen]]
    main = base[base["dataset"].isin(["c13k_exp", "c13k_desc", "cpc18_exp",
                                      "cpc18_desc"]) & base[screen]]
    x = 0.0
    xt, xl, rot = [], [], []
    for a in ARCH_ORDER:
        _strip(ax, x, suf[suf["arch"] == a][v], AR_C, rng=rng)
        xt.append(x); xl.append(a.replace("TRANSFORMER", "TRANSF.")); rot.append(True)
        x += 1
    x += 1.0
    c2 = []
    for pool, lab in [("c13k", "c13k\npool"), ("cpc", "CPC18\npool")]:
        _strip(ax, x, sm[sm["problems"] == pool][v], DS_C, alpha=0.45,
               rng=rng, width=0.3)
        xt.append(x); xl.append(lab); rot.append(False); c2.append(x)
        x += 1.55
    x += 0.6
    c3 = []
    for d in ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]:
        sub = main[main["dataset"] == d]
        for a in ARCH_ORDER:
            _strip(ax, x, sub[sub["arch"] == a][v], DS_C,
                   marker=ARCH_MARKERS.get(a, "o"), s=7, alpha=0.75, rng=rng,
                   width=0.34)
        ax.plot([x - 0.4, x + 0.4], [sub[v].median()] * 2, color=INK,
                lw=1.1, zorder=4)
        xt.append(x); xl.append(DATASET_SHORT.get(d, d)); rot.append(False)
        c3.append(x)
        x += 1.55
    allv = pd.concat([suf[v], sm[v], main[v]]).dropna()
    lo, hi = allv.quantile(0.005), allv.quantile(0.995)
    rng_ = hi - lo
    ax.set_ylim(lo - 0.06 * rng_, hi + 0.52 * rng_)
    ax.set_xlim(-0.7, x - 0.6)
    sep1 = (xt[7] + c2[0]) / 2
    sep2 = (c2[-1] + c3[0]) / 2
    for xs in (sep1, sep2):
        ax.axvline(xs, color=FAINT, lw=0.7, ls=":")
    ax.set_ylabel(ylab, fontsize=6.0)
    r = ac.loc[v]
    y_txt = hi + 0.49 * rng_
    box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    ax.text(3.5, y_txt,
            r"floor:  $\omega^2_{\rm arch}$ = " f"{r.arch_artifact_floor:.2f}",
            ha="center", va="top", fontsize=5.4, color=AR_C, bbox=box)
    ax.text(np.mean(c2), y_txt,
            "floor:\n" r"$\omega^2_{\rm pool}$ = "
            f"{r.dataset_artifact_floor:.2f}",
            ha="center", va="top", fontsize=5.4, color=DS_C, bbox=box)
    ax.text(np.mean(c3), y_txt,
            "observed:\n" r"$\omega^2_{\rm prov}$ = "
            f"{r.dataset_observed:.2f}" "\n" r"$\omega^2_{\rm arch}$ = "
            f"{r.arch_observed:.2f}",
            ha="center", va="top", fontsize=5.4, color=INK, bbox=box)
    if show_x:
        ax.set_xticks(xt, xl, fontsize=4.8)
        for lab, rr in zip(ax.get_xticklabels(), rot):
            if rr:
                lab.set_rotation(50); lab.set_ha("right")
    else:
        ax.set_xticks(xt, [""] * len(xt))
    ax.tick_params(axis="x", length=0)
    return 3.5, float(np.mean(c2)), float(np.mean(c3))


def _p_ridge(ax, bt, boot, which, show_y=True):
    """Bootstrap distribution of the observed effect against that of its
    floor, one ridge per read-out."""
    col = AR_C if which == "arch" else DS_C
    d = bt.sort_values(["family", "value"], ascending=[False, True])
    rows = [v for v in READOUT_ORDER if v in d["value"].tolist()]
    grid = np.linspace(0, 1, 400)
    n = len(rows)
    for i, v in enumerate(rows):
        y0 = n - 1 - i
        obs = boot[f"{v}|obs_{which}"]
        flo = boot[f"{v}|floor_{which}"]
        for arr, c, al, z in [(flo, "#9a9a9a", 0.85, 2), (obs, col, 0.72, 3)]:
            k = _kde(arr, grid)
            k = k / max(k.max(), 1e-9) * 0.9
            ax.fill_between(grid, y0, y0 + k, color=c, alpha=al, lw=0,
                            zorder=z)
            ax.plot(grid, y0 + k, color=c, lw=0.45, zorder=z + 0.1)
        r = d[d["value"] == v].iloc[0]
        m = r[f"{which}_lo"] - r[f"{which}_floor_hi"]
        ok = bool(r[f"{which}_clears"])
        ax.text(1.03, y0 + 0.12, f"{m:+.2f}", ha="left", va="bottom",
                fontsize=5.2, color=col if ok else MUTED,
                fontweight="bold" if ok else "normal", clip_on=False)
    ax.set_yticks([n - 1 - i for i in range(n)],
                  [_compact(v) for v in rows] if show_y else [""] * n,
                  fontsize=5.4)
    ax.axhline(5 - 0.06, color=FAINT, lw=0.7, ls=":")
    ax.set_ylim(-0.1, n + 0.05)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$\omega^2$")
    k = int(bt[f"{which}_clears"].sum())
    nm = "architecture" if which == "arch" else "provenance"
    ax.set_title(f"{nm}: clears its floor on {k}/{len(bt)}", fontsize=6.4,
                 color=MUTED, pad=4)


def main2_artifact_floor(bt, ac, boot, base, sim, figdir):
    """The headline, in four panels.

    a  what the floor is: with the generating theory held identical, the
       read-out still spreads across architectures (orange) -- that spread
       is the architecture floor; swapping the problem pool barely moves it
       (that is the provenance floor); real corpora (right) spread by arm.
    b  architecture: bootstrap distribution of the observed omega^2 against
       the floor's, per read-out. c  the same for provenance.
    d  both margins at once: every read-out in the quadrant where provenance
       clears and architecture does not.
    `boot` is artifact_boot_samples.npz from 06; `ac` artifact_corrected.csv
    indexed by value; `base`/`sim` the run-level tables.
    """
    from matplotlib import gridspec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ac = ac.set_index("value") if "value" in ac.columns else ac

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(11.4, 4.7))
        gs = gridspec.GridSpec(1, 4, figure=fig,
                               width_ratios=[1.85, 0.9, 0.9, 1.0],
                               wspace=0.5)
        gsa = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0],
                                               hspace=0.12)
        ax1 = fig.add_subplot(gsa[0]); ax2 = fig.add_subplot(gsa[1])
        cx = _p_concept_row(ax1, base, sim, "rare_event_index",
                            "identifiable", ac, False, "rare-event index")
        _p_concept_row(ax2, base, sim, "tau", "theory_identifiable", ac,
                       True, r"$\tau$")
        for xc, txt, col in [(cx[0], "by architecture", AR_C),
                             (cx[1], "by pool", DS_C),
                             (cx[2], "by training arm", INK)]:
            ax1.text(xc, 1.02, txt, ha="center", va="bottom",
                     transform=ax1.get_xaxis_transform(), fontsize=5.6,
                     color=col)
        ax1.text((cx[0] + cx[1]) / 2, 1.13,
                 "fixed agent (truth identical)", ha="center", va="bottom",
                 transform=ax1.get_xaxis_transform(), fontsize=5.8,
                 color=MUTED)
        ax1.text(cx[2], 1.13, "real corpora", ha="center", va="bottom",
                 transform=ax1.get_xaxis_transform(), fontsize=5.8,
                 color=MUTED)
        _tag(ax1, "a")

        axb = fig.add_subplot(gs[1]); _p_ridge(axb, bt, boot, "arch")
        _tag(axb, "b")
        axc = fig.add_subplot(gs[2]); _p_ridge(axc, bt, boot, "ds",
                                               show_y=False)
        _tag(axc, "c")
        axb.legend(handles=[Patch(color="#9a9a9a", alpha=0.85,
                                  label="floor (fixed agent)"),
                            Patch(color=AR_C, alpha=0.72,
                                  label="observed, architecture"),
                            Patch(color=DS_C, alpha=0.72,
                                  label="observed, provenance")],
                   fontsize=5.2, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(1.05, -0.14), columnspacing=1.0,
                   handlelength=1.4)

        # d: quadrant of margins
        ax = fig.add_subplot(gs[3])
        d = bt.copy()
        d["mx"] = d["arch_lo"] - d["arch_floor_hi"]
        d["my"] = d["ds_lo"] - d["ds_floor_hi"]
        xlo = min(d["mx"].min() - 0.06, -0.6)
        xhi = max(d["mx"].max() + 0.08, 0.12)
        ylo = min(d["my"].min() - 0.08, -0.12)
        yhi = max(d["my"].max() + 0.08, 0.6)
        ax.axhspan(0, yhi, xmin=0, xmax=(0 - xlo) / (xhi - xlo),
                   color=DS_C, alpha=0.07, zorder=0)
        ax.axhline(0, color=INK, lw=0.8, zorder=2)
        ax.axvline(0, color=INK, lw=0.8, zorder=2)
        for fam, mk in [("CPT coordinate", "o"), ("behavioural index", "s")]:
            sub = d[d["family"] == fam]
            ax.scatter(sub["mx"], sub["my"], s=26, marker=mk, c=INK,
                       edgecolors="white", linewidths=0.5, zorder=4,
                       label=fam)
        order = d.sort_values("my")
        for k, r in enumerate(order.itertuples()):
            dx = 5 if (k % 2 == 0) else -5
            ax.annotate(_compact(r.value), (r.mx, r.my),
                        xytext=(dx, 3 if k % 2 == 0 else -3),
                        textcoords="offset points", fontsize=5.0,
                        ha="left" if dx > 0 else "right", va="center",
                        color=INK)
        n_q = int(((d["mx"] < 0) & (d["my"] > 0)).sum())
        ax.text(xlo + 0.02, yhi - 0.02,
                f"provenance clears,\narchitecture does not:\n{n_q}/{len(d)}",
                ha="left", va="top", fontsize=5.8, color=DS_C)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel("architecture margin", fontsize=6.2)
        ax.set_ylabel("provenance margin", fontsize=6.2)
        ax.legend(fontsize=5.2, loc="lower right", frameon=False,
                  handletextpad=0.3)
        ax.set_title("both margins at once", fontsize=6.4, color=MUTED,
                     pad=4)
        _tag(ax, "d")

        fig.text(0.5, -0.04,
                 "a: one point per trained model, black bars are medians.  "
                 "b\u2013d: 400 within-cell bootstrap resamples; margin = "
                 "observed lower bound \u2212 floor upper bound.\n"
                 "The floor is what the read-out reports on fixed-agent "
                 "arms, where every run encodes the same theory.",
                 ha="center", va="top", fontsize=6.0, color=MUTED)
        fig.tight_layout(w_pad=1.0)
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


def _kde(x, grid):
    from scipy.stats import gaussian_kde
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.std(x) < 1e-9:
        y = np.zeros_like(grid)
        if len(x):
            y[np.argmin(np.abs(grid - np.median(x)))] = 1.0
        return y
    return gaussian_kde(x, bw_method=0.25)(grid)


ARCH_ORDER = ["LINEAR", "MLP_S", "MLP_L", "RESMLP", "SET", "TRANSFORMER",
              "CPT_NET", "GBT"]
ARCH_SHADES = dict(zip(ARCH_ORDER, plt.get_cmap("Greys")(
    np.linspace(0.25, 0.95, len(ARCH_ORDER)))))
READOUT_ORDER = ["alpha", "beta", "lam", "gamma_pos", "gamma_neg", "tau",
                 "rare_event_index", "certainty_index", "reflection_index",
                 "loss_aversion_kappa", "ev_consistency"]
PROV_DEFS = ["respondents", "problem pool", "arm size", "largest component",
             "composite (pool x size x respondents)"]
ARCH_DEFS = ["sufficiency suite", "controlled suite"]


def _short(v):
    return _pname(v).split(" (")[0].replace("_", " ")


COMPACT = {"rare_event_index": "rare-event", "certainty_index": "certainty",
           "reflection_index": "reflection",
           "loss_aversion_kappa": r"loss-aversion $\hat\kappa$",
           "ev_consistency": "EV-consistency"}


def _compact(v):
    return COMPACT.get(v, _short(v))


def supp_robustness(fd, fs, jk, sym, boot, perm, figdir):
    """Robustness of the headline, in six panels (11_robustness.py):
      a  signed margin (observed lower bound - floor upper bound) of every
         read-out under every floor definition, both factors
      b  the same margins as a ladder over increasingly harsh provenance
         floors, survivors of the harshest one named
      c  bootstrap distributions behind the test: observed provenance effect
         against the composite floor, per read-out
      d  leave-one-architecture-out, per read-out
      e  the distilled CE expressions as functions: distance matrix ordered
         by training arm, then architecture
      f  restricted-permutation nulls of PERMANOVA pseudo-F for both factors
    `boot` and `perm` are the npz archives 11_robustness.py writes.
    """
    from matplotlib import gridspec
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fd = fd.copy()
    fd["margin"] = fd["obs_lo"] - fd["floor_hi"]
    prov = fd[fd["factor"] == "provenance"]
    arch = fd[fd["factor"] == "architecture"]
    rows = [v for v in READOUT_ORDER if v in prov["value"].unique()]
    ylab = [_short(v) for v in rows]

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13.4, 7.6))
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0],
                               width_ratios=[1.25, 1.0, 1.0], hspace=0.42,
                               wspace=0.38)

        # ---------------- a: margin matrix ------------------------------
        ax = fig.add_subplot(gs[0, 0])
        cols = PROV_DEFS + ARCH_DEFS
        Mx = np.full((len(rows), len(cols)), np.nan)
        clears = np.zeros_like(Mx, dtype=bool)
        for j, d in enumerate(cols):
            src = prov if d in PROV_DEFS else arch
            sub = src[src["floor_def"] == d].set_index("value")
            for i, v in enumerate(rows):
                if v in sub.index:
                    Mx[i, j] = sub.loc[v, "margin"]
                    clears[i, j] = bool(sub.loc[v, "clears_interval"])
        lim = np.nanmax(np.abs(Mx))
        im = ax.imshow(Mx, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim),
                       aspect="auto", interpolation="nearest")
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = Mx[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=5.0, fontweight="bold" if clears[i, j]
                        else "normal",
                        color="white" if abs(v) > 0.45 * lim else INK)
        ax.set_xticks(range(len(cols)),
                      [FLOOR_DEF_SHORT.get(c, c) for c in cols], fontsize=5.6,
                      rotation=35, ha="right")
        ax.set_yticks(range(len(rows)), ylab, fontsize=5.6)
        ax.axvline(len(PROV_DEFS) - 0.5, color=INK, lw=1.2)
        ax.axhline(5.5, color=INK, lw=0.6, ls=":")
        ax.text((len(PROV_DEFS) - 1) / 2, -1.05, "provenance floors",
                ha="center", va="bottom", fontsize=6.2, color=DS_C)
        ax.text(len(PROV_DEFS) + 0.5, -1.05, "architecture floors",
                ha="center", va="bottom", fontsize=6.2, color=AR_C)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("margin", fontsize=5.8)
        cb.ax.tick_params(labelsize=5.2)
        ax.set_title("margin = observed low $-$ floor high;  bold = clears",
                     fontsize=6.2, color=MUTED, pad=16)
        _tag(ax, "a")

        # ---------------- b: survival ladder ----------------------------
        ax = fig.add_subplot(gs[0, 1])
        xs = np.arange(len(PROV_DEFS))
        comp = prov[prov["floor_def"] == PROV_DEFS[-1]].set_index("value")
        survivors = [v for v in rows if comp.loc[v, "clears_interval"]]
        hi_cols = ["#0072B2", "#009E73", "#CC79A7", "#E69F00"]
        for v in rows:
            m = [prov[(prov["floor_def"] == d) & (prov["value"] == v)]
                 ["margin"].iloc[0] for d in PROV_DEFS]
            if v in survivors:
                c = hi_cols[survivors.index(v) % len(hi_cols)]
                ax.plot(xs, m, color=c, lw=1.8, marker="o", ms=3.6,
                        zorder=5, markeredgecolor="white",
                        markeredgewidth=0.5)
                ax.annotate(_short(v), (xs[-1], m[-1]),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=5.8, color=c, va="center", ha="left")
            else:
                ax.plot(xs, m, color=FAINT, lw=1.0, marker="o", ms=2.4,
                        zorder=3, alpha=0.9)
        ax.axhline(0, color=INK, lw=0.9, zorder=4)
        ax.axhspan(-1, 0, color="#f3f3f3", zorder=0)
        med = fs[fs["factor"] == "provenance"].set_index("floor_def")
        ax.set_xticks(xs, [f"{FLOOR_DEF_SHORT[d]}\n(floor {med.loc[d, 'median_floor']:.2f})"
                           for d in PROV_DEFS], fontsize=5.4)
        ax.set_ylabel("observed low $-$ floor high")
        ax.set_ylim(min(-0.62, prov["margin"].min() - 0.04),
                    prov["margin"].max() + 0.06)
        ax.set_xlim(-0.3, len(PROV_DEFS) - 0.2)
        counts = [int(med.loc[d, "clears_interval"]) for d in PROV_DEFS]
        for x, n in zip(xs, counts):
            ax.text(x, ax.get_ylim()[1] - 0.01, f"{n}/11", ha="center",
                    va="top", fontsize=5.8, color=DS_C, fontweight="bold")
        ax.set_title("provenance as the floor gets harsher",
                     fontsize=6.4, color=MUTED, pad=4)
        _tag(ax, "b")

        # ---------------- c: ridgeline of bootstrap distributions ------
        ax = fig.add_subplot(gs[0, 2])
        grid = np.linspace(0, 1, 400)
        step = 1.0
        pk = prov[prov["floor_def"] == "problem pool"].set_index("value")
        ck = comp
        for i, v in enumerate(rows):
            y0 = (len(rows) - 1 - i) * step
            obs = boot[f"{v}|obs_ds"]
            flo = boot[f"{v}|floor_ds|{PROV_DEFS[-1]}"]
            poo = boot[f"{v}|floor_ds|problem pool"]
            for arr, col, al, z in [(poo, "#cfcfcf", 0.9, 2),
                                    (flo, "#6e6e6e", 0.85, 3),
                                    (obs, DS_C, 0.75, 4)]:
                k = _kde(arr, grid)
                k = k / max(k.max(), 1e-9) * 0.92 * step
                ax.fill_between(grid, y0, y0 + k, color=col, alpha=al,
                                lw=0, zorder=z)
                ax.plot(grid, y0 + k, color=col, lw=0.5, zorder=z + 0.1)
            p = ck.loc[v, "p_holm"]
            tag = (f"$p_{{\\mathrm{{Holm}}}}$ {'<0.001' if p < 1e-3 else f'= {p:.3f}'}")
            ax.text(1.0, y0 + 0.12 * step, tag, ha="right", va="bottom",
                    fontsize=5.0, color=INK if ck.loc[v, "clears_interval"]
                    else MUTED)
        ax.set_yticks([(len(rows) - 1 - i) * step for i in range(len(rows))],
                      ylab, fontsize=5.6)
        ax.set_ylim(-0.1, len(rows) * step + 0.1)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\omega^2$")
        ax.axhline(5 * step - 0.05, color=FAINT, lw=0.7, ls=":")
        ax.legend(handles=[Patch(color=DS_C, alpha=0.75,
                                 label="observed provenance effect"),
                           Patch(color="#6e6e6e", label="composite floor"),
                           Patch(color="#cfcfcf", label="problem-pool floor")],
                  fontsize=5.2, loc="upper center", frameon=False, ncol=3,
                  bbox_to_anchor=(0.5, -0.13), columnspacing=1.0,
                  handlelength=1.4)
        ax.set_title("bootstrap distributions behind the test",
                     fontsize=6.4, color=MUTED, pad=4)
        _tag(ax, "c")

        # ---------------- d: jackknife dot matrix ----------------------
        ax = fig.add_subplot(gs[1, 0])
        lo_order = ["none"] + ARCH_ORDER
        lo_order = [l for l in lo_order if l in jk["left_out"].unique()]
        for j, lo in enumerate(lo_order):
            sub = jk[jk["left_out"] == lo].set_index("value")
            for i, v in enumerate(rows):
                if v not in sub.index:
                    continue
                r = sub.loc[v]
                if r["ds_clears"]:
                    ax.scatter(j, i, s=46, color=DS_C, zorder=4,
                               edgecolors="white", linewidths=0.5)
                else:
                    ax.scatter(j, i, s=46, facecolors="white",
                               edgecolors=FAINT, linewidths=0.8, zorder=3)
                if r["ds_clears_composite"]:
                    ax.scatter(j, i, s=120, facecolors="none",
                               edgecolors=INK, linewidths=0.9, zorder=5)
                if r["arch_clears"]:
                    ax.scatter(j, i, s=22, marker="s", color=AR_C, zorder=6,
                               edgecolors="white", linewidths=0.4)
            n_p = int(sub["ds_clears"].sum())
            n_c = int(sub["ds_clears_composite"].sum())
            n_a = int(sub["arch_clears"].sum())
            ax.text(j, -1.0, f"{n_p}", ha="center", va="center", fontsize=6.0,
                    color=DS_C, fontweight="bold")
            ax.text(j, -1.7, f"{n_c}", ha="center", va="center", fontsize=5.4,
                    color=INK)
            ax.text(j, -2.4, f"{n_a}", ha="center", va="center", fontsize=5.4,
                    color=AR_C)
        ax.text(-0.9, -1.0, "pool", ha="right", va="center", fontsize=5.4,
                color=DS_C)
        ax.text(-0.9, -1.7, "composite", ha="right", va="center",
                fontsize=5.4, color=INK)
        ax.text(-0.9, -2.4, "architecture", ha="right", va="center",
                fontsize=5.4, color=AR_C)
        ax.set_xticks(range(len(lo_order)),
                      ["all" if l == "none" else l for l in lo_order],
                      fontsize=5.4, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)), ylab, fontsize=5.6)
        ax.set_ylim(len(rows) - 0.4, -3.0)
        ax.set_xlim(-0.7, len(lo_order) - 0.3)
        ax.axhline(5.5, color=FAINT, lw=0.7, ls=":")
        ax.axhline(-0.5, color=INK, lw=0.6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("architecture left out")
        ax.legend(handles=[
            Line2D([], [], marker="o", color=DS_C, ls="", ms=5.5,
                   label="provenance clears, pool floor"),
            Line2D([], [], marker="o", mfc="none", mec=INK, ls="", ms=8,
                   label="also clears composite floor"),
            Line2D([], [], marker="s", color=AR_C, ls="", ms=4,
                   label="architecture clears")],
            fontsize=5.2, loc="lower left", bbox_to_anchor=(0.0, -0.42),
            ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.0)
        ax.set_title("leave one architecture out (counts above)",
                     fontsize=6.4, color=MUTED, pad=4)
        _tag(ax, "d")

        # ---------------- e: distance matrix ---------------------------
        ax = fig.add_subplot(gs[1, 1])
        D = np.sqrt(np.maximum(perm["functional|D2"], 0))
        dsl = perm["functional|dataset"].astype(str)
        arl = perm["functional|arch"].astype(str)
        ds_order = [d for d in ["c13k_exp", "c13k_desc", "cpc18_exp",
                                "cpc18_desc"] if d in set(dsl)]
        key = [(ds_order.index(d), ARCH_ORDER.index(a) if a in ARCH_ORDER
                else 99) for d, a in zip(dsl, arl)]
        order = np.argsort(np.array([k[0] * 100 + k[1] for k in key]),
                           kind="stable")
        Do = D[np.ix_(order, order)]
        n = len(order)
        im = ax.imshow(Do, cmap="magma_r", interpolation="nearest",
                       aspect="equal", zorder=2)
        bounds = np.cumsum([int((dsl == d).sum()) for d in ds_order])
        for b in bounds[:-1]:
            ax.axhline(b - 0.5, color="white", lw=1.0, zorder=3)
            ax.axvline(b - 0.5, color="white", lw=1.0, zorder=3)
        # colour strips for arm (outer) and architecture (inner)
        strip = 2.2
        for k, idx in enumerate(order):
            ax.add_patch(plt.Rectangle((-2 * strip - 1.0, k - 0.5), strip, 1,
                                       color=DATASET_COLORS.get(dsl[idx],
                                                                MUTED),
                                       lw=0, clip_on=False, zorder=4))
            ax.add_patch(plt.Rectangle((-strip - 0.6, k - 0.5), strip, 1,
                                       color=ARCH_SHADES.get(arl[idx], MUTED),
                                       lw=0, clip_on=False, zorder=4))
            ax.add_patch(plt.Rectangle((k - 0.5, -2 * strip - 1.0), 1, strip,
                                       color=DATASET_COLORS.get(dsl[idx],
                                                                MUTED),
                                       lw=0, clip_on=False, zorder=4))
            ax.add_patch(plt.Rectangle((k - 0.5, -strip - 0.6), 1, strip,
                                       color=ARCH_SHADES.get(arl[idx], MUTED),
                                       lw=0, clip_on=False, zorder=4))
        starts = np.concatenate([[0], bounds[:-1]])
        for d, s0, s1 in zip(ds_order, starts, bounds):
            ax.text((s0 + s1) / 2 - 0.5, n + 1.5, DATASET_SHORT.get(d, d),
                    ha="center", va="top", fontsize=5.4,
                    color=DATASET_COLORS.get(d, MUTED))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-2 * strip - 1.2, n - 0.5)
        ax.set_ylim(n - 0.5, -2 * strip - 1.2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("distance between recovered CE(x, p) functions",
                     fontsize=5.6)
        cb.ax.tick_params(labelsize=5.2)
        ax.legend(handles=[Patch(color=ARCH_SHADES[a], label=a)
                           for a in ARCH_ORDER],
                  fontsize=4.8, loc="upper left", bbox_to_anchor=(-0.02, -0.08),
                  ncol=4, frameon=False, handlelength=1.0, columnspacing=0.8,
                  title="inner strip: architecture", title_fontsize=5.2)
        fsym = sym[sym["view"].str.startswith("functional")].iloc[0]
        ax.set_title(f"{n} runs ordered by arm, then architecture",
                     fontsize=6.4, color=MUTED, pad=16)
        _tag(ax, "e")

        # ---------------- f: permutation nulls -------------------------
        ax = fig.add_subplot(gs[1, 2])
        for view, ls in [("functional", "-"), ("syntactic", ":")]:
            for fac, col, key in [("provenance", DS_C, "dataset"),
                                  ("architecture", AR_C, "arch")]:
                null = perm[f"{view}|null_F_{key}"]
                row = sym[sym["view"].str.startswith(view)].iloc[0]
                fobs = row[f"permanova_F_{key}"]
                pval = row[f"permanova_p_{key}"]
                bins = np.linspace(0.4, max(3.6, fobs + 0.2), 56)
                ax.hist(null, bins=bins, histtype="step", color=col, lw=1.1,
                        ls=ls, density=True, zorder=3,
                        alpha=1.0 if view == "functional" else 0.7)
                ax.axvline(fobs, color=col, lw=1.6 if view == "functional"
                           else 1.0, ls=ls, zorder=4)
                if view == "functional":
                    import matplotlib.transforms as mtr
                    ax.text(fobs, 0.80 if key == "dataset" else 0.62,
                            f" {fac}\n $F$ = {fobs:.2f}\n $p$ "
                            f"{'< 0.001' if pval < 1e-3 else f'= {pval:.3f}'}",
                            transform=mtr.blended_transform_factory(
                                ax.transData, ax.transAxes),
                            fontsize=5.4, color=col, va="top", ha="left")
        ax.set_xlabel("PERMANOVA pseudo-$F$")
        ax.set_ylabel("density of the permutation null")
        ax.legend(handles=[Line2D([], [], color=INK, ls="-",
                                  label="functional view (CE grid)"),
                           Line2D([], [], color=INK, ls=":",
                                  label="syntactic view (operators)")],
                  fontsize=5.2, loc="upper right", frameon=False)
        ax.set_title("2,000 permutations within strata of the other factor",
                     fontsize=6.4, color=MUTED, pad=4)
        _tag(ax, "f")

        _save(fig, figdir, "supp_robustness.pdf")
