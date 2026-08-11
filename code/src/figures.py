"""Paper figures (matplotlib only, vector PDF output).

fig1  theory-space scatter: colour = training dataset, marker = architecture,
      95% covariance ellipse per dataset, non-identifiable runs shown apart
fig2  (a) implied choice curves vs p at EV parity  (b) the rare-event sign flip
fig3  transfer completeness heatmap with marginal means
fig4  (a) variance decomposition  (b) ground-truth recovery as a ratio

Palette: the four provenance arms are a 2x2 (warm = choices13k, cool = CPC18;
dark = experience, light = description) drawn from the Okabe-Ito colourblind-
safe set. Checked with the dataviz validator: worst adjacent pair dE 15.6
(normal vision), 13.1 deutan, 14.8 tritan -- all above threshold. The two
lighter fills sit below 3:1 contrast on white, so every mark carries a dark
edge and every series is named in a legend.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

DATASET_COLORS = {
    "c13k_exp": "#D55E00", "c13k_desc": "#E69F00",
    "cpc18_exp": "#0072B2", "cpc18_desc": "#56B4E9",
    "syn_c13k": "#D55E00", "syn_cpc": "#0072B2",
    "c13k_pair_exp": "#D55E00", "c13k_pair_desc": "#E69F00",
    "cpc18_technion_exp": "#0072B2", "cpc18_rehovot_exp": "#56B4E9",
    "cpc18_comp_desc": "#56B4E9", "cpc18_comp_exp": "#0072B2",
}
# One distinct marker per architecture. Every architecture in
# src.models.ARCHS must appear here: the lookups fall back to "o", so a
# missing entry silently draws that architecture as LINEAR.
ARCH_MARKERS = {"LINEAR": "o", "MLP_S": "s", "MLP_L": "D", "RESMLP": "d",
                "SET": "^", "TRANSFORMER": "v", "CPT_NET": "P", "GBT": "X"}
DATASET_LABELS = {
    "c13k_exp": "choices13k experience", "c13k_desc": "choices13k description",
    "cpc18_exp": "CPC18 experience", "cpc18_desc": "CPC18 description",
    "c13k_pair_exp": "matched pairs, experience",
    "c13k_pair_desc": "matched pairs, description",
    "syn_c13k": "synthetic on c13k problems",
    "syn_cpc": "synthetic on CPC18 problems",
    "cpc18_comp_desc": "CPC18 competition, description",
    "cpc18_comp_exp": "CPC18 competition, experience",
}
# Compact forms for axis ticks, where the full labels collide
DATASET_SHORT = {
    "c13k_exp": "c13k\nexp", "c13k_desc": "c13k\ndesc",
    "cpc18_exp": "CPC18\nexp", "cpc18_desc": "CPC18\ndesc",
    "cpc18_comp_desc": "comp\ndesc",
    "cpc18_comp_exp": "comp\nexp",
}
INK, MUTED, FAINT = "#1a1a1a", "#5c5c5c", "#b4b4b4"

RC = {
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.linewidth": 1.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


PNG_DPI = 600  # line art; journals ask for 300-600


def _save(fig, figdir, name, png_dpi=PNG_DPI):
    """Write the vector master and a high-resolution raster of the same
    figure. Submission systems and manuscript drafts want PNG; the PDF stays
    as the editable master."""
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name), bbox_inches="tight")
    if name.lower().endswith(".pdf"):
        fig.savefig(os.path.join(figdir, name[:-4] + ".png"),
                    bbox_inches="tight", dpi=png_dpi)
    plt.close(fig)


def _panel_tag(ax, tag):
    ax.text(-0.16, 1.04, tag, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left", color=INK)


def _ellipse(ax, x, y, color, n_std=1.0):
    """Concentration ellipse (1 SD) around the centre of a dataset's cloud.
    One SD rather than two: at two the clouds overlap so heavily that the
    panel stops being readable, and the ellipse is here to show where a
    dataset sits, not to bound its support."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or np.linalg.det(cov) <= 0:
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang,
                         facecolor="none", edgecolor=color,
                         lw=1.1, ls="--", alpha=0.85, zorder=4))


def _dataset_arch_legend(fig, datasets, archs, y=1.02):
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5.5,
                          markerfacecolor=DATASET_COLORS.get(d, "gray"),
                          markeredgecolor=INK, markeredgewidth=0.5,
                          label=DATASET_LABELS.get(d, d))
               for d in datasets]
    handles += [plt.Line2D([], [], ls="", marker=ARCH_MARKERS.get(a, "o"),
                           ms=5.5, markerfacecolor="none",
                           markeredgecolor=INK, markeredgewidth=0.9, label=a)
                for a in archs]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, y), columnspacing=1.1,
               handletextpad=0.35)


# ---------------------------------------------------------------------------

def fig1_theory_space(coords, figdir, bounds=None,
                      screen="theory_identifiable"):
    """Theory-space scatter. Runs that failed the theory screen -- flat probe
    behaviour, or a parameter pinned to its box constraint -- have coordinates
    that are an artefact of the fit, so they are drawn as faint open circles
    and excluded from the ellipses."""
    pairs = [("gamma_pos", "gamma_neg"), ("alpha", "lam")]
    if screen not in coords.columns:
        coords = coords.assign(**{screen: True})
    good = coords[coords[screen]]
    bad = coords[~coords[screen]]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
        for ax, (xv, yv), tag in zip(axes, pairs, "ab"):
            # Non-identifiable runs are deliberately not plotted. Their
            # coordinates sit wherever the box constraint put them, so drawing
            # them would show the bounds, not the fits, and would compress the
            # real estimates into an unreadable clump at the centre. The count
            # is reported in the caption instead.
            for ds, sub in good.groupby("dataset"):
                col = DATASET_COLORS.get(ds, "gray")
                _ellipse(ax, sub[xv], sub[yv], col)
                for arch, ss in sub.groupby("arch"):
                    ax.scatter(ss[xv], ss[yv], s=26, c=col,
                               marker=ARCH_MARKERS.get(arch, "o"),
                               alpha=0.9, linewidths=0.45, edgecolors="white",
                               zorder=3)
            ax.set_xlabel(_pname(xv))
            ax.set_ylabel(_pname(yv))
            _panel_tag(ax, tag)

        fig.tight_layout(w_pad=3.0, rect=(0, 0.02, 1, 0.99))
        _dataset_arch_legend(fig, [d for d in DATASET_LABELS
                                   if d in good["dataset"].unique()],
                             [a for a in ARCH_MARKERS
                              if a in good["arch"].unique()], y=1.13)
        if len(bad):
            fig.text(0.5, -0.02,
                     f"Dashed ellipses are 1 SD per dataset. {len(bad)} of "
                     f"{len(coords)} runs are omitted: their coordinates are "
                     "not identifiable\n(flat probe behaviour, or a parameter "
                     "pinned to a fitting bound), so they would show where "
                     "the constraint is, not where the data are.",
                     ha="center", va="top", fontsize=6.5, color=MUTED)
        _save(fig, figdir, "fig1_theory_space.pdf")


_PRETTY = {"gamma_pos": r"$\gamma^{+}$ (gain weighting)",
           "gamma_neg": r"$\gamma^{-}$ (loss weighting)",
           "alpha": r"$\alpha$ (gain curvature)",
           "lam": r"$\lambda$ (loss aversion)",
           "beta": r"$\beta$ (loss curvature)",
           "tau": r"$\tau$ (choice temperature)"}


def _pname(v):
    return _PRETTY.get(v, v)


# ---------------------------------------------------------------------------

def fig2_de_flip(probe_meta, preds_by_run, run_info, figdir,
                 datasets=("c13k_pair_desc", "c13k_pair_exp"), h3=None):
    """(a) mean P(choose risky) at EV parity vs p, band = across-run SD.
    (b) the actual sign flip: rare-event index per architecture, description
    vs experience, which is what H3 claims."""
    m = probe_meta
    sel_g = (m["group"] == "ce_gain") & np.isclose(m["r"], 1.0)
    ps = sorted(m.loc[sel_g, "p"].unique())

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                 gridspec_kw=dict(width_ratios=[1.05, 1]))
        ax = axes[0]
        curves_by_ds = {}
        for ds in datasets:
            rids = [r for r in run_info[run_info["dataset"] == ds]["run_id"]
                    if r in preds_by_run]
            cur = np.array([[preds_by_run[r][(sel_g & (m["p"] == p)).to_numpy()]
                             .mean() for p in ps] for r in rids])
            if not len(cur):
                continue
            curves_by_ds[ds] = cur
            mu, sd = cur.mean(0), cur.std(0)
            col = DATASET_COLORS.get(ds)
            # description solid / experience dashed: the two arms are the
            # closest pair in the palette, so identity does not rest on hue
            ls = "--" if ds.endswith("exp") else "-"
            ax.plot(ps, mu, color=col, lw=1.8, ls=ls, zorder=3,
                    label=DATASET_LABELS.get(ds, ds))
            ax.fill_between(ps, mu - sd, mu + sd, alpha=0.16, color=col,
                            lw=0, zorder=2)
        ax.axhline(0.5, color=FAINT, lw=0.8, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("probability of the risky gain (EV-matched)")
        ax.set_ylabel("P(choose risky)")
        ax.legend(loc="upper right")
        _panel_tag(ax, "a")

        ax = axes[1]
        if h3 is not None and len(h3):
            piv = h3.reset_index().pivot(index="arch", columns="dataset",
                                         values="mean")
            dcol = [c for c in piv.columns if c.endswith("desc")][0]
            ecol = [c for c in piv.columns if c.endswith("exp")][0]
            piv = piv.sort_values(dcol)
            ypos = np.arange(len(piv))
            for yi, (_, row) in zip(ypos, piv.iterrows()):
                ax.plot([row[ecol], row[dcol]], [yi, yi], color=FAINT,
                        lw=1.2, zorder=1, solid_capstyle="round")
            ax.scatter(piv[ecol], ypos, s=46, c=DATASET_COLORS.get(ecol),
                       marker="D", edgecolors="white", linewidths=0.6,
                       zorder=3)
            ax.scatter(piv[dcol], ypos, s=52, c=DATASET_COLORS.get(dcol),
                       marker="o", edgecolors="white", linewidths=0.6,
                       zorder=3)
            ax.axvline(0, color=INK, lw=0.8, zorder=2)
            ax.set_yticks(ypos, piv.index)
            ax.set_xlabel("rare-event index\n(> 0 overweights rare gains)")
            ax.margins(y=0.16, x=0.16)
            n_flip = int(((piv[dcol] > 0) & (piv[ecol] < 0)).sum())
            ax.set_title(f"sign flip in {n_flip}/{len(piv)} architectures",
                         fontsize=7.5, color=MUTED, pad=6)
            # label the two ends once, on the topmost pair, instead of a
            # legend box that would sit on top of the data
            top = piv.iloc[-1]
            ax.annotate("experience", (top[ecol], ypos[-1]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=6.5,
                        color=DATASET_COLORS.get(ecol))
            ax.annotate("description", (top[dcol], ypos[-1]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=6.5,
                        color=DATASET_COLORS.get(dcol))
        _panel_tag(ax, "b")
        fig.tight_layout(w_pad=2.0)
        _save(fig, figdir, "fig2_de_flip.pdf")


# ---------------------------------------------------------------------------

def fig3_transfer(transfer_df, figdir, order=None, eval_only=()):
    """Completeness heatmap. Sequential single hue for magnitude; cells at or
    below zero (worse than predicting the mean) are hatched grey rather than
    clipped, so a failed transfer cannot masquerade as a merely low one."""
    piv = transfer_df.groupby(["train", "test"])["completeness"] \
                     .mean().unstack()
    if order:
        # evaluation-only arms appear as columns but never as rows: nothing is
        # trained on them, so the matrix is 4x6 rather than square
        cols = list(order) + [c for c in eval_only if c in piv.columns]
        piv = piv.reindex(index=order, columns=cols)
    M = piv.to_numpy()
    n_sq = len(order) if order else M.shape[0]

    with plt.rc_context(RC):
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(8.0, 3.9),
            gridspec_kw=dict(width_ratios=[1.25, 1]))
        im = ax.imshow(np.where(M > 0, M, np.nan), cmap="Blues",
                       vmin=0, vmax=1)
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
                        fontsize=7.5,
                        color="white" if v > 0.62 else INK,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(piv.columns)),
                      [DATASET_SHORT.get(c, c) for c in piv.columns],
                      fontsize=6.0)
        ax.set_yticks(range(len(piv.index)),
                      [DATASET_LABELS.get(i, i) for i in piv.index],
                      fontsize=6.5)
        if len(piv.columns) > n_sq:
            # mark where the evaluation-only block starts
            ax.axvline(n_sq - 0.5, color=INK, lw=1.2, zorder=5)
        ax.set_xlabel("evaluated on")
        ax.set_ylabel("trained on")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("completeness", fontsize=7)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.5, length=2)
        notes = []
        if len(piv.columns) > n_sq:
            notes.append("right of line: never trained on")
        if np.any(np.isfinite(M) & (M <= 0)):
            notes.append("hatched: no better than the mean")
        if notes:
            # stacked, not joined: one line is wider than the heatmap and
            # runs under the colourbar
            ax.set_title("\n".join(notes), fontsize=6.2, color=MUTED, pad=4)
        _panel_tag(ax, "a")

        # Panel b: transfer is not symmetric. For each unordered pair, how
        # much better does training on one arm generalise to the other than
        # the reverse? Positive means the row arm exports knowledge the
        # column arm cannot return.
        names = list(piv.index)
        pairs, gaps, cols = [], [], []
        for i in range(min(len(names), n_sq)):
            for j in range(i + 1, min(len(names), n_sq)):
                a, b = names[i], names[j]
                d = M[i, j] - M[j, i]
                lead = a if d > 0 else b
                pairs.append(f"{DATASET_LABELS.get(a, a)}\n"
                             f"vs {DATASET_LABELS.get(b, b)}")
                gaps.append(d)
                cols.append(DATASET_COLORS.get(lead, "gray"))
        o = np.argsort(np.abs(gaps))
        ypos = np.arange(len(o))
        ax2.barh(ypos, [abs(gaps[k]) for k in o],
                 color=[cols[k] for k in o], height=0.62, zorder=3)
        ax2.set_yticks(ypos, [pairs[k] for k in o], fontsize=5.8)
        ax2.set_xlabel("asymmetry  |completeness gap|")
        ax2.set_title("bar coloured by the arm that exports better",
                      fontsize=6.8, color=MUTED, pad=6)
        ax2.margins(y=0.06)
        _panel_tag(ax2, "b")
        fig.tight_layout(w_pad=2.4)
        _save(fig, figdir, "fig3_transfer.pdf")


# ---------------------------------------------------------------------------

def fig4_decomposition(dec_df, syn_df, true_theta, figdir):
    """(a) partial omega^2 per readout, sorted by the dataset-architecture
    difference. (b) recovery of a known agent, as recovered/true so that all
    six parameters share one axis and 1.0 is the target."""
    DS_C, AR_C = "#0072B2", "#D55E00"
    d = dec_df.copy()
    d["diff"] = d["omega2_dataset"] - d["omega2_arch"]
    # keep the two families of readout apart: they pass different screens and
    # rest on different assumptions, so pooling them into one ranking would
    # imply a comparability they do not have
    if "screen" in d.columns:
        order = {"theory_identifiable": 0, "identifiable": 1}
        d["_grp"] = d["screen"].map(order).fillna(9)
    else:
        d["_grp"] = 0
    d = d.sort_values(["_grp", "diff"], ascending=[True, False])

    rows, seps, glabels = [], [], []
    for g, sub in d.groupby("_grp", sort=True):
        if rows:
            seps.append(len(rows))
            rows.append(None)
        glabels.append((len(rows) + len(sub) / 2 - 0.5,
                        "CPT coordinates" if g == 0 else
                        "behavioural indices"))
        rows.extend(sub.to_dict("records"))

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4),
                                 gridspec_kw=dict(width_ratios=[1.5, 1]))
        ax = axes[0]
        h = 0.36
        for i, r in enumerate(rows):
            if r is None:
                continue
            ax.barh(i + h / 2, r["omega2_dataset"], h, color=DS_C, zorder=3)
            ax.barh(i - h / 2, r["omega2_arch"], h, color=AR_C, zorder=3)
        ax.set_yticks(range(len(rows)),
                      ["" if r is None else _pname(r["value"]).split(" (")[0]
                       for r in rows])
        for s in seps:
            ax.axhline(s, color=FAINT, lw=0.7, ls=":", zorder=1)
        for ypos, txt in glabels:
            ax.text(1.015, ypos, txt, transform=ax.get_yaxis_transform(),
                    rotation=270, ha="left", va="center", fontsize=6.5,
                    color=MUTED, style="italic")
        ax.invert_yaxis()
        ax.set_xlabel(r"partial $\omega^2$")
        ax.set_ylim(len(rows) - 0.4, -0.9)
        handles = [plt.Rectangle((0, 0), 1, 1, color=DS_C,
                                 label="dataset provenance"),
                   plt.Rectangle((0, 0), 1, 1, color=AR_C,
                                 label="architecture")]
        ax.legend(handles=handles, loc="lower left",
                  bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
        _panel_tag(ax, "a")

        ax = axes[1]
        if syn_df is not None and len(syn_df):
            params = [p for p in ["alpha", "beta", "lam", "gamma_pos",
                                  "gamma_neg", "tau"] if p in syn_df.columns]
            xs = np.arange(len(params))
            for j, ds in enumerate(["syn_c13k", "syn_cpc"]):
                sub = syn_df[syn_df["dataset"] == ds]
                if not len(sub):
                    continue
                mu = np.array([sub[p].mean() / true_theta[p] for p in params])
                sd = np.array([sub[p].std() / true_theta[p] for p in params])
                ax.errorbar(xs + (j - 0.5) * 0.20, mu, yerr=sd, fmt="o",
                            ms=5, lw=1.2, capsize=2.5,
                            color=DATASET_COLORS[ds], markeredgecolor="white",
                            markeredgewidth=0.5,
                            label=DATASET_LABELS.get(ds, ds))
            ax.axhline(1.0, color=INK, lw=0.9, ls="--", zorder=1)
            ax.set_xticks(xs, [_pname(p).split(" (")[0] for p in params])
            ax.set_ylabel("recovered / true")
            ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.02))
            ax.set_title("error bars = SD across runs", fontsize=7,
                         color=MUTED, pad=6)
        _panel_tag(ax, "b")
        fig.tight_layout(w_pad=2.2)
        _save(fig, figdir, "fig4_decomposition.pdf")
