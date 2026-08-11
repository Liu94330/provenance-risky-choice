"""Figures for the attribution readout.

Kept here rather than in code/src/figures_paper.py so the whole attribution
analysis -- Shapley computation, decomposition, figures and its test -- lives
in one directory. Styling (palette, rc block, panel tags) is imported from the
main figure module so these plots stay visually part of the same set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.figures import (DATASET_COLORS, DATASET_LABELS, DATASET_SHORT, FAINT,
                         INK, MUTED, RC, _panel_tag, _save)


def fig_attribution(fp, dec, figdir):
    """Attribution readout: which parts of a gamble each network relies on.

    The only readout here that assumes no functional form -- it asks how much
    the prediction moves when part of the input is withheld, nothing more. The
    Shapley values are exact over eleven semantic feature groups rather than
    sampled over 55 raw features, so architectures are comparable without an
    approximation-noise caveat.

    Panel c is the honest part: attribution has a much higher artifact floor
    than the other readouts, and only some groups clear it.
    """
    attr = [c for c in fp.columns if c.startswith("attr_")]
    names = [c.replace("attr_", "") for c in attr]
    prof = fp.groupby("dataset")[attr].mean()
    prof.columns = names
    order = prof.abs().max().sort_values(ascending=False).index.tolist()

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5),
                                 gridspec_kw=dict(width_ratios=[1.25, 1.15, 1]))

        ax = axes[0]
        y = np.arange(len(order))
        w = 0.2
        for k, ds in enumerate([d for d in DATASET_LABELS
                                if d in prof.index]):
            ax.barh(y + (k - 1.5) * w, prof.loc[ds, order], height=w * 0.92,
                    color=DATASET_COLORS.get(ds, "gray"), zorder=3,
                    label=DATASET_LABELS.get(ds, ds))
        ax.axvline(0, color=INK, lw=0.9, zorder=4)
        ax.set_yticks(y, order, fontsize=5.8)
        ax.invert_yaxis()
        ax.set_xlabel("normalised attribution")
        ax.legend(fontsize=5.4, loc="lower right")
        ax.set_title("what each arm's networks rely on", fontsize=6.4,
                     color=MUTED, pad=4)
        _panel_tag(ax, "a")

        ax = axes[1]
        d = dec.set_index("feature_group").reindex(order).reset_index()
        y = np.arange(len(d))
        ax.barh(y - 0.2, d["omega2_dataset"], height=0.36, color="#0072B2",
                zorder=3, label="provenance")
        ax.barh(y + 0.2, d["omega2_arch"], height=0.36, color="#D55E00",
                zorder=3, label="architecture")
        ax.scatter(d["dataset_artifact_floor"], y - 0.2, s=16, marker="|",
                   c=INK, zorder=5, linewidths=1.4)
        ax.scatter(d["arch_artifact_floor"], y + 0.2, s=16, marker="|",
                   c=INK, zorder=5, linewidths=1.4)
        ax.set_yticks(y, ["" for _ in y])
        ax.invert_yaxis()
        ax.set_xlabel(r"$\omega^2$")
        ax.legend(fontsize=5.6, loc="lower right")
        ax.set_title("ticks: each factor's own artifact floor", fontsize=6.4,
                     color=MUTED, pad=4)
        _panel_tag(ax, "b")

        ax = axes[2]
        g = "EV difference"
        col = f"attr_{g}"
        if col in fp.columns:
            dss = [d2 for d2 in DATASET_LABELS if d2 in fp["dataset"].values]
            for k, ds in enumerate(dss):
                v = fp[fp["dataset"] == ds][col]
                ax.scatter(np.full(len(v), k)
                           + np.linspace(-0.16, 0.16, len(v)), v, s=16,
                           c=DATASET_COLORS.get(ds, "gray"), alpha=0.85,
                           linewidths=0.3, edgecolors="white", zorder=3)
                ax.plot([k - 0.28, k + 0.28], [v.mean()] * 2, color=INK,
                        lw=1.6, zorder=4)
            ax.axhline(0, color=FAINT, lw=0.9, ls="--", zorder=1)
            ax.set_xticks(range(len(dss)),
                          [DATASET_SHORT.get(d2, d2) for d2 in dss],
                          fontsize=5.8)
            ax.set_ylabel("attribution to  EV difference")
            row = dec[dec["feature_group"] == g]
            if len(row):
                r = row.iloc[0]
                ax.set_title(
                    f"provenance $\\omega^2$={r['omega2_dataset']:.2f} "
                    f"vs floor {r['dataset_artifact_floor']:.2f}",
                    fontsize=6.2, color=MUTED, pad=4)
        _panel_tag(ax, "c")
        fig.tight_layout(w_pad=2.0)
        _save(fig, figdir, "fig_attribution.pdf")


def _beeswarm(ax, df, value="shap", colour="p", order=None, cmap="viridis",
              size=5.0, max_spread=0.38):
    """Beeswarm: one point per probe, positioned by its attribution and
    spread vertically by local density so overlapping points stay countable.

    The standard presentation for Shapley values, and the reason to keep the
    per-sample values: a group whose mean is near zero can still carry a wide
    two-sided spread, which is a model using it in opposite directions in
    different regions rather than not using it.
    """
    groups = order or list(df[value].groupby(df["feature_group"]).mean()
                           .sort_values().index)
    cvals = df[colour].to_numpy(dtype=float)
    lo, hi = np.nanpercentile(cvals, [2, 98])
    norm = plt.Normalize(lo, hi)
    cm = plt.get_cmap(cmap)
    for yi, g in enumerate(groups):
        sub = df[df["feature_group"] == g]
        v = sub[value].to_numpy(dtype=float)
        if not len(v):
            continue
        # density-based vertical offset: bin on x, spread within bin
        nb = 44
        edges = np.linspace(np.nanmin(v), np.nanmax(v) + 1e-12, nb + 1)
        which = np.clip(np.digitize(v, edges) - 1, 0, nb - 1)
        off = np.zeros(len(v))
        for b in range(nb):
            m = which == b
            n = int(m.sum())
            if n <= 1:
                continue
            k = np.arange(n) - (n - 1) / 2
            off[m] = k / max(np.abs(k).max(), 1) * min(
                max_spread, 0.06 * np.sqrt(n))
        ax.scatter(v, yi + off, s=size, c=cm(norm(sub[colour].to_numpy())),
                   linewidths=0, alpha=0.85, zorder=3)
    ax.axvline(0, color=INK, lw=0.9, zorder=4)
    ax.set_yticks(range(len(groups)), groups, fontsize=5.8)
    ax.set_ylim(-0.7, len(groups) - 0.3)
    return groups, plt.cm.ScalarMappable(norm=norm, cmap=cm)


def fig_attribution_rich(per, cross, fp, coords, figdir, arch="MLP_L"):
    """Attribution readout, per probe rather than averaged.

    (a) beeswarm over every probe for one arm -- the spread is the point
    (b) the same group for all four arms, so the provenance difference is
        visible as a distribution rather than a bar
    (c) dependence: how the attribution to EV difference varies with the
        actual EV difference, which is the model's functional form
    (d) conditional map across probability bands and domain
    (e) cross-check against the CPT readout, computed by an unrelated route
    """
    ref = per[per["dataset"] == "c13k_exp"]
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(11.4, 6.6))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.32, 1, 1],
                              height_ratios=[1, 1], hspace=0.46, wspace=0.42)

        ax = fig.add_subplot(gs[:, 0])
        order, sm = _beeswarm(ax, ref)
        ax.set_xlabel("Shapley value  (probe-level)")
        ax.set_title(f"{arch} trained on choices13k experience\n"
                     f"{len(ref) // max(ref['feature_group'].nunique(), 1)} "
                     "probes, colour = probability of the risky gain",
                     fontsize=6.4, color=MUTED, pad=6)
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("p", fontsize=6.0)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.6, length=2)
        _panel_tag(ax, "a")

        ax = fig.add_subplot(gs[0, 1])
        g = "EV difference"
        dss = [d for d in DATASET_LABELS if d in per["dataset"].unique()]
        # Model-level mean attribution, which is what the omega^2 in
        # table 20 decomposes. Plotting the per-probe values here instead
        # would drown a +-0.06 between-arm difference in a +-0.4 within-model
        # spread and wrongly read as "no separation".
        if fp is not None and f"attr_{g}" in fp.columns:
            for k, ds in enumerate(dss):
                v = fp[fp["dataset"] == ds][f"attr_{g}"].dropna()
                if not len(v):
                    continue
                ax.scatter(np.full(len(v), k)
                           + np.linspace(-0.17, 0.17, len(v)), v, s=20,
                           c=DATASET_COLORS.get(ds, "gray"), alpha=0.85,
                           linewidths=0.3, edgecolors="white", zorder=3)
                ax.plot([k - 0.28, k + 0.28], [v.mean()] * 2, color=INK,
                        lw=1.7, zorder=4)
            ax.axhline(0, color=FAINT, lw=0.9, ls="--", zorder=1)
            ax.set_xticks(range(len(dss)),
                          [DATASET_SHORT.get(d, d) for d in dss],
                          fontsize=5.6)
            ax.set_ylabel(f"model-level attribution\n{g} (normalised)")
            ax.set_title("one point per trained model (5 archs x 5 seeds)",
                         fontsize=6.2, color=MUTED, pad=5)
        _panel_tag(ax, "b")

        ax = fig.add_subplot(gs[0, 2])
        for ds in dss:
            v = per[(per["dataset"] == ds) & (per["feature_group"] == g)]
            if "dEV" not in v.columns or v["dEV"].isna().all():
                continue
            v = v.sort_values("dEV")
            ax.scatter(v["dEV"], v["shap"], s=4,
                       c=DATASET_COLORS.get(ds, "gray"), alpha=0.35,
                       linewidths=0, zorder=2)
            q = pd.qcut(v["dEV"], 8, duplicates="drop")
            tr = v.groupby(q, observed=True).agg(x=("dEV", "mean"),
                                                 y=("shap", "mean"))
            ax.plot(tr["x"], tr["y"], lw=1.8,
                    color=DATASET_COLORS.get(ds, "gray"), zorder=4)
        ax.axhline(0, color=FAINT, lw=0.8, ls="--", zorder=1)
        ax.axvline(0, color=FAINT, lw=0.8, ls="--", zorder=1)
        lo, hi = np.nanpercentile(per["dEV"].dropna(), [1, 99])
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"actual EV difference  (B $-$ A)")
        ax.set_ylabel("Shapley value")
        ax.set_title("dependence: thick line = binned mean per arm",
                     fontsize=6.2, color=MUTED, pad=5)
        _panel_tag(ax, "c")

        ax = fig.add_subplot(gs[1, 1])
        bands = [(0.0, 0.06, "p<=.05"), (0.06, 0.35, ".1-.3"),
                 (0.35, 0.75, ".5-.7"), (0.75, 1.01, "p>=.8")]
        cells, ylab = [], []
        for lo, hi, nm in bands:
            for dom in ["gains", "losses"]:
                m = per[(per["p"] >= lo) & (per["p"] < hi)
                        & (per["domain"] == dom)]
                cells.append([m[(m["dataset"] == ds)
                                & (m["feature_group"] == g)]["shap"].mean()
                              for ds in dss])
                ylab.append(f"{nm}  {dom[:4]}")
        M = np.array(cells, dtype=float)
        lim = np.nanmax(np.abs(M)) or 1e-9
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                            fontsize=5.0,
                            color="white" if abs(M[i, j]) > lim * 0.6
                            else INK)
        ax.set_xticks(range(len(dss)),
                      [DATASET_SHORT.get(d, d).replace("\n", " ")
                       for d in dss], fontsize=5.2, rotation=20, ha="right")
        ax.set_yticks(range(len(ylab)), ylab, fontsize=5.2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Shapley", fontsize=5.8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.4, length=2)
        ax.set_title("conditional on where in probability space",
                     fontsize=6.2, color=MUTED, pad=5)
        _panel_tag(ax, "d")

        ax = fig.add_subplot(gs[1, 2])
        if (fp is not None and coords is not None
                and cross is not None and len(cross)):
            co = coords[coords["theory_identifiable"]]
            key = ["dataset", "arch", "seed"]
            m = fp.merge(co[key + ["lam"]], on=key, how="inner")
            if len(m) and "attr_P(loss) of B" in m.columns:
                for ds in [d for d in DATASET_LABELS
                           if d in m["dataset"].unique()]:
                    s2 = m[m["dataset"] == ds]
                    ax.scatter(s2["attr_P(loss) of B"], s2["lam"], s=20,
                               c=DATASET_COLORS.get(ds, "gray"), alpha=0.85,
                               linewidths=0.3, edgecolors="white", zorder=3)
                r = cross[cross["attribution"] == "P(loss) of B"]
                rho = r["spearman_rho"].iloc[0] if len(r) else np.nan
                ax.set_xlabel("attribution to  P(loss) of B")
                ax.set_ylabel(r"fitted  $\lambda$")
                ax.set_title(f"independent readouts agree: "
                             fr"$\rho$={rho:+.2f}", fontsize=6.2,
                             color=MUTED, pad=5)
        _panel_tag(ax, "e")

        _save(fig, figdir, "fig_attribution_rich.pdf")
