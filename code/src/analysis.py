"""Headline statistics.

H1  Two-factor variance decomposition of every theory coordinate / behavioural
    index: partial omega^2 for dataset vs architecture (seeds = replicates),
    plus a multivariate permutation test (PERMANOVA-style pseudo-F) on the
    full behavioural fingerprints, and silhouette separation by labels.
H2  Ground-truth recovery bias in the synthetic arms.
H4  Cross-provenance transfer matrix with noise-ceiling normalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# H1: variance decomposition (balanced two-way ANOVA, partial omega^2)
# ---------------------------------------------------------------------------

def two_way_omega2(df, value, f1="dataset", f2="arch"):
    """Two-way fixed-effects ANOVA on `value`; returns partial omega^2 (and
    eta^2) for both factors and their interaction.

    Uses the unweighted-means (Type III style) decomposition so that dropping
    runs unevenly across cells -- which the identifiability filter does -- does
    not let one factor absorb the other's variance through unequal cell sizes.
    With equal cell sizes this reduces exactly to the balanced formulas."""
    y = df[value].to_numpy(dtype=float)
    N = len(y)
    ss_tot = ((y - y.mean()) ** 2).sum()

    cell_mean = df.groupby([f1, f2])[value].mean().unstack()
    cell_n = df.groupby([f1, f2])[value].size().unstack()
    if cell_mean.isna().any().any():
        raise ValueError(f"{value}: empty cell(s) in the {f1} x {f2} design; "
                         "unweighted-means ANOVA is undefined")
    a, b = cell_mean.shape
    n_h = a * b / (1.0 / cell_n.to_numpy()).sum()      # harmonic mean cell n
    grand_u = cell_mean.to_numpy().mean()

    ss1 = n_h * b * ((cell_mean.mean(axis=1) - grand_u) ** 2).sum()
    ss2 = n_h * a * ((cell_mean.mean(axis=0) - grand_u) ** 2).sum()
    resid = (cell_mean.to_numpy()
             - cell_mean.mean(axis=1).to_numpy()[:, None]
             - cell_mean.mean(axis=0).to_numpy()[None, :] + grand_u)
    ss_int = n_h * (resid ** 2).sum()
    # within-cell error is unaffected by the weighting scheme
    ss_err = float(((df[value] - df.groupby([f1, f2])[value]
                     .transform("mean")) ** 2).sum())

    df1, df2, dfi = a - 1, b - 1, (a - 1) * (b - 1)
    df_err = max(N - a * b, 1)
    ms_err = ss_err / df_err

    def omega_p(ss, dfx):
        return max((ss - dfx * ms_err) / (ss + (N - dfx) * ms_err), 0.0)

    # under unequal cell sizes the four unweighted SS no longer add up to the
    # raw total, so eta^2 is normalised by their sum (identical to ss_tot when
    # the design is balanced)
    ss_ref = ss1 + ss2 + ss_int + ss_err

    return dict(
        value=value, n=N,
        omega2_dataset=omega_p(ss1, df1),
        omega2_arch=omega_p(ss2, df2),
        omega2_interaction=omega_p(ss_int, dfi),
        eta2_dataset=ss1 / ss_ref, eta2_arch=ss2 / ss_ref,
        eta2_interaction=ss_int / ss_ref, eta2_error=ss_err / ss_ref,
    )


def probe_level_omega2(F, f1_labels, f2_labels):
    """Run the same two-way decomposition on every column of F at once.

    F is [n_runs, n_probes]. Looping two_way_omega2 over ~1,700 probes takes
    minutes; the design is identical for every column, so the cell means and
    sums of squares are computed as array operations instead. Returns
    (omega2_f1, omega2_f2), each [n_probes].
    """
    f1 = pd.Index(pd.unique(f1_labels))
    f2 = pd.Index(pd.unique(f2_labels))
    a, b, N = len(f1), len(f2), F.shape[0]
    i1 = f1.get_indexer(f1_labels)
    i2 = f2.get_indexer(f2_labels)
    cell = i1 * b + i2

    counts = np.bincount(cell, minlength=a * b).astype(float)
    if (counts == 0).any():
        raise ValueError("empty cell in the probe-level design")
    sums = np.stack([np.bincount(cell, weights=F[:, j], minlength=a * b)
                     for j in range(F.shape[1])], axis=1)
    means = (sums / counts[:, None]).reshape(a, b, -1)     # [a, b, P]

    n_h = a * b / (1.0 / counts).sum()
    grand = means.mean((0, 1))                             # [P]
    m1, m2 = means.mean(1), means.mean(0)                  # [a,P] , [b,P]
    ss1 = n_h * b * ((m1 - grand) ** 2).sum(0)
    ss2 = n_h * a * ((m2 - grand) ** 2).sum(0)
    resid = means - m1[:, None, :] - m2[None, :, :] + grand
    ss_int = n_h * (resid ** 2).sum((0, 1))
    fitted = means.reshape(a * b, -1)[cell]
    ss_err = ((F - fitted) ** 2).sum(0)

    df1, df2 = a - 1, b - 1
    ms_err = ss_err / max(N - a * b, 1)

    def omega_p(ss, dfx):
        return np.maximum((ss - dfx * ms_err)
                          / (ss + (N - dfx) * ms_err), 0.0)

    return omega_p(ss1, df1), omega_p(ss2, df2)


def experience_trend(df, value, block_col="block", arch_col="arch",
                     n_perm=5000, seed=0):
    """Is `value` monotone in experience stage, and is the trend shared?

    Blocks are ordered, so an unordered ANOVA throws away the hypothesis. For
    each architecture the Spearman correlation between block index and the
    readout is computed across seeds; the architecture-level correlations are
    then averaged. Significance comes from permuting block labels WITHIN
    architecture, which keeps each architecture's own level and spread and
    tests only the ordering.

    Returns mean rho, the per-architecture rhos, and a two-sided p-value.
    """
    from scipy.stats import spearmanr

    sub = df[[block_col, arch_col, value]].dropna()
    if sub[block_col].nunique() < 3:
        return None

    def mean_rho(vals):
        rs = []
        for a, g in sub.assign(_v=vals).groupby(arch_col):
            if g[block_col].nunique() < 3 or g["_v"].std() == 0:
                continue
            r = spearmanr(g[block_col], g["_v"]).correlation
            if np.isfinite(r):
                rs.append(r)
        return (float(np.mean(rs)) if rs else np.nan), rs

    obs, per_arch = mean_rho(sub[value].to_numpy())
    if not np.isfinite(obs):
        return None

    rng = np.random.default_rng(seed)
    v = sub[value].to_numpy()
    groups = [np.where(sub[arch_col].to_numpy() == a)[0]
              for a in pd.unique(sub[arch_col])]
    hits = 0
    for _ in range(n_perm):
        vp = v.copy()
        for idx in groups:
            vp[idx] = rng.permutation(vp[idx])
        r, _ = mean_rho(vp)
        if np.isfinite(r) and abs(r) >= abs(obs):
            hits += 1
    return dict(value=value, mean_rho=obs, n_arch=len(per_arch),
                min_rho=float(np.min(per_arch)),
                max_rho=float(np.max(per_arch)),
                same_sign=int(np.sum(np.sign(per_arch) == np.sign(obs))),
                p_value=(hits + 1) / (n_perm + 1))


def recovery_error(df, true_theta, params, group_cols):
    """Relative recovery error of each known parameter, per design cell.

    Reported as |recovered - true| / |true| so parameters on different scales
    (tau ~ 6, gamma ~ 0.65) can be averaged into one number per cell. This is
    what converts the sufficiency sweep into a threshold someone can act on.
    """
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys), n_runs=len(g))
        errs = []
        for p in params:
            if p not in g.columns or not len(g[p].dropna()):
                continue
            est = g[p].mean()
            truth = true_theta[p]
            rel = abs(est - truth) / max(abs(truth), 1e-9)
            rec[f"err_{p}"] = rel
            errs.append(rel)
        if not errs:
            continue
        rec["mean_rel_err"] = float(np.mean(errs))
        rec["max_rel_err"] = float(np.max(errs))
        rows.append(rec)
    return pd.DataFrame(rows)


def sufficiency_threshold(err_df, size_col, tol=0.10, err_col="mean_rel_err"):
    """Smallest swept arm size whose error is within `tol`, per respondent
    count. Returns NaN where no swept size reaches the tolerance, which is
    itself the useful answer -- it says the tolerance is unreachable at that
    number of respondents, not that the design is fine."""
    out = []
    for keys, g in err_df.groupby([c for c in err_df.columns
                                   if c not in (size_col, "n_runs", err_col)
                                   and not c.startswith("err_")
                                   and not c.startswith("max_")]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ok = g[g[err_col] <= tol]
        rec = dict(zip([c for c in err_df.columns
                        if c not in (size_col, "n_runs", err_col)
                        and not c.startswith("err_")
                        and not c.startswith("max_")], keys))
        rec[f"min_{size_col}_within_{tol:g}"] = (
            ok[size_col].min() if len(ok) else np.nan)
        rec["best_error"] = g[err_col].min()
        out.append(rec)
    return pd.DataFrame(out)


def cell_resampler(df, strata):
    """Within-cell bootstrap: returns draw(rng) -> a resample of `df` in which
    every design cell (defined by `strata`) is resampled with replacement to
    its own size, so the layout stays balanced in every replicate. Works on
    positional indices, which is an order of magnitude faster than a
    groupby/apply with per-cell sampling and gives the same distribution."""
    cells = [np.asarray(ix) for ix in df.groupby(strata).indices.values()]

    def draw(rng):
        take = np.concatenate([c[rng.integers(0, len(c), len(c))]
                               for c in cells])
        return df.iloc[take]
    return draw


def bootstrap_omega2(df, value, effect, strata, estimator, n_boot=400,
                     seed=0, return_samples=False):
    """Percentile CI for one omega^2, resampling runs within design cells.

    Comparing an observed effect with an artifact floor is only meaningful
    with intervals: both are point estimates from a few hundred runs, and
    without them "0.80 versus 0.75" says nothing. Resampling happens within
    cells so the design stays balanced in every replicate.

    `effect` may be one key or a list of keys of the estimator's result; the
    estimator runs once per replicate either way, so asking for several
    effects at once (e.g. both factors of a two-way decomposition) costs
    nothing extra. With `return_samples` the full bootstrap distribution is
    returned as well, for tests that need more than an interval.
    """
    rng = np.random.default_rng(seed)
    effects = [effect] if isinstance(effect, str) else list(effect)
    draw = cell_resampler(df, strata)
    out = {e: [] for e in effects}
    for _ in range(n_boot):
        try:
            r = estimator(draw(rng))
        except Exception:
            continue
        if r is None:
            continue
        for e in effects:
            v = r.get(e) if isinstance(r, dict) else r
            if v is not None and np.isfinite(v):
                out[e].append(v)

    def summarise(vals):
        if not vals:
            d = dict(value=value, lo=np.nan, hi=np.nan, n_boot=0)
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            d = dict(value=value, lo=float(lo), hi=float(hi),
                     n_boot=len(vals))
        if return_samples:
            d["samples"] = np.asarray(vals, dtype=float)
        return d
    if isinstance(effect, str):
        return summarise(out[effect])
    return {e: summarise(out[e]) for e in effects}


def artifact_corrected(real, artifact_arch, artifact_pool, params):
    """Compare each observed effect against what the readout produces when
    there is nothing to find.

    On arms generated by a single fixed agent the true coordinates are
    identical for every run, so any omega^2 the readout still reports is
    manufactured by the readout itself: different architectures fit slightly
    different prediction surfaces, and the CPT fit turns those into different
    coordinates. That level is the floor an observed effect has to clear
    before it can be called a difference in encoded theory.

    The two floors are not interchangeable. Architecture perturbs the
    prediction surface directly and so has a high floor; swapping the problem
    pool barely moves it. Each observed effect is therefore compared with the
    floor for its own factor.

    `artifact_arch` and `artifact_pool` map parameter -> omega^2 measured on
    the fixed-agent arms.
    """
    rows = []
    for p in params:
        if p not in real.index:
            continue
        r = real.loc[p]
        fa = artifact_arch.get(p, np.nan)
        fp = artifact_pool.get(p, np.nan)
        rows.append(dict(
            value=p,
            arch_observed=r["omega2_arch"],
            arch_artifact_floor=fa,
            arch_excess=r["omega2_arch"] - fa,
            arch_survives=bool(r["omega2_arch"] > fa),
            dataset_observed=r["omega2_dataset"],
            dataset_artifact_floor=fp,
            dataset_excess=r["omega2_dataset"] - fp,
            dataset_survives=bool(r["omega2_dataset"] > fp),
        ))
    return pd.DataFrame(rows)


def factorial_omega2(df, value, factors, arch_col="arch"):
    """Effect size of each manipulated factor, plus architecture, on `value`.

    For the controlled synthetic design: every factor is crossed with every
    other, so unlike the observational arms there is no confounding to argue
    about and each factor's share is interpretable on its own. Architecture is
    entered as one more factor so the comparison that matters -- provenance
    components against architecture -- is made on a single scale.

    Uses a main-effects-only decomposition (each factor's sum of squares from
    its own marginal means, error from the full-cell residual), which is what
    a balanced factorial supports without fitting every interaction.
    """
    d = df.dropna(subset=[value]).copy()
    cols = list(factors) + ([arch_col] if arch_col else [])
    cols = [c for c in cols if c in d.columns and d[c].nunique() > 1]
    if len(cols) < 2:
        return None
    y = d[value].to_numpy(dtype=float)
    N = len(y)
    grand = y.mean()

    ss = {}
    for c in cols:
        s = 0.0
        for _, sub in d.groupby(c):
            s += len(sub) * (sub[value].mean() - grand) ** 2
        ss[c] = s
    cell = d.groupby(cols)[value]
    ss_err = float(((d[value] - cell.transform("mean")) ** 2).sum())
    df_err = max(N - cell.ngroups, 1)
    ms_err = ss_err / df_err

    out = dict(value=value, n=N)
    for c in cols:
        dfx = d[c].nunique() - 1
        out[f"omega2_{c}"] = max(
            (ss[c] - dfx * ms_err) / (ss[c] + (N - dfx) * ms_err), 0.0)
    return out


def abstraction_ladder(levels, f1_labels, f2_labels):
    """Distance-based R^2 for both factors at several levels of abstraction.

    `levels` is an ordered list of (name, matrix) with the same rows. Each
    matrix is column-standardised, so the comparison across levels is between
    shares of that level's own variance, not raw magnitudes -- which is what
    makes 1,742 raw predictions comparable with 6 fitted coordinates.
    """
    def r2(F, labels):
        Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
        D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        n = len(labels)
        sst = D2[np.triu_indices(n, 1)].sum() / n
        ssw = 0.0
        for g in pd.unique(labels):
            i = np.where(np.asarray(labels) == g)[0]
            if len(i) > 1:
                s = D2[np.ix_(i, i)]
                ssw += s[np.triu_indices(len(i), 1)].sum() / len(i)
        return float((sst - ssw) / sst)

    rows = []
    for name, F in levels:
        r_ds, r_ar = r2(F, f1_labels), r2(F, f2_labels)
        rows.append(dict(level=name, dim=F.shape[1], r2_dataset=r_ds,
                         r2_arch=r_ar, ratio=r_ds / r_ar if r_ar else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H1 multivariate: restricted-permutation pseudo-F on fingerprints
# ---------------------------------------------------------------------------

def _pseudo_f(D2, labels):
    """PERMANOVA pseudo-F from a squared distance matrix and group labels."""
    n = len(labels)
    groups = pd.unique(labels)
    sst = D2[np.triu_indices(n, 1)].sum() / n
    ssw = 0.0
    for g in groups:
        idx = np.where(labels == g)[0]
        if len(idx) > 1:
            sub = D2[np.ix_(idx, idx)]
            ssw += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
    ssb = sst - ssw
    a = len(groups)
    return (ssb / (a - 1)) / (ssw / max(n - a, 1))


def permanova(F, labels, strata, n_perm=2000, seed=0):
    """Test the effect of `labels` on fingerprint matrix F [n, d], permuting
    labels WITHIN each stratum (so the other factor is held fixed)."""
    rng = np.random.default_rng(seed)
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    labels = np.asarray(labels); strata = np.asarray(strata)
    f_obs = _pseudo_f(D2, labels)
    cnt = 0
    for _ in range(n_perm):
        perm = labels.copy()
        for s in pd.unique(strata):
            idx = np.where(strata == s)[0]
            perm[idx] = rng.permutation(perm[idx])
        if _pseudo_f(D2, perm) >= f_obs:
            cnt += 1
    return dict(pseudo_F=float(f_obs), p_value=(cnt + 1) / (n_perm + 1))


def silhouette_by(F, labels):
    from sklearn.metrics import silhouette_score
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    if len(pd.unique(labels)) < 2:
        return np.nan
    return float(silhouette_score(Z, labels))


# ---------------------------------------------------------------------------
# H4: transfer / completeness
# ---------------------------------------------------------------------------

def completeness(mse, var_y, floor_mse):
    """Noise-ceiling-normalised accuracy in [~0, 1]:
    (1 - mse/var) / (1 - floor/var). 1 = explains everything explainable."""
    denom = 1.0 - floor_mse / max(var_y, 1e-12)
    return (1.0 - mse / max(var_y, 1e-12)) / max(denom, 1e-12)


def transfer_table(runs_df, meta):
    """runs_df needs columns: dataset (train), arch, seed, and one column
    'mse_on_<d>' per evaluation dataset. Returns tidy long table with raw MSE
    and completeness, averaged over seeds."""
    eval_cols = [c for c in runs_df.columns if c.startswith("mse_on_")]
    rows = []
    for (dtr, arch), sub in runs_df.groupby(["dataset", "arch"]):
        for c in eval_cols:
            dte = c.replace("mse_on_", "")
            mse = sub[c].mean()
            rows.append(dict(
                train=dtr, test=dte, arch=arch, mse=mse,
                completeness=completeness(mse, meta[dte]["var_y"],
                                          meta[dte]["noise_floor_mse"])))
    return pd.DataFrame(rows)
