"""Robustness of the floor-clearing result -> results/tables/robustness_*.csv,
results/tables/symbolic_*.csv and results/robustness.md

Four questions a referee will put to the headline (Fig. 2), answered on the
same 2,240 runs without retraining anything:

  1. Floor definition. 06 measures the provenance floor as the problem-pool
     share of the controlled factorial. But the two corpora also differ in arm
     size and respondents per problem, and the same factorial measures those
     shares too. Alternative floors: the arm-size share, the respondent share,
     the largest of the three, and a composite floor in which all three
     non-behavioural components of provenance enter as one factor (so their
     interactions count against provenance as well). Architecture also gets a
     second floor, from the controlled suite (640 runs) instead of the
     sufficiency suite (480).
  2. Family-wise error. Eleven read-outs per factor are tested; the paper's
     "interval strictly above the floor interval" rule is applied per read-out.
     Here each comparison gets a bootstrap p-value for H0: observed <= floor,
     and the eleven are Holm-adjusted.
  3. Leave-one-architecture-out. GBT is deterministic, so its ten seeds are
     ten identical fits that add no within-cell variance, and LINEAR/SET are
     the ones the screens remove on the small arms. Dropping each architecture
     in turn shows whether any single one carries -- or sinks -- the result.
  4. H5 made testable. The symbolic-regression expressions are evaluated on
     the probe grid, so each run becomes a function rather than a string, and
     the same PERMANOVA / silhouette machinery as the fingerprints asks whether
     those functions sort by provenance or by architecture.

Run after 05 and 06: python3 code/scripts/11_robustness.py
"""
import _boot  # noqa
import os
import re

import numpy as np
import pandas as pd

from src.analysis import (bootstrap_omega2, factorial_omega2, permanova,
                          silhouette_by, two_way_omega2)
from src.models import ARCHS
from src.probes import PG, XG
from src.theoryfit import PARAM_NAMES
from src.train import RESULTS

MAIN = ["c13k_exp", "c13k_desc", "cpc18_exp", "cpc18_desc"]
COORDS = list(PARAM_NAMES)
INDICES = ["rare_event_index", "certainty_index", "reflection_index",
           "loss_aversion_kappa", "ev_consistency"]
FACTORS = ["problems", "agent", "n_subj", "n_problems"]
MIN_CELL = 3
N_BOOT = 400        # same as 06
N_BOOT_JK = 200     # jackknife: 8 x 11 x 3 bootstraps, so lighter
SEED = 0
ALPHA = 0.05

TABDIR = os.path.join(RESULTS, "tables")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load():
    coords = pd.read_csv(os.path.join(TABDIR, "theory_coords.csv"))
    idx = pd.read_csv(os.path.join(TABDIR, "indices.csv"))
    keep = [c for c in idx.columns if c not in coords.columns
            or c == "run_id"]
    df = coords.merge(idx[keep], on="run_id")
    base = df[df["subsample_n"].isna() & df["reweight_to"].isna()].copy()
    main_all = base[base["dataset"].isin(MAIN)].copy()

    sim = base[base["dataset"].str.startswith("sim_", na=False)].copy()
    f = sim["dataset"].str.extract(
        r"sim_(?P<problems>[a-z0-9]+)_(?P<agent>desc|exp)"
        r"_n(?P<n_subj>\d+)_p(?P<n_problems>\d+)")
    sim = pd.concat([sim, f], axis=1)
    # one factor holding everything about provenance except the behaviour
    sim["nuisance"] = (sim["problems"] + "_n" + sim["n_subj"]
                       + "_p" + sim["n_problems"])

    suf = base[base["dataset"].str.startswith("suf_", na=False)].copy()
    g = suf["dataset"].str.extract(r"suf_p(?P<n_problems>\d+)_n(?P<n_subj>\d+)")
    suf["n_problems"] = g["n_problems"].astype(int)
    suf["n_subj"] = g["n_subj"].astype(int)
    return main_all, sim, suf


def screen_of(v):
    return "theory_identifiable" if v in COORDS else "identifiable"


def dropped_archs(main_all, screen, min_cell=MIN_CELL):
    d = main_all[main_all[screen]]
    cnt = d.groupby(["dataset", "arch"]).size().unstack(fill_value=0)
    return cnt.columns[(cnt < min_cell).any()].tolist()


def design_sets(main_all, sim, suf, exclude=()):
    """(observed, controlled, sufficiency) frames per screen, on a common
    architecture set: whatever the screen drops, plus `exclude`."""
    out = {}
    for scr in ["theory_identifiable", "identifiable"]:
        m = main_all[~main_all["arch"].isin(exclude)]
        dr = set(dropped_archs(m, scr)) | set(exclude)
        R = main_all[main_all[scr] & ~main_all["arch"].isin(dr)]
        M = sim[sim[scr] & ~sim["arch"].isin(dr)]
        S = suf[suf[scr] & ~suf["arch"].isin(dr)]
        out[scr] = (R, M, S, sorted(dr))
    return out


# ---------------------------------------------------------------------------
# bootstraps: observed effects and every floor, with full distributions
# ---------------------------------------------------------------------------

def boot_all(R, M, S, v, n_boot, seed=SEED):
    """Bootstrap distributions (within-cell resampling) of the observed
    two-way effects and of every floor definition for one read-out."""
    obs = bootstrap_omega2(
        R, v, ["omega2_arch", "omega2_dataset"], ["dataset", "arch"],
        lambda s, v=v: two_way_omega2(s, v), n_boot=n_boot, seed=seed,
        return_samples=True)
    f_suf = bootstrap_omega2(
        S, v, "omega2_arch", ["n_problems", "n_subj", "arch"],
        lambda s, v=v: factorial_omega2(s, v, ["n_problems", "n_subj"]),
        n_boot=n_boot, seed=seed, return_samples=True)
    f_ctrl = bootstrap_omega2(
        M, v, ["omega2_arch", "omega2_problems", "omega2_n_problems",
               "omega2_n_subj"], FACTORS + ["arch"],
        lambda s, v=v: factorial_omega2(s, v, FACTORS),
        n_boot=n_boot, seed=seed, return_samples=True)
    f_comp = bootstrap_omega2(
        M, v, "omega2_nuisance", ["nuisance", "agent", "arch"],
        lambda s, v=v: factorial_omega2(s, v, ["nuisance", "agent"]),
        n_boot=n_boot, seed=seed, return_samples=True)
    comp = {k: f_ctrl[k]["samples"] for k in f_ctrl}
    n = min(len(comp["omega2_problems"]), len(comp["omega2_n_problems"]),
            len(comp["omega2_n_subj"]))
    max3 = np.max(np.stack([comp["omega2_problems"][:n],
                            comp["omega2_n_problems"][:n],
                            comp["omega2_n_subj"][:n]]), axis=0)
    return dict(
        obs_arch=obs["omega2_arch"]["samples"],
        obs_ds=obs["omega2_dataset"]["samples"],
        floors_arch={"sufficiency suite": f_suf["samples"],
                     "controlled suite": comp["omega2_arch"]},
        floors_ds={"problem pool": comp["omega2_problems"],
                   "arm size": comp["omega2_n_problems"],
                   "respondents": comp["omega2_n_subj"],
                   "largest component": max3,
                   "composite (pool x size x respondents)":
                       f_comp["samples"]})


def point_all(R, M, S, v):
    """Point estimates on the same frames."""
    o = two_way_omega2(R, v)
    fs = factorial_omega2(S, v, ["n_problems", "n_subj"])
    fc = factorial_omega2(M, v, FACTORS)
    fn = factorial_omega2(M, v, ["nuisance", "agent"])
    comp3 = [fc["omega2_problems"], fc["omega2_n_problems"],
             fc["omega2_n_subj"]]
    return dict(
        obs_arch=o["omega2_arch"], obs_ds=o["omega2_dataset"], n=o["n"],
        floors_arch={"sufficiency suite": fs["omega2_arch"],
                     "controlled suite": fc["omega2_arch"]},
        floors_ds={"problem pool": fc["omega2_problems"],
                   "arm size": fc["omega2_n_problems"],
                   "respondents": fc["omega2_n_subj"],
                   "largest component": max(comp3),
                   "composite (pool x size x respondents)":
                       fn["omega2_nuisance"]})


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def p_exceeds(obs, floor):
    """Bootstrap p for H0: observed <= floor, over all (obs, floor) pairs.
    The two distributions come from independent resamples, so every pair is
    a draw from the difference. Floored at 1/(pairs+1) so it never prints 0."""
    diff = obs[:, None] - floor[None, :]
    k = int((diff <= 0).sum())
    return (k + 1) / (diff.size + 1)


def holm(p):
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


# ---------------------------------------------------------------------------
# 1 + 2: floor definitions and family-wise correction
# ---------------------------------------------------------------------------

def floor_definition_table(sets, samples=None):
    """`samples`, if a dict, receives every bootstrap distribution keyed
    '<readout>|<quantity>' so the figure can draw the distributions rather
    than only their endpoints."""
    rows = []
    for v in COORDS + INDICES:
        scr = screen_of(v)
        R, M, S, dr = sets[scr]
        pt = point_all(R, M, S, v)
        bt = boot_all(R, M, S, v, N_BOOT)
        if samples is not None:
            samples[f"{v}|obs_ds"] = bt["obs_ds"]
            samples[f"{v}|obs_arch"] = bt["obs_arch"]
            for k, arr in bt["floors_ds"].items():
                samples[f"{v}|floor_ds|{k}"] = arr
            for k, arr in bt["floors_arch"].items():
                samples[f"{v}|floor_arch|{k}"] = arr
        fam = "CPT coordinate" if v in COORDS else "behavioural index"
        for factor, okey, fkey in [("architecture", "obs_arch", "floors_arch"),
                                   ("provenance", "obs_ds", "floors_ds")]:
            o_lo, o_hi = ci(bt[okey])
            for fdef, fsamp in bt[fkey].items():
                f_lo, f_hi = ci(fsamp)
                rows.append(dict(
                    factor=factor, floor_def=fdef, value=v, family=fam,
                    n=pt["n"], dropped_archs=",".join(dr),
                    obs=pt[okey], obs_lo=o_lo, obs_hi=o_hi,
                    floor=pt[fkey][fdef], floor_lo=f_lo, floor_hi=f_hi,
                    clears_point=bool(pt[okey] > pt[fkey][fdef]),
                    clears_interval=bool(o_lo > f_hi),
                    p_raw=p_exceeds(bt[okey], fsamp)))
        print(f"  {v}")
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for (factor, fdef), g in t.groupby(["factor", "floor_def"]):
        t.loc[g.index, "p_holm"] = holm(g["p_raw"].to_numpy())
    t["clears_holm"] = t["p_holm"] < ALPHA
    return t


def floor_summary(t):
    s = (t.groupby(["factor", "floor_def"], sort=False)
          .agg(n_readouts=("value", "size"),
               clears_point=("clears_point", "sum"),
               clears_interval=("clears_interval", "sum"),
               clears_holm=("clears_holm", "sum"),
               median_floor=("floor", "median"),
               median_obs=("obs", "median"))
          .reset_index())
    return s


# ---------------------------------------------------------------------------
# 3: leave-one-architecture-out
# ---------------------------------------------------------------------------

def jackknife_table(main_all, sim, suf):
    rows = []
    for left_out in ["none"] + list(ARCHS):
        exclude = () if left_out == "none" else (left_out,)
        sets = design_sets(main_all, sim, suf, exclude=exclude)
        for v in COORDS + INDICES:
            scr = screen_of(v)
            R, M, S, dr = sets[scr]
            if R["arch"].nunique() < 2:
                continue
            pt = point_all(R, M, S, v)
            bt = boot_all(R, M, S, v, N_BOOT_JK)
            oa_lo, oa_hi = ci(bt["obs_arch"])
            od_lo, od_hi = ci(bt["obs_ds"])
            fa_lo, fa_hi = ci(bt["floors_arch"]["sufficiency suite"])
            fd_lo, fd_hi = ci(bt["floors_ds"]["problem pool"])
            fc_lo, fc_hi = ci(
                bt["floors_ds"]["composite (pool x size x respondents)"])
            rows.append(dict(
                left_out=left_out, value=v, n=pt["n"],
                archs_in_design=R["arch"].nunique(),
                omega2_arch=pt["obs_arch"], omega2_dataset=pt["obs_ds"],
                arch_lo=oa_lo, arch_hi=oa_hi,
                arch_floor_hi=fa_hi, arch_clears=bool(oa_lo > fa_hi),
                ds_lo=od_lo, ds_hi=od_hi,
                ds_floor_hi=fd_hi, ds_clears=bool(od_lo > fd_hi),
                ds_floor_composite_hi=fc_hi,
                ds_clears_composite=bool(od_lo > fc_hi)))
        print(f"  left out {left_out}")
    return pd.DataFrame(rows)


def jackknife_summary(j):
    return (j.groupby("left_out", sort=False)
             .agg(n_readouts=("value", "size"),
                  archs_in_design=("archs_in_design", "max"),
                  median_omega2_dataset=("omega2_dataset", "median"),
                  median_omega2_arch=("omega2_arch", "median"),
                  provenance_clears=("ds_clears", "sum"),
                  provenance_clears_composite=("ds_clears_composite", "sum"),
                  architecture_clears=("arch_clears", "sum"))
             .reset_index())


# ---------------------------------------------------------------------------
# 4: symbolic expressions as functions
# ---------------------------------------------------------------------------

_TOK = re.compile(r"\s*(X\d+|[A-Za-z_]+|-?\d+\.?\d*(?:[eE][-+]?\d+)?|[(),])")


def _parse(expr):
    toks = _TOK.findall(expr)
    pos = 0

    def node():
        nonlocal pos
        t = toks[pos]; pos += 1
        if t == "(" or t == ")" or t == ",":
            raise ValueError(f"unexpected {t!r} in {expr}")
        if pos < len(toks) and toks[pos] == "(":
            pos += 1
            args = [node()]
            while toks[pos] == ",":
                pos += 1
                args.append(node())
            assert toks[pos] == ")"; pos += 1
            return (t, args)
        if t.startswith("X"):
            return ("var", int(t[1:]))
        return ("const", float(t))
    tree = node()
    assert pos == len(toks), expr
    return tree


def _eval(tree, X):
    """gplearn's protected primitives, so the function evaluated here is the
    one the regressor actually fitted."""
    kind = tree[0]
    if kind == "var":
        return X[:, tree[1]]
    if kind == "const":
        return np.full(len(X), tree[1])
    name, args = tree
    a = [_eval(t, X) for t in args]
    if name == "add":
        return a[0] + a[1]
    if name == "sub":
        return a[0] - a[1]
    if name == "mul":
        return a[0] * a[1]
    if name == "div":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(a[1]) > 0.001, a[0] / a[1], 1.0)
    if name == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(a[0]) > 0.001, np.log(np.abs(a[0])), 0.0)
    if name == "sqrt":
        return np.sqrt(np.abs(a[0]))
    if name in ("neg",):
        return -a[0]
    raise ValueError(f"unknown primitive {name}")


def symbolic_fingerprints(sr):
    """Evaluate every distilled CE expression on the probe grid the
    regression was fitted on (x/100 x p), gain and loss concatenated."""
    grid = np.array([(x / 100.0, p) for x in XG for p in PG])
    rows, F = [], []
    clipped = 0
    for rid, g in sr.groupby("run_id"):
        if set(g["domain"]) != {"gain", "loss"}:
            continue
        vec = []
        for dom in ["gain", "loss"]:
            e = g[g["domain"] == dom]["ce_expression"].iloc[0]
            y = _eval(_parse(str(e)), grid)
            y = np.nan_to_num(y, nan=0.0, posinf=3.0, neginf=-3.0)
            clipped += int((np.abs(y) > 3.0).sum())
            vec.append(np.clip(y, -3.0, 3.0))
        F.append(np.concatenate(vec))
        r = g.iloc[0]
        rows.append(dict(run_id=rid, dataset=r["dataset"], arch=r["arch"],
                         seed=r["seed"]))
    return pd.DataFrame(rows), np.stack(F), clipped


def syntactic_fingerprints(sr):
    ops = ["add", "sub", "mul", "div", "log", "sqrt"]
    rows, F = [], []
    for rid, g in sr.groupby("run_id"):
        if set(g["domain"]) != {"gain", "loss"}:
            continue
        vec = []
        for dom in ["gain", "loss"]:
            gg = g[g["domain"] == dom].iloc[0]
            e = str(gg["ce_expression"])
            vec += [e.count(o + "(") for o in ops] + [gg["ce_complexity"]]
        F.append(vec)
        r = g.iloc[0]
        rows.append(dict(run_id=rid, dataset=r["dataset"], arch=r["arch"]))
    return pd.DataFrame(rows), np.asarray(F, dtype=float)


def distance_r2(F, labels):
    """Share of total squared distance that lies between label groups --
    the distance-based R^2 PERMANOVA's pseudo-F is built from."""
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    n = len(labels)
    sst = D2[np.triu_indices(n, 1)].sum() / n
    ssw = 0.0
    labels = np.asarray(labels)
    for g in pd.unique(labels):
        ix = np.where(labels == g)[0]
        if len(ix) > 1:
            sub = D2[np.ix_(ix, ix)]
            ssw += sub[np.triu_indices(len(ix), 1)].sum() / len(ix)
    return float((sst - ssw) / sst) if sst > 0 else np.nan


def symbolic_stats(info, F, view, extra=None):
    """`extra`, if a dict, receives the permutation nulls and the distance
    matrix of this view for plotting."""
    ds, ar = info["dataset"].to_numpy(), info["arch"].to_numpy()
    pd_ = permanova(F, ds, strata=ar, seed=SEED, return_null=True)
    pa_ = permanova(F, ar, strata=ds, seed=SEED, return_null=True)
    if extra is not None:
        tag = view.split(" ")[0]
        extra[f"{tag}|null_F_dataset"] = pd_["null"]
        extra[f"{tag}|null_F_arch"] = pa_["null"]
        extra[f"{tag}|D2"] = pd_["D2"]
        extra[f"{tag}|dataset"] = ds.astype(str)
        extra[f"{tag}|arch"] = ar.astype(str)
        extra[f"{tag}|run_id"] = info["run_id"].to_numpy().astype(str)
    return dict(view=view, n_runs=len(info), dim=F.shape[1],
                permanova_F_dataset=pd_["pseudo_F"],
                permanova_p_dataset=pd_["p_value"],
                permanova_F_arch=pa_["pseudo_F"],
                permanova_p_arch=pa_["p_value"],
                silhouette_dataset=silhouette_by(F, ds),
                silhouette_arch=silhouette_by(F, ar),
                r2_dataset=distance_r2(F, ds), r2_arch=distance_r2(F, ar))


def pca2(F):
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    U, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = U[:, :2] * Sv[:2]
    ev = (Sv ** 2) / (Sv ** 2).sum()
    return pcs, ev[:2]


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-jackknife", action="store_true",
                    help="reuse the existing jackknife tables (the slow "
                         "stage); for iterating on the other outputs")
    args = ap.parse_args()

    main_all, sim, suf = load()
    lines = ["# Robustness of the floor-clearing result\n"]

    # ---- 1 + 2 ----------------------------------------------------------
    print("floor definitions + family-wise correction:")
    sets = design_sets(main_all, sim, suf)
    samples = {}
    fd = floor_definition_table(sets, samples=samples)
    fd.to_csv(os.path.join(TABDIR, "robustness_floor_definitions.csv"),
              index=False)
    np.savez_compressed(os.path.join(TABDIR, "robustness_boot_samples.npz"),
                        **samples)
    fs = floor_summary(fd)
    fs.to_csv(os.path.join(TABDIR, "robustness_floor_summary.csv"),
              index=False)
    lines.append("## 1. Which floor, and does the result depend on it?\n")
    lines.append(
        "The provenance floor used in the main analysis is the problem-pool "
        "share of the controlled factorial. The same factorial also measures "
        "the arm-size and respondent shares; 'largest component' takes the "
        "max of the three per bootstrap replicate, 'composite' enters all "
        "three as one 8-level factor so their interactions count too. "
        "Architecture is given its usual floor (sufficiency suite) and a "
        "second one from the controlled suite.\n")
    lines.append(
        "Clears: point = observed > floor; interval = observed 95% interval "
        "strictly above the floor interval (the rule used in the paper); "
        "Holm = Holm-adjusted bootstrap p < 0.05 across the 11 read-outs for "
        "H0: observed <= floor.\n")
    lines.append(fs.round(3).to_string(index=False) + "\n")

    # ---- 3 --------------------------------------------------------------
    jk_path = os.path.join(TABDIR, "robustness_jackknife.csv")
    if args.skip_jackknife and os.path.exists(jk_path):
        print("leave-one-architecture-out: reusing", jk_path)
        jk = pd.read_csv(jk_path)
    else:
        print("leave-one-architecture-out:")
        jk = jackknife_table(main_all, sim, suf)
        jk.to_csv(jk_path, index=False)
    js = jackknife_summary(jk)
    js.to_csv(os.path.join(TABDIR, "robustness_jackknife_summary.csv"),
              index=False)
    lines.append("## 2. Leave one architecture out\n")
    lines.append(
        f"Interval test with {N_BOOT_JK} resamples; provenance floor = "
        "problem pool (as in the paper) and, separately, the composite floor; "
        "architecture floor = sufficiency suite. 'none' is the full design.\n")
    lines.append(js.round(3).to_string(index=False) + "\n")

    # ---- 4 --------------------------------------------------------------
    print("symbolic expressions as functions:")
    srp = os.path.join(TABDIR, "sr_expressions.csv")
    if os.path.exists(srp):
        sr = pd.read_csv(srp)
        sr = sr[sr["dataset"].isin(MAIN) & np.isfinite(sr["ce_r2"])]
        info, F, clipped = symbolic_fingerprints(sr)
        info2, F2 = syntactic_fingerprints(sr)
        extra = {}
        st = pd.DataFrame([symbolic_stats(info, F, "functional (CE grid)",
                                          extra=extra),
                           symbolic_stats(info2, F2,
                                          "syntactic (operator counts)",
                                          extra=extra)])
        st.to_csv(os.path.join(TABDIR, "symbolic_clustering.csv"),
                  index=False)
        np.savez_compressed(os.path.join(TABDIR, "symbolic_permanova.npz"),
                            **extra)
        pcs, ev = pca2(F)
        out = info.copy()
        out["pc1"], out["pc2"] = pcs[:, 0], pcs[:, 1]
        out.attrs = {}
        out.to_csv(os.path.join(TABDIR, "symbolic_fingerprint_pca.csv"),
                   index=False)
        with open(os.path.join(TABDIR, "symbolic_fingerprint_pca.json"),
                  "w") as f:
            import json
            json.dump(dict(explained=[float(e) for e in ev],
                           n_clipped_values=int(clipped),
                           n_values=int(F.size)), f)
        lines.append("## 3. H5: do the distilled expressions sort by "
                     "provenance or by architecture?\n")
        lines.append(
            "Each run's gain and loss expressions are evaluated on the "
            f"{len(XG)} x {len(PG)} (x/100, p) grid the regression was fitted "
            "on, giving a {}-dimensional function fingerprint per run; "
            "PERMANOVA permutes within strata of the other factor "
            "(2,000 permutations). The syntactic view uses operator counts "
            "and program length instead, which is what Fig. S2c shows.\n"
            .format(F.shape[1]))
        lines.append(st.round(3).to_string(index=False) + "\n")
        lines.append(f"({clipped} of {F.size} evaluated values clipped to "
                     "[-3, 3])\n")
        print(st.round(3).to_string(index=False))

    with open(os.path.join(RESULTS, "robustness.md"), "w") as f:
        f.write("\n".join(lines))
    print("\nwrote results/robustness.md")
