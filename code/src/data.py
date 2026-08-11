"""Build every provenance arm as canonical arrays and cache to data/processed.

Arms (all Amb == 0, aggregate B-rates):
  c13k_exp        choices13k, Feedback=True   (~9,831 problems, US MTurk, n~16)
  c13k_desc       choices13k, Feedback=False  (~1,928 problems)
  c13k_pair_exp   the 1,266 problems present in BOTH protocols, experience arm
  c13k_pair_desc  same problems, description arm (perfect within-problem pair)
  cpc18_desc      CPC18 estimation set, block 1 (no feedback), 182 problems,
                  Israeli lab samples, n~97
  cpc18_exp       CPC18 blocks 2-5 (feedback), per-problem mean rate
  cpc18_technion_exp / cpc18_rehovot_exp   population split of cpc18_exp
  syn_c13k / syn_cpc   the SAME ground-truth CPT agent simulated on the c13k
                  problem set vs the CPC18 problem set at matched n
                  (isolates problem-space sampling with known ground truth)

sem2 = squared standard error of the aggregate rate (bRate_std^2 / n) -- used
as the binomial/subject noise floor for completeness normalisation.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .paths import cpc18_raw_path
from .problems import cpt_rate, expand_lottery, problem_arrays

from .paths import EXTERNAL, PROCESSED  # noqa: E402  (single source of truth)

# ground-truth agent for the synthetic sampling-provenance arm
TRUE_THETA = dict(alpha=0.88, beta=0.88, lam=2.25,
                  gamma_pos=0.65, gamma_neg=0.65, tau=6.0)


# ---------------------------------------------------------------------------
# choices13k
# ---------------------------------------------------------------------------

def _load_c13k_raw(c13k_dir):
    df = pd.read_csv(os.path.join(c13k_dir, "c13k_selections.csv"))
    with open(os.path.join(c13k_dir, "c13k_problems.json")) as f:
        gambles = json.load(f)
    assert len(df) == len(gambles), "selections.csv and problems.json disagree"
    return df, gambles


def _c13k_rows_to_problems(df, gambles):
    plist = []
    for ridx, row in df.iterrows():
        g = gambles[str(ridx)]
        A = ([x for p, x in g["A"]], [p for p, x in g["A"]])
        B = ([x for p, x in g["B"]], [p for p, x in g["B"]])
        n = float(row["n"])
        plist.append(dict(
            A=A, B=B, corr=int(row["Corr"]), y=float(row["bRate"]),
            sem2=float(row["bRate_std"]) ** 2 / n, n=n,
            pid=f"c13k_{row['Problem']}_{'fb' if row['Feedback'] else 'nofb'}"))
    return plist


def build_c13k_arms(c13k_dir):
    df, gambles = _load_c13k_raw(c13k_dir)
    df = df[~df["Amb"].astype(bool)].copy()
    exp = df[df["Feedback"].astype(bool)]
    desc = df[~df["Feedback"].astype(bool)]
    arms = {
        "c13k_exp": _c13k_rows_to_problems(exp, gambles),
        "c13k_desc": _c13k_rows_to_problems(desc, gambles),
    }
    both = set(exp["Problem"]) & set(desc["Problem"])
    arms["c13k_pair_exp"] = _c13k_rows_to_problems(
        exp[exp["Problem"].isin(both)], gambles)
    arms["c13k_pair_desc"] = _c13k_rows_to_problems(
        desc[desc["Problem"].isin(both)], gambles)

    # Experience gradient. choices13k assigns each problem to ONE block, so
    # these are between-problem comparisons: block position indexes how much
    # feedback the respondent had accumulated, and problem effects are
    # randomised across blocks rather than held fixed. Each block still
    # carries ~3,000 problems, which is what makes the contrast usable. The
    # CPC18 gradient below is the within-problem version of the same idea.
    for b in sorted(exp["Block"].unique()):
        rows = exp[exp["Block"] == b]
        arms[f"c13k_blk{int(b)}"] = _c13k_rows_to_problems(rows, gambles)
    return arms


# ---------------------------------------------------------------------------
# CPC18 (estimation set with full distributions + rates we derived from the
# raw trial data; derivation reproduces the published B.1-B.5 to <1e-9)
# ---------------------------------------------------------------------------

def _cpc_row_option(row, side):
    xs, ps = [], []
    for k in range(1, 11):
        v = row[f"{side}v{k}"]
        p = row[f"{side}p{k}"]
        if pd.notna(v) and pd.notna(p) and p > 0:
            xs.append(float(v))
            ps.append(float(p))
    return xs, ps


def build_cpc18_arms():
    fd = pd.read_csv(os.path.join(EXTERNAL, "CPC18_EstSet_fullDist.csv"))
    fd = fd[fd["Amb"] == 0].copy()
    rates = pd.read_csv(os.path.join(EXTERNAL, "cpc18_rates_overall.csv"))
    byloc = pd.read_csv(os.path.join(EXTERNAL, "cpc18_rates_bylocation.csv"))

    def one_arm(rate_df, blocks, tag):
        plist = []
        for _, row in fd.iterrows():
            sub = rate_df[(rate_df["GameID"] == row["GameID"]) &
                          (rate_df["block"].isin(blocks))]
            if len(sub) == 0:
                continue
            y = float(sub["bRate"].mean())
            # variance of a mean of block means; blocks share subjects, so
            # this is an approximation (documented in the design)
            sem2 = float((sub["bRate_std"] ** 2 / sub["n"]).mean() / len(sub))
            n = float(sub["n"].mean())
            plist.append(dict(
                A=_cpc_row_option(row, "a"), B=_cpc_row_option(row, "b"),
                corr=int(row["Corr"]), y=y, sem2=sem2, n=n,
                pid=f"cpc18_{row['GameID']}_{tag}"))
        return plist

    arms = {
        "cpc18_desc": one_arm(rates, [1], "desc"),
        "cpc18_exp": one_arm(rates, [2, 3, 4, 5], "exp"),
        "cpc18_technion_exp": one_arm(
            byloc[byloc["Location"] == "Technion"], [2, 3, 4, 5], "tech"),
        "cpc18_rehovot_exp": one_arm(
            byloc[byloc["Location"] == "Rehovot"], [2, 3, 4, 5], "reho"),
    }
    # Experience gradient, within problem: the same 210 problems and the same
    # respondents at five stages of feedback. Block 1 is the first encounter
    # (identical to the description arm) and block 5 is fully experienced, so
    # this is the description-experience gap resolved into the trajectory that
    # produces it rather than treated as a two-level contrast.
    for b in range(1, 6):
        arms[f"cpc18_blk{b}"] = one_arm(rates, [b], f"blk{b}")
    return arms


# ---------------------------------------------------------------------------
# CPC18 competition problems (GameID 211-270) from the full raw release
# ---------------------------------------------------------------------------

# Resolved at call time by src.paths.cpc18_raw_path(); see its docstring for
# the locations searched and how to override with CPC18_RAW.


def build_cpc18_competition_arms(raw_path=None):
    """The 48 problems that have never entered this pipeline.

    GameID 211-270 (Sets 8-9) exist only in the full raw release: they have no
    row in CPC18_EstSet_fullDist.csv, so until the lottery expansion was
    derived they could not be used at all. Their 240 subjects have zero
    overlap with the 686 subjects behind the estimation-set arms, and the
    problem pool is measurably different -- mean choose-B is 0.38 against 0.49,
    and 13 of the 48 put a real multi-outcome lottery on option A, which never
    happens in GameID 1-150.

    Deliberately NOT read from this file: GameID 1-210. The full release
    carries 150 subject-problem observations that the official calibration
    release (record 845873) drops, so recomputing the existing arms from here
    would silently move every published CPC18 number. The existing arms keep
    using the bundled aggregate rates, which reproduce record 845873 to 1e-16.

    Returns {} if the raw file is absent, so the pipeline still runs without it.
    """
    path = raw_path or cpc18_raw_path()
    if not path or not os.path.exists(path):
        return {}

    cols = ["GameID", "SubjID", "block", "B", "Amb", "Corr", "Location",
            "Ha", "pHa", "La", "LotShapeA", "LotNumA",
            "Hb", "pHb", "Lb", "LotShapeB", "LotNumB"]
    d = pd.read_csv(path, usecols=cols, low_memory=False)
    d = d[(d["Amb"] == 0) & (d["GameID"] > 210)]
    if not len(d):
        return {}

    spec = d.drop_duplicates("GameID").set_index("GameID")

    def one_arm(rows, blocks, tag, location=None):
        sub = rows[rows["block"].isin(blocks)]
        if location:
            sub = sub[sub["Location"] == location]
        # two stages, and the order matters: a subject contributes many trials
        # to the same problem, so rates are formed per subject FIRST and the
        # spread is taken across subjects. sem2 is the squared standard error
        # of the aggregate, i.e. across-subject variance / n_subjects.
        per_subj = sub.groupby(["GameID", "block", "SubjID"])["B"].mean()
        per_block = per_subj.groupby(["GameID", "block"]).agg(
            bRate="mean", bRate_std="std", n="size")
        plist = []
        for gid, g in per_block.groupby(level=0):
            r = spec.loc[gid]
            ax, ap = expand_lottery(r["Ha"], r["pHa"], r["La"],
                                    r["LotShapeA"], r["LotNumA"])
            bx, bp = expand_lottery(r["Hb"], r["pHb"], r["Lb"],
                                    r["LotShapeB"], r["LotNumB"])
            # Same multi-block combination as build_cpc18_arms so the noise
            # floors, and therefore completeness, stay comparable across arms.
            # It treats blocks 2-5 as independent although they share
            # subjects; correcting that is an all-arms change (it would move
            # every published completeness) and is left to one.
            sem2 = float((g["bRate_std"] ** 2 / g["n"]).mean() / len(g))
            plist.append(dict(
                A=(ax, ap), B=(bx, bp), corr=int(r["Corr"]),
                y=float(g["bRate"].mean()), sem2=sem2,
                n=float(g["n"].mean()), pid=f"cpc18c_{gid}_{tag}"))
        return plist

    arms = {
        "cpc18_comp_desc": one_arm(d, [1], "desc"),
        "cpc18_comp_exp": one_arm(d, [2, 3, 4, 5], "exp"),
    }
    return {k: v for k, v in arms.items() if v}


# ---------------------------------------------------------------------------
# synthetic ground-truth arms (same agent, different problem distributions)
# ---------------------------------------------------------------------------

def simulate_agent(problem_list, theta=TRUE_THETA, n_subj=16, n_trials=5,
                   seed=1234):
    rng = np.random.default_rng(seed)
    arrs = problem_arrays(problem_list)
    p_true = cpt_rate(arrs, (theta["alpha"], theta["beta"], theta["lam"],
                             theta["gamma_pos"], theta["gamma_neg"],
                             theta["tau"]))
    out = []
    for i, pr in enumerate(problem_list):
        subj_rates = rng.binomial(n_trials, p_true[i], size=n_subj) / n_trials
        y = subj_rates.mean()
        sem2 = subj_rates.var(ddof=1) / n_subj if n_subj > 1 else 0.0
        q = dict(pr)
        q.update(y=float(y), sem2=float(sem2), n=float(n_subj),
                 pid="syn_" + str(pr["pid"]))
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# controlled provenance experiment (2^4 factorial, ground truth known)
# ---------------------------------------------------------------------------

# The description-condition agent overweights rare events (Prelec gamma < 1);
# the experience-condition agent weights probabilities close to linearly, which
# is the standard summary of what feedback does to choice. Everything else is
# held identical, so any difference the networks show between the two is caused
# by that one property of the generating agent and nothing else.
AGENT_DESC = dict(TRUE_THETA)
AGENT_EXP = dict(TRUE_THETA, gamma_pos=1.0, gamma_neg=1.0)

CONTROLLED_FACTORS = dict(
    problems=("c13k", "cpc"),     # which problem distribution to draw from
    agent=("desc", "exp"),        # generating agent's weighting function
    n_subj=(16, 100),             # respondents per problem -> aggregation noise
    n_problems=(182, 1500),       # size of the training arm
)


def build_controlled_arms(pools, seed=20260811):
    """A 2^4 factorial over the things 'provenance' actually bundles together.

    Comparing two real datasets confounds everything at once: choices13k and
    CPC18 differ in problem pool, respondents per problem, number of problems,
    and protocol simultaneously, so no observational contrast can say which of
    those drives the theory a network ends up encoding. Here each factor is
    manipulated independently against a KNOWN generating agent, which turns the
    question from correlational into causal and -- because synthetic arms are
    unlimited -- is not capped by how many public datasets happen to exist.

    `pools` maps 'c13k'/'cpc' to a problem list to draw from. Returns 16 arms
    named sim_{problems}_{agent}_n{n_subj}_p{n_problems}.
    """
    rng = np.random.default_rng(seed)
    arms = {}
    for pi, pool_key in enumerate(CONTROLLED_FACTORS["problems"]):
        pool = pools[pool_key]
        for ai, agent_key in enumerate(CONTROLLED_FACTORS["agent"]):
            theta = AGENT_DESC if agent_key == "desc" else AGENT_EXP
            for ni, n_subj in enumerate(CONTROLLED_FACTORS["n_subj"]):
                for si, n_problems in enumerate(
                        CONTROLLED_FACTORS["n_problems"]):
                    # sample with replacement when the pool is smaller than the
                    # requested size, so problem count varies independently of
                    # which pool it came from
                    k = min(n_problems, len(pool)) if n_problems <= len(pool) \
                        else n_problems
                    idx = rng.choice(len(pool), size=k,
                                     replace=k > len(pool))
                    subset = [pool[i] for i in idx]
                    # distinct seed per cell so the noise draws are independent
                    cell_seed = int(seed + 1000 * pi + 100 * ai + 10 * ni + si)
                    plist = simulate_agent(subset, theta=theta,
                                           n_subj=n_subj, seed=cell_seed)
                    name = (f"sim_{pool_key}_{agent_key}"
                            f"_n{n_subj}_p{n_problems}")
                    # simulate_agent reuses the source pid, which collides once
                    # a pool is sampled with replacement across cells
                    for j, q in enumerate(plist):
                        q["pid"] = f"{name}_{j}"
                    arms[name] = plist
    return arms


# ---------------------------------------------------------------------------
# design-sufficiency grid: how much data does this method need to work?
# ---------------------------------------------------------------------------

# Problem counts and respondents-per-problem to sweep. The real arms sit inside
# this range -- CPC18 has 182 problems at ~100 respondents, choices13k has
# 9,831 at ~16 -- so the answer says directly which of them could ever have
# supported a trustworthy readout.
SUFFICIENCY_PROBLEMS = (60, 120, 250, 500, 1000)
SUFFICIENCY_SUBJECTS = (8, 16, 32, 128)


def build_sufficiency_arms(pool, seed=20260812):
    """Vary only arm size and respondents per problem, against a fixed known
    agent, so the recovery error becomes a function of the design.

    Every other analysis here asks what a network learned. This one asks a
    prior question the method needs answered before any of that is
    interpretable: with a KNOWN generating agent and no confounds, how many
    problems and how many respondents per problem does it take before the
    fitted CPT coordinates track the truth? Without this, "CPC18 models encode
    a different theory" cannot be separated from "182 problems is too few to
    tell", and the same doubt applies to anyone reusing the method.
    """
    rng = np.random.default_rng(seed)
    arms = {}
    for pi, n_problems in enumerate(SUFFICIENCY_PROBLEMS):
        for si, n_subj in enumerate(SUFFICIENCY_SUBJECTS):
            idx = rng.choice(len(pool), size=n_problems,
                             replace=n_problems > len(pool))
            subset = [pool[i] for i in idx]
            plist = simulate_agent(subset, theta=AGENT_DESC, n_subj=n_subj,
                                  seed=int(seed + 100 * pi + si))
            name = f"suf_p{n_problems}_n{n_subj}"
            for j, q in enumerate(plist):
                q["pid"] = f"{name}_{j}"
            arms[name] = plist
    return arms


# ---------------------------------------------------------------------------
# orchestration / caching
# ---------------------------------------------------------------------------

def prepare_all(c13k_dir, out_dir=PROCESSED, seed=1234):
    os.makedirs(out_dir, exist_ok=True)
    arms = {}
    arms.update(build_c13k_arms(c13k_dir))
    arms.update(build_cpc18_arms())
    arms.update(build_cpc18_competition_arms())
    arms["syn_c13k"] = simulate_agent(arms["c13k_exp"], seed=seed)
    arms["syn_cpc"] = simulate_agent(
        arms["cpc18_desc"] + arms["cpc18_exp"], seed=seed + 1)
    arms.update(build_controlled_arms(
        {"c13k": arms["c13k_exp"],
         "cpc": arms["cpc18_desc"] + arms["cpc18_exp"]}))
    arms.update(build_sufficiency_arms(arms["c13k_exp"]))
    meta = {}
    for name, plist in arms.items():
        arrs = problem_arrays(plist)
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **arrs)
        meta[name] = dict(
            n_problems=len(plist),
            mean_rate=float(np.nanmean(arrs["y"])),
            noise_floor_mse=float(np.nanmean(arrs["sem2"])),
            var_y=float(np.nanvar(arrs["y"])),
        )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_arm(name, out_dir=PROCESSED):
    z = np.load(os.path.join(out_dir, f"{name}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def split_indices(n, seed=0, frac=(0.8, 0.1, 0.1)):
    """Fixed train/val/test split by problem (same for every architecture)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def take(arrs, idx):
    return {k: (v[idx] if getattr(v, "ndim", 0) >= 1 and len(v) == len(arrs["y"])
                else v) for k, v in arrs.items()}


def subsample(arrs, n_keep, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arrs["y"]), size=n_keep, replace=False)
    return take(arrs, np.sort(idx))


def importance_weights(source_arrs, target_arrs, clip=(0.1, 10.0), seed=0):
    """Weights that reweight the source problem distribution toward the
    target's (feature-matched control). Logistic classifier on flat problem
    features; w = p_target/(1-p_target) odds, clipped and normalised."""
    from sklearn.linear_model import LogisticRegression
    from .problems import flat_features
    Xs = flat_features(source_arrs)
    Xt = flat_features(target_arrs)
    X = np.concatenate([Xs, Xt])
    lab = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X, lab)
    p = clf.predict_proba(Xs)[:, 1]
    w = np.clip(p / np.maximum(1 - p, 1e-6), *clip)
    return (w / w.mean()).astype(np.float32)
