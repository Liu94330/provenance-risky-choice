"""Symbolic-regression distillation.

For each trained model we extract its certainty-equivalent surface CE(x, p)
(a model-free behavioural summary computed from the probe battery, no CPT
assumed) and distill it into a closed-form expression with symbolic
regression. If the paper's claim is right, the recovered EXPRESSION FAMILIES
cluster by training dataset, not by architecture.

Backends: gplearn (default, pure python, runs anywhere) or PySR (optional,
higher quality; pip install pysr, first run downloads Julia).
"""

from __future__ import annotations

import numpy as np


def distill_ce(ce_df, backend="gplearn", seed=0, generations=30,
               population=800):
    """ce_df: DataFrame with columns x, p, ce (from probes.ce_surface).
    Fits CE = f(x, p). Returns dict(expression, r2, complexity, backend)."""
    X = np.array(ce_df[["x", "p"]].to_numpy(dtype=float))
    X[:, 0] = np.abs(X[:, 0]) / 100.0  # scale as in the models
    y = np.abs(ce_df["ce"].to_numpy(dtype=float)) / 100.0

    if backend == "pysr":
        try:
            from pysr import PySRRegressor
        except ImportError as e:
            raise RuntimeError("pip install pysr for the pysr backend") from e
        model = PySRRegressor(
            niterations=60, binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp", "log"], maxsize=25,
            model_selection="best", random_state=seed, verbosity=0,
            temp_equation_file=True)
        model.fit(X, y, variable_names=["x", "p"])
        best = model.get_best()
        return dict(expression=str(best["equation"]),
                    r2=float(1 - best["loss"] / max(np.var(y), 1e-12)),
                    complexity=int(best["complexity"]), backend="pysr")

    from gplearn.genetic import SymbolicRegressor
    est = SymbolicRegressor(
        population_size=population, generations=generations,
        function_set=("add", "sub", "mul", "div", "log", "sqrt"),
        parsimony_coefficient=5e-4, metric="mse", random_state=seed,
        const_range=(-3.0, 3.0), verbose=0)
    est.fit(X, y)
    pred = est.predict(X)
    r2 = float(1 - np.mean((pred - y) ** 2) / max(np.var(y), 1e-12))
    return dict(expression=str(est._program), r2=r2,
                complexity=int(est._program.length_), backend="gplearn")


def distill_weighting_curve(ce_df, x_ref=80, backend="gplearn", seed=0):
    """1-D distillation of the implied weighting curve w(p) ~ CE(x_ref, p)/x_ref."""
    sub = ce_df[np.isclose(np.abs(ce_df["x"]), x_ref)].sort_values("p")
    if len(sub) < 5:
        return None
    Xp = sub[["p"]].to_numpy(dtype=float)
    y = sub["ce_ratio"].to_numpy(dtype=float) * sub["p"].to_numpy()
    # ce_ratio = CE/(p*x); implied w-proxy = CE/x = ce_ratio * p
    if backend == "pysr":
        from pysr import PySRRegressor
        model = PySRRegressor(niterations=40,
                              binary_operators=["+", "-", "*", "/", "^"],
                              unary_operators=["exp", "log"], maxsize=20,
                              random_state=seed, verbosity=0,
                              temp_equation_file=True)
        model.fit(Xp, y, variable_names=["p"])
        best = model.get_best()
        return dict(expression=str(best["equation"]),
                    complexity=int(best["complexity"]), backend="pysr")
    from gplearn.genetic import SymbolicRegressor
    est = SymbolicRegressor(population_size=600, generations=25,
                            function_set=("add", "sub", "mul", "div", "log",
                                          "sqrt"),
                            parsimony_coefficient=5e-4, random_state=seed,
                            verbose=0)
    est.fit(Xp, y)
    pred = est.predict(Xp)
    r2 = float(1 - np.mean((pred - y) ** 2) / max(np.var(y), 1e-12))
    return dict(expression=str(est._program), r2=r2,
                complexity=int(est._program.length_), backend="gplearn")
