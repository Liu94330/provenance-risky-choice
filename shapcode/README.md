# shapcode — attribution readout

Fourth readout of what a trained network encodes, alongside the CPT
coordinates, behavioural indices and symbolic distillation in `code/`. It is
the only one that assumes **no functional form**: it asks how much a prediction
moves when part of the input is withheld, and nothing else. The other three can
inherit a mis-specified theory; this one cannot.

Kept beside `code/` rather than inside it so the main pipeline never imports
it, and so the Shapley implementation ships with its own correctness test.

```
shapcode/
  shap_readout.py      exact grouped Shapley over feature groups
  shap_figures.py      the two figures (styling imported from code/src/figures)
  run_decomposition.py 160 models (100 main-arm + 60 fixed-agent floor)
                       -> attributions, variance decomposition, artifact
                       floor, cross-check against the CPT readout
  run_figures.py       per-probe values -> beeswarm / dependence / conditional
  test_shapley.py      the Shapley axioms, verified to machine precision
  _boot.py             puts code/ and shapcode/ on the path
```

## Why not `shap.KernelExplainer`

Two reasons, both about being able to compare architectures honestly.

**Grouped, not per-feature.** The model input is 55 padded slots; `A_x[7]`
means nothing on its own, so per-feature attributions are unreadable. Features
are partitioned into eleven semantic groups (`EV difference`, `P(loss) of B`,
`extremes of B`, …) defined in `FEATURE_GROUPS`.

**Exact, not sampled.** With eleven groups all 2¹¹ coalitions can be
enumerated, so the values are the Shapley values rather than an estimate of
them — under a second per model. Nothing then rests on how well a sampler
converged, which matters because the conclusion is a comparison *between*
architectures and a sampling-noise term would sit right on top of it.

`test_shapley.py` checks efficiency, symmetry, the null player and additivity,
against a linear model whose values are known in closed form (machine
precision, ~1e-15) and against a trained network (~6e-8, the float32 limit of
the torch path). Since the values are computed here rather than by a library,
nothing else would catch an error in the coalition weighting.

## Per-probe values are kept

`grouped_shapley(..., per_sample=True)` returns the full `[n_probes]` vector
per group at no extra cost, because the coalition values are already computed
per row. This matters: for `EV difference` the per-probe values swing far wider
than their own average — the mean over probes is near zero (+0.001 on
`c13k_desc`) while the per-probe SD is 0.145, about **1.7× the mean absolute
value** — so averaging first reports "unused" for a group the model leans on in
one region of probability space and against in another.

## Usage

```bash
python3 shapcode/test_shapley.py        # axioms, run this first
python3 shapcode/run_decomposition.py   # ~15 min; 100 main + 60 floor models
python3 shapcode/run_figures.py         # ~1 min, 4 models
```

All three are in `run_all.sh`. Outputs land in `results/tables/` and
`results/figures/`; `code/scripts/10_main_figures.py` files the figures into
`paper_results/figures/supp/` as S7 and S8.

## Scope and result, stated plainly

**Covers 5 of 8 architectures.** LINEAR, MLP_S, MLP_L, RESMLP and GBT consume
the flat 55-d encoding. SET, TRANSFORMER and CPT_NET read per-outcome slots, so
these feature groups do not exist for them.

**Weaker than the other three readouts.** Attribution has a much higher
artifact floor — on fixed-agent arms, where every run shares the same true
generating process, architecture alone still produces ω² of 0.12–0.93. Data
provenance clears its own floor on 5/11 groups, architecture on 3/11. Compare
the CPT coordinates and behavioural indices, where provenance clears on 10/11
and architecture on 0/11.

**One clean result.** Attribution to `EV difference`: provenance ω² = 0.881
against an artifact floor of 0.019. CPC18-trained networks weight the
expected-value difference (+0.06); choices13k-trained networks do not (−0.02).
This agrees with the CPT readout, where CPC18 also shows the higher τ, by a
completely unrelated route.

Treat it as supporting evidence, not as a main result. The floor is too high
for anything else here to carry weight.
