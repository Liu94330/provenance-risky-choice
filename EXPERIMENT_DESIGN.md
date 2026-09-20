# Experiment design

**Title.** Training-Data Provenance Governs the Cognitive Theories Encoded
by Neural Models of Risky Choice.

**Question.** When neural networks trained to predict human risky choices are
interrogated as "discovered theories" (Peterson et al., Science 2021; cf. the
debate around foundation models of cognition, Binz et al., Nature 2025), how
much of what they appear to learn about human decision-making is fixed by the
provenance of the training data — problem sampling, response protocol,
participant population, aggregation — rather than by the model class?

**Approach.** Treat every trained model as a participant. All models, from
every (dataset x architecture x seed) cell, answer the same fixed probe
battery; each model is thereby reduced to (i) a behavioural fingerprint,
(ii) interpretable effect indices, (iii) CPT theory coordinates fitted to its
probe behaviour, and (iv) an exact grouped-Shapley attribution over eleven
semantic gamble-feature groups (`shapcode/`), the one readout that assumes no
functional form. A closed form distilled from the certainty-equivalent surface
by symbolic regression is reported as supplementary (Figure S2).

The headline test compares each readout's two-factor variance decomposition
against an artifact floor measured on fixed-agent arms, where the true
generating theory is held literally constant. The uncorrected decomposition
alone favours architecture, and is reported as such.

A design advantage established during data audit: choices13k problems were
generated with the CPC competition problem-space algorithm, and the payoff
support coincides exactly across datasets ([-50, 256]). Dataset differences
therefore cannot be attributed to disjoint problem spaces; what remains is
precisely the provenance decomposition we test.

## Factors

**D — data provenance** (all arms exclude ambiguous problems, target =
aggregate P(choose B)):

| arm | problems | provenance |
|---|---|---|
| c13k_exp | 9,831 | US MTurk, feedback (experience), n~16/problem |
| c13k_desc | 1,928 | US MTurk, no feedback (description) |
| c13k_pair_exp / _desc | 1,266 | the same problems under BOTH protocols |
| cpc18_desc | 182 | Israeli labs, CPC18 block 1 (description), n~97 |
| cpc18_exp | 182 | CPC18 blocks 2-5 (experience) |
| cpc18_technion_exp / cpc18_rehovot_exp | 182 | two lab populations, same problems and protocol |
| syn_c13k / syn_cpc | 9,831 / 364 | one ground-truth CPT agent (alpha=beta=.88, lambda=2.25, Prelec gamma=.65, tau=6) simulated on each problem distribution at n=16x5 |
| cpc18_comp_desc / cpc18_comp_exp | 48 | CPC18 competition set, evaluation only, never trained on |
| cpc18_blk1..blk5 / c13k_blk2..blk5 | 182 / ~2,450 | experience gradient, blocks as an ordered factor |
| sim_{c13k,cpc}_{desc,exp}_n{16,100}_p{182,1500} | 182 / 1,500 | 2^4 controlled factorial against a known agent (16 arms) |
| suf_p{60,120,250,500,1000}_n{8,16,32,128} | 60-1,000 | design-sufficiency sweep against a known agent (20 arms) |

**A — architecture:** LINEAR (56 params), MLP_S (2x64, 7,809),
MLP_L (4x256, 211,969), RESMLP (wide residual MLP, 2,134,529), SET (DeepSets,
permutation-invariant, 16,897), TRANSFORMER (self-attention across outcome
slots, 75,649), CPT_NET (differentiable rank- and sign-dependent CPT, Prelec-I
weighting, 6 params), GBT (gradient-boosted trees, non-neural baseline).
Learning rates are calibrated per architecture under the final 600-epoch budget
and pinned in `code/src/models.ARCH_LR`.

**S — seeds:** 10 in the main and synthetic suites; 5 in paired, population,
size_control, reweight_control, controlled and experience_gradient; 3 in
sufficiency.

Main grid: 4 provenance arms x 8 architectures x 10 seeds = 320 runs. Nine
suites over 57 arms bring the total to 2,240 training units (main 320,
controlled 640, sufficiency 480, experience_gradient 360, synthetic 160,
paired 80, population 80, size_control 80, reweight_control 40; one row each in
`results/runs_*.csv`).

## Shared probe battery (1,742 two-outcome problems)

Certainty-equivalent grids for gains and losses (x in {5..200}, p in
{.01...99}, sure amount = r * p * x, r in {.1..1.5}; 840 each), a mixed-gamble
loss-aversion ladder (32), and common-ratio (Allais) ladders in both domains
(30). Option A is always the sure option, so model output = P(choose risky).

**Readouts.** (1) fingerprint = the probe prediction vector; (2) indices:
rare-event index (sign encodes the description-experience gap direction),
certainty (common-ratio) index, reflection index, loss-aversion kappa,
EV-consistency; (3) CPT coordinates (alpha, beta, lambda, gamma+, gamma-,
tau) fitted to probe predictions by multi-start L-BFGS-B — power value
function, Prelec-I weighting, rank-dependent, identical closed form as the
numpy/torch twin implementations (unit-tested to 1e-6 agreement); (4)
symbolic regression (gplearn default, PySR optional) on the model's CE(x, p)
surface, extracted model-free by interpolating the P = .5 crossing.

## Hypotheses and analyses

**H1.** Against an artifact floor measured on fixed-agent arms, the provenance
effect clears its own floor on every readout and the architecture effect clears
its floor on almost none (11/11 vs 2/11 by point estimate; 10/11 vs 0/11 by
bootstrap 95% interval). The uncorrected partial omega^2 points the other way
and is reported openly: median dataset 0.360 vs architecture 0.487 across the
11 readouts, with dataset larger on 5/11; multivariate fingerprints agree with
the uncorrected view (PERMANOVA architecture F=27.2 vs dataset F=17.8, both
p=0.0005; silhouette 0.136 by architecture vs 0.058 by dataset). The correction
is the claim, not the raw decomposition.

**H2.** With the data-generating agent held literally identical, the theory
recovered by the same architectures differs systematically between the two
problem distributions (parameter-recovery bias with seed bootstrap CIs) —
sampling provenance alone distorts theory inference, before any human
variability enters. Recovery is biased on both distributions and the bias
differs between them: from the c13k pool the agent is recovered at alpha .80 /
beta 1.11 / lam 3.00 / tau 4.49, from the CPC pool at alpha .79 / beta 1.17 /
lam 2.44 / tau 3.51, against true values .88 / .88 / 2.25 / 6.0. The largest
distribution-induced gaps are on lam (+0.56) and tau (+0.98).

**H3.** On the 1,266 matched problems the rare-event index falls from
description to experience in every architecture, and crosses zero in 5 of 8
(CPT_NET, GBT, MLP_L, MLP_S, RESMLP). LINEAR, SET and TRANSFORMER stay positive
under experience; MLP_L is borderline. The protocol effect is consistent in
direction across model classes even where the sign does not change.

**H4.** Train-on-row, test-on-column transfer matrix: the four main arms as
training sources, evaluated on those four plus the two held-out CPC18
competition arms (48 problems never trained on), reported as completeness = (1 - MSE/Var) / (1 - floor/Var) with the noise
floor taken from between-subject variance (bRate_std^2 / n per problem;
for cpc18_exp the mean-of-blocks floor is an approximation since blocks share
subjects). Transfer is strongly asymmetric by training source rather than
split cleanly within vs across provenance: median completeness 0.837 within and
0.665 across for choices13k-trained models, 0.404 within and 0.286 across for
CPC18-trained models, collapsing to 0.468 and -0.027 on the 48 unseen
competition problems. The asymmetry is confounded with CPC18 having only 182
problems and must be read together with the `size_control` arms.

**H5 (descriptive, supplementary).** Symbolic distillation of the
certainty-equivalent surface is reported as a recovery check rather than as a
test: Figure S2 shows CE-surface r^2 and expression complexity by arm (median
r^2 ~0.99 everywhere; the arms differ in the lower tail). No clustering
statistic over the distilled expressions is computed.

**Controls.** Size-matched subsampling (all arms to 182 problems);
feature-matched importance reweighting of c13k toward the CPC problem
distribution (logistic-classifier odds weights) to isolate the sampling
component of provenance; binomial-noise robustness of aggregate rates.
A 2^4 complete factorial against a known generating agent (problem pool x agent
weighting x respondents per problem x arm size, architecture entered as a fifth
factor; 640 units) separates the components of provenance. A design-sufficiency
sweep (arm size 60-1000 x 8-128 respondents; 480 units) bounds what any CPT
readout can recover. The same fixed-agent arms supply the artifact floor for
both factors. The experience gradient replicates the protocol effect across two
independent designs (CPC18 blocks 1-5 and choices13k blocks 2-5).

## Training protocol

Fixed 80/10/10 split by problem (split seed 0, identical across
architectures); Adam with early stopping on validation MSE (patience 40);
MSE objective on aggregate rates; global payoff scale 100 fixed across
datasets so cross-dataset transfer is well defined; model seeds vary
initialisation and batching only. Undertrained models inflate the
architecture factor, so all headline runs train to early-stopping
convergence. Learning rates are calibrated per architecture under the same
600-epoch budget (`09_lr_calibration.py` -> `code/src/models.ARCH_LR`); a single
shared rate leaves SET and MLP_L far from optimum and collapses RESMLP to a
constant, which would be scored as an architecture effect.

Every readout is screened for identifiability before entering an analysis:
`identifiable` (probe SD >= 0.02) for the behavioural indices, keeping 309/320
main-grid runs, and `theory_identifiable` (the above plus no CPT parameter
pinned to a box bound) for the CPT coordinates, keeping 276/320.
