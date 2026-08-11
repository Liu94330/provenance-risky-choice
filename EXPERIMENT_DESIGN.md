# Experiment design

**Working title.** Dataset provenance, not architecture, determines what
neural networks learn about human risky choice.

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
probe behaviour, and (iv) a closed-form expression distilled from its
certainty-equivalent surface by symbolic regression. The headline test is a
two-factor variance decomposition of these readouts.

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

**A — architecture:** LINEAR, MLP_S (2x64), MLP_L (4x256), SET (DeepSets,
permutation-invariant), CPT_NET (differentiable rank- and sign-dependent CPT,
Prelec-I weighting), GBT (gradient-boosted trees). **S — seeds:** 10.
Main grid: 4 datasets x 6 architectures x 10 seeds = 240 runs; control suites
bring the total to ~480.

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

**H1.** Partial omega^2 for the dataset factor exceeds the architecture factor
across theory coordinates and indices (balanced two-way ANOVA, seeds as
replicates); multivariate confirmation by PERMANOVA with restricted
permutations (dataset labels permuted within architecture strata and vice
versa) plus silhouette separation of fingerprints by either labelling.

**H2.** With the data-generating agent held literally identical, the theory
recovered by the same architectures differs systematically between the two
problem distributions (parameter-recovery bias with seed bootstrap CIs) —
sampling provenance alone distorts theory inference, before any human
variability enters. Smoke runs already show near-exact recovery on the c13k
distribution vs. inflated (alpha, beta, lambda, gamma) on the CPC
distribution.

**H3.** On the 1,266 matched problems, every architecture flips the sign of
the rare-event index between description and experience arms — each model
class independently rediscovers the description-experience gap, and the
protocol effect dwarfs architecture differences.

**H4.** Train-on-row, test-on-column transfer matrix over the four main arms,
reported as completeness = (1 - MSE/Var) / (1 - floor/Var) with the noise
floor taken from between-subject variance (bRate_std^2 / n per problem;
for cpc18_exp the mean-of-blocks floor is an approximation since blocks share
subjects). Within-provenance near ceiling, cross-provenance far below.

**H5.** Distilled expressions cluster by training dataset, not architecture.

**Controls.** Size-matched subsampling (all arms to 182 problems);
feature-matched importance reweighting of c13k toward the CPC problem
distribution (logistic-classifier odds weights) to isolate the sampling
component of provenance; binomial-noise robustness of aggregate rates.

## Training protocol

Fixed 80/10/10 split by problem (split seed 0, identical across
architectures); Adam with early stopping on validation MSE (patience 40);
MSE objective on aggregate rates; global payoff scale 100 fixed across
datasets so cross-dataset transfer is well defined; model seeds vary
initialisation and batching only. Undertrained models inflate the
architecture factor, so all headline runs train to early-stopping
convergence.
