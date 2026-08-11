# Paper tables -- provenance vs architecture

## Table 1. Provenance arms.

_Aggregate choose-B rate per problem after dropping ambiguous items. The noise floor is the expected MSE of a perfect model given finite respondents per problem; ceiling r2 is the most any model can explain._

| arm                      |   problems |   mean_rate |   variance |   noise_floor_mse |   ceiling_r2 | description                            |
|:-------------------------|-----------:|------------:|-----------:|------------------:|-------------:|:---------------------------------------|
| c13k_exp                 |       9831 |      0.5098 |     0.0496 |            0.0068 |       0.8631 | MTurk, feedback-based, n~16/problem    |
| c13k_desc                |       1928 |      0.5054 |     0.0431 |            0.0075 |       0.8249 | MTurk, described, n~16/problem         |
| c13k_pair_exp            |       1266 |      0.5213 |     0.0516 |            0.0072 |       0.8611 | matched subset, experience protocol    |
| c13k_pair_desc           |       1266 |      0.505  |     0.0429 |            0.0075 |       0.8262 | matched subset, description protocol   |
| c13k_blk2                |       2420 |      0.5053 |     0.0509 |            0.0068 |       0.8669 |                                        |
| c13k_blk3                |       2461 |      0.5087 |     0.0493 |            0.0068 |       0.8616 |                                        |
| c13k_blk4                |       2447 |      0.511  |     0.0498 |            0.0068 |       0.8637 |                                        |
| c13k_blk5                |       2503 |      0.514  |     0.0484 |            0.0068 |       0.8598 |                                        |
| cpc18_desc               |        182 |      0.4894 |     0.0599 |            0.0014 |       0.9766 | two Israeli labs, described, n~97-125  |
| cpc18_exp                |        182 |      0.5037 |     0.0714 |            0.0003 |       0.996  | two Israeli labs, experience, n~97-125 |
| cpc18_technion_exp       |        182 |      0.504  |     0.0795 |            0.0006 |       0.9927 | Technion sample only                   |
| cpc18_rehovot_exp        |        182 |      0.5035 |     0.0656 |            0.0006 |       0.9914 | Rehovot sample only                    |
| cpc18_blk1               |        182 |      0.4894 |     0.0599 |            0.0014 |       0.9766 |                                        |
| cpc18_blk2               |        182 |      0.5076 |     0.0697 |            0.0012 |       0.9833 |                                        |
| cpc18_blk3               |        182 |      0.5086 |     0.0707 |            0.0011 |       0.984  |                                        |
| cpc18_blk4               |        182 |      0.5011 |     0.0728 |            0.0011 |       0.9843 |                                        |
| cpc18_blk5               |        182 |      0.4974 |     0.0744 |            0.0012 |       0.9844 |                                        |
| cpc18_comp_desc          |         48 |      0.3785 |     0.0362 |            0.0011 |       0.9689 |                                        |
| cpc18_comp_exp           |         48 |      0.3766 |     0.0479 |            0.0002 |       0.9952 |                                        |
| syn_c13k                 |       9831 |      0.467  |     0.0302 |            0.0028 |       0.9083 | simulated CPT agent on c13k problems   |
| syn_cpc                  |        364 |      0.4848 |     0.0297 |            0.0027 |       0.9076 | simulated CPT agent on CPC18 problems  |
| sim_c13k_desc_n16_p182   |        182 |      0.4755 |     0.0241 |            0.003  |       0.8776 |                                        |
| sim_c13k_desc_n16_p1500  |       1500 |      0.4688 |     0.0297 |            0.0028 |       0.9062 |                                        |
| sim_c13k_desc_n100_p182  |        182 |      0.4489 |     0.0286 |            0.0004 |       0.9848 |                                        |
| sim_c13k_desc_n100_p1500 |       1500 |      0.4627 |     0.0299 |            0.0004 |       0.9853 |                                        |
| sim_c13k_exp_n16_p182    |        182 |      0.4701 |     0.0282 |            0.0029 |       0.8971 |                                        |
| sim_c13k_exp_n16_p1500   |       1500 |      0.4521 |     0.0302 |            0.0027 |       0.9097 |                                        |
| sim_c13k_exp_n100_p182   |        182 |      0.4525 |     0.0331 |            0.0004 |       0.987  |                                        |
| sim_c13k_exp_n100_p1500  |       1500 |      0.4555 |     0.0287 |            0.0004 |       0.9847 |                                        |
| sim_cpc_desc_n16_p182    |        182 |      0.4922 |     0.0293 |            0.0027 |       0.9076 |                                        |
| sim_cpc_desc_n16_p1500   |       1500 |      0.4859 |     0.0316 |            0.0028 |       0.9109 |                                        |
| sim_cpc_desc_n100_p182   |        182 |      0.4835 |     0.0272 |            0.0004 |       0.9836 |                                        |
| sim_cpc_desc_n100_p1500  |       1500 |      0.4819 |     0.0276 |            0.0004 |       0.9839 |                                        |
| sim_cpc_exp_n16_p182     |        182 |      0.4669 |     0.0293 |            0.0028 |       0.9037 |                                        |
| sim_cpc_exp_n16_p1500    |       1500 |      0.4684 |     0.0293 |            0.0027 |       0.9068 |                                        |
| sim_cpc_exp_n100_p182    |        182 |      0.4734 |     0.0243 |            0.0004 |       0.9815 |                                        |
| sim_cpc_exp_n100_p1500   |       1500 |      0.4752 |     0.0257 |            0.0004 |       0.9826 |                                        |
| suf_p60_n8               |         60 |      0.5038 |     0.0256 |            0.0056 |       0.7799 |                                        |
| suf_p60_n16              |         60 |      0.516  |     0.0293 |            0.0025 |       0.9151 |                                        |
| suf_p60_n32              |         60 |      0.4558 |     0.0234 |            0.0014 |       0.9397 |                                        |
| suf_p60_n128             |         60 |      0.4587 |     0.0288 |            0.0004 |       0.9878 |                                        |
| suf_p120_n8              |        120 |      0.4671 |     0.0306 |            0.0058 |       0.8091 |                                        |
| suf_p120_n16             |        120 |      0.4378 |     0.0291 |            0.0029 |       0.9015 |                                        |
| suf_p120_n32             |        120 |      0.4547 |     0.0271 |            0.0014 |       0.9484 |                                        |
| suf_p120_n128            |        120 |      0.4712 |     0.0254 |            0.0004 |       0.9862 |                                        |
| suf_p250_n8              |        250 |      0.4595 |     0.0333 |            0.0056 |       0.8326 |                                        |
| suf_p250_n16             |        250 |      0.4878 |     0.0293 |            0.0029 |       0.8996 |                                        |
| suf_p250_n32             |        250 |      0.4574 |     0.0332 |            0.0013 |       0.9604 |                                        |
| suf_p250_n128            |        250 |      0.4651 |     0.0311 |            0.0003 |       0.9891 |                                        |
| suf_p500_n8              |        500 |      0.4812 |     0.0317 |            0.0056 |       0.8237 |                                        |
| suf_p500_n16             |        500 |      0.4666 |     0.0301 |            0.0028 |       0.9068 |                                        |
| suf_p500_n32             |        500 |      0.4759 |     0.0294 |            0.0014 |       0.9531 |                                        |
| suf_p500_n128            |        500 |      0.4608 |     0.0273 |            0.0003 |       0.9873 |                                        |
| suf_p1000_n8             |       1000 |      0.4726 |     0.034  |            0.0054 |       0.8395 |                                        |
| suf_p1000_n16            |       1000 |      0.4636 |     0.0304 |            0.0028 |       0.9095 |                                        |
| suf_p1000_n32            |       1000 |      0.4653 |     0.0292 |            0.0014 |       0.9526 |                                        |
| suf_p1000_n128           |       1000 |      0.4696 |     0.0288 |            0.0003 |       0.988  |                                        |

## Table 2. Fit quality of every (arm x architecture) cell.

_Mean over 10 seeds on a held-out 15% split. Completeness normalises by that arm's noise ceiling._

|                               |   test_mse |   test_mse_sd |   epochs |   fit_s |   completeness |
|:------------------------------|-----------:|--------------:|---------:|--------:|---------------:|
| ('c13k_desc', 'CPT_NET')      |     0.0135 |        0.0001 |    170.6 |  0.7771 |         0.8322 |
| ('c13k_desc', 'GBT')          |     0.0118 |        0      |    400   |  1.7574 |         0.88   |
| ('c13k_desc', 'LINEAR')       |     0.0168 |        0.0005 |    577.8 |  0.2768 |         0.7392 |
| ('c13k_desc', 'MLP_L')        |     0.0123 |        0.0009 |    160.9 |  0.809  |         0.8659 |
| ('c13k_desc', 'MLP_S')        |     0.0131 |        0.0005 |    110.4 |  0.1244 |         0.8434 |
| ('c13k_desc', 'RESMLP')       |     0.0128 |        0.0012 |    126.3 |  3.8644 |         0.8519 |
| ('c13k_desc', 'SET')          |     0.0134 |        0.0008 |    170.4 |  0.7977 |         0.835  |
| ('c13k_desc', 'TRANSFORMER')  |     0.0134 |        0.0013 |    205.2 | 14.1194 |         0.835  |
| ('c13k_exp', 'CPT_NET')       |     0.0215 |        0      |     63.6 |  1.3443 |         0.6562 |
| ('c13k_exp', 'GBT')           |     0.0095 |        0      |    400   |  1.7257 |         0.9367 |
| ('c13k_exp', 'LINEAR')        |     0.0205 |        0      |    523.3 |  1.0226 |         0.6796 |
| ('c13k_exp', 'MLP_L')         |     0.0106 |        0.0003 |     89.4 |  1.9657 |         0.9109 |
| ('c13k_exp', 'MLP_S')         |     0.0109 |        0.0002 |    117.5 |  0.5618 |         0.9039 |
| ('c13k_exp', 'RESMLP')        |     0.0106 |        0.0003 |     67.2 |  9.0981 |         0.9109 |
| ('c13k_exp', 'SET')           |     0.0114 |        0.0002 |    147.6 |  3.1771 |         0.8923 |
| ('c13k_exp', 'TRANSFORMER')   |     0.0105 |        0.0002 |    154.9 | 49.8664 |         0.9133 |
| ('cpc18_desc', 'CPT_NET')     |     0.0199 |        0      |    356   |  0.2421 |         0.684  |
| ('cpc18_desc', 'GBT')         |     0.0092 |        0      |    400   |  0.2967 |         0.8668 |
| ('cpc18_desc', 'LINEAR')      |     0.0593 |        0.0041 |     45.9 |  0.0106 |         0.0109 |
| ('cpc18_desc', 'MLP_L')       |     0.0387 |        0.016  |     98.5 |  0.1117 |         0.3628 |
| ('cpc18_desc', 'MLP_S')       |     0.0477 |        0.0211 |     88.7 |  0.0247 |         0.2091 |
| ('cpc18_desc', 'RESMLP')      |     0.0294 |        0.0117 |     70.8 |  0.3941 |         0.5217 |
| ('cpc18_desc', 'SET')         |     0.023  |        0.015  |    128.1 |  0.0991 |         0.631  |
| ('cpc18_desc', 'TRANSFORMER') |     0.0276 |        0.0148 |    118.2 |  0.881  |         0.5525 |
| ('cpc18_exp', 'CPT_NET')      |     0.026  |        0      |    359.9 |  0.2431 |         0.6385 |
| ('cpc18_exp', 'GBT')          |     0.0118 |        0      |    400   |  0.31   |         0.8381 |
| ('cpc18_exp', 'LINEAR')       |     0.0971 |        0.0036 |     44.2 |  0.0103 |        -0.3612 |
| ('cpc18_exp', 'MLP_L')        |     0.036  |        0.0116 |    209.4 |  0.234  |         0.4979 |
| ('cpc18_exp', 'MLP_S')        |     0.0428 |        0.0119 |    132.2 |  0.0363 |         0.4023 |
| ('cpc18_exp', 'RESMLP')       |     0.0433 |        0.0063 |     90.9 |  0.5114 |         0.3952 |
| ('cpc18_exp', 'SET')          |     0.0273 |        0.0056 |    130.4 |  0.0992 |         0.6202 |
| ('cpc18_exp', 'TRANSFORMER')  |     0.0676 |        0.0263 |    121.4 |  0.8987 |         0.0535 |

## Table 3. H1. Two-factor variance decomposition of every readout.

_Partial omega^2 from an unweighted-means ANOVA, seeds as replicates. CPT coordinates use the theory screen, behavioural indices the looser behaviour screen; dropped_archs lists architectures with fewer than three surviving runs in some arm._

| value               |   n |   omega2_dataset |   omega2_arch |   omega2_interaction |   eta2_dataset |   eta2_arch |   eta2_interaction |   eta2_error | screen              | dropped_archs   |
|:--------------------|----:|-----------------:|--------------:|---------------------:|---------------:|------------:|-------------------:|-------------:|:--------------------|:----------------|
| alpha               | 255 |           0.3667 |        0.8028 |               0.2904 |         0.0976 |      0.6762 |             0.0792 |       0.147  | theory_identifiable | LINEAR          |
| beta                | 255 |           0.3596 |        0.4869 |               0.2934 |         0.1962 |      0.3328 |             0.1663 |       0.3047 | theory_identifiable | LINEAR          |
| lam                 | 255 |           0.3139 |        0.2113 |               0.4429 |         0.1864 |      0.1158 |             0.344  |       0.3538 | theory_identifiable | LINEAR          |
| gamma_pos           | 255 |           0.5194 |        0.453  |               0.505  |         0.2784 |      0.217  |             0.2779 |       0.2268 | theory_identifiable | LINEAR          |
| gamma_neg           | 255 |           0.2874 |        0.0438 |               0.3597 |         0.2068 |      0.0345 |             0.3151 |       0.4436 | theory_identifiable | LINEAR          |
| tau                 | 255 |           0.628  |        0.7643 |               0.4118 |         0.2565 |      0.4928 |             0.1163 |       0.1343 | theory_identifiable | LINEAR          |
| rare_event_index    | 309 |           0.6941 |        0.4965 |               0.3514 |         0.4754 |      0.2104 |             0.1272 |       0.187  | identifiable        | nan             |
| certainty_index     | 309 |           0.6888 |        0.4854 |               0.5604 |         0.4095 |      0.178  |             0.2474 |       0.1652 | identifiable        | nan             |
| reflection_index    | 309 |           0.1026 |        0.5596 |               0.533  |         0.0352 |      0.3671 |             0.3433 |       0.2544 | identifiable        | nan             |
| loss_aversion_kappa | 309 |           0.2314 |        0.2439 |               0.0724 |         0.183  |      0.2033 |             0.0859 |       0.5278 | identifiable        | nan             |
| ev_consistency      | 309 |           0.2304 |        0.6955 |               0.434  |         0.0711 |      0.5306 |             0.192  |       0.2062 | identifiable        | nan             |

## Table 4. Identifiability screens, runs kept per cell.

_behaviour screen: probe SD >= 0.02. theory screen: that, plus no CPT parameter pinned to a fitting bound._

|                                       | CPT_NET   | GBT   | LINEAR   | MLP_L   | MLP_S   | RESMLP   | SET   | TRANSFORMER   |
|:--------------------------------------|:----------|:------|:---------|:--------|:--------|:---------|:------|:--------------|
| ('identifiable', 'c13k_desc')         | 10/10     | 10/10 | 10/10    | 10/10   | 10/10   | 10/10    | 10/10 | 10/10         |
| ('identifiable', 'c13k_exp')          | 10/10     | 10/10 | 10/10    | 10/10   | 10/10   | 10/10    | 10/10 | 10/10         |
| ('identifiable', 'cpc18_desc')        | 10/10     | 10/10 | 9/10     | 8/10    | 5/10    | 10/10    | 9/10  | 10/10         |
| ('identifiable', 'cpc18_exp')         | 10/10     | 10/10 | 9/10     | 10/10   | 10/10   | 10/10    | 10/10 | 9/10          |
| ('theory_identifiable', 'c13k_desc')  | 10/10     | 10/10 | 10/10    | 10/10   | 10/10   | 10/10    | 9/10  | 10/10         |
| ('theory_identifiable', 'c13k_exp')   | 10/10     | 10/10 | 10/10    | 10/10   | 10/10   | 10/10    | 10/10 | 7/10          |
| ('theory_identifiable', 'cpc18_desc') | 10/10     | 10/10 | 1/10     | 7/10    | 3/10    | 6/10     | 9/10  | 9/10          |
| ('theory_identifiable', 'cpc18_exp')  | 10/10     | 10/10 | 0/10     | 10/10   | 10/10   | 9/10     | 10/10 | 6/10          |

## Table 5. Sensitivity of H1 to the identifiability threshold.

_Behavioural indices only, so the comparison keeps all six architectures at every threshold._

|   probe_sd_threshold |   n_runs |   median_omega2_dataset |   median_omega2_arch |   n_readouts_dataset_wins | dropped_archs   |
|---------------------:|---------:|------------------------:|---------------------:|--------------------------:|:----------------|
|                 0    |      320 |                  0.2608 |               0.4824 |                         3 | nan             |
|                 0.01 |      317 |                  0.2559 |               0.5062 |                         3 | nan             |
|                 0.02 |      309 |                  0.2314 |               0.4965 |                         2 | nan             |
|                 0.05 |      290 |                  0.3396 |               0.5016 |                         2 | LINEAR          |
|                 0.1  |      285 |                  0.3497 |               0.4957 |                         2 | LINEAR          |

## Table 6. H2. Recovery of a known agent from two problem distributions.

_One simulated CPT agent, identical parameters, simulated on the c13k and CPC18 problem sets; every network then read back through the same probe battery._

| dataset          |   ('alpha', 'mean') |   ('alpha', 'std') |   ('beta', 'mean') |   ('beta', 'std') |   ('lam', 'mean') |   ('lam', 'std') |   ('gamma_pos', 'mean') |   ('gamma_pos', 'std') |   ('gamma_neg', 'mean') |   ('gamma_neg', 'std') |   ('tau', 'mean') |   ('tau', 'std') |
|:-----------------|--------------------:|-------------------:|-------------------:|------------------:|------------------:|-----------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|------------------:|-----------------:|
| syn_c13k         |              0.8029 |             0.0624 |             1.1101 |            0.285  |            2.9986 |           1.3111 |                  0.6031 |                 0.0833 |                  0.7067 |                 0.1422 |            4.4863 |           1.6921 |
| syn_cpc          |              0.7856 |             0.1119 |             1.1692 |            0.2742 |            2.4369 |           1.2183 |                  0.5972 |                 0.1299 |                  0.7654 |                 0.316  |            3.507  |           1.6841 |
| generating agent |            nan      |           nan      |           nan      |          nan      |          nan      |         nan      |                nan      |               nan      |                nan      |               nan      |          nan      |         nan      |

## Table 7. H3. Rare-event index on the 1,266 matched problems.

_Same problems, same respondents pool, protocol swapped. A sign flip is description > 0 and experience < 0._

| arch        |   ('mean', 'c13k_pair_desc') |   ('mean', 'c13k_pair_exp') |   ('std', 'c13k_pair_desc') |   ('std', 'c13k_pair_exp') |
|:------------|-----------------------------:|----------------------------:|----------------------------:|---------------------------:|
| CPT_NET     |                       0.0568 |                     -0.0005 |                      0.0003 |                     0.0004 |
| GBT         |                       0.0083 |                     -0.0198 |                      0      |                     0      |
| LINEAR      |                       0.0516 |                      0.0207 |                      0.0039 |                     0.005  |
| MLP_L       |                       0.0773 |                     -0.0005 |                      0.0107 |                     0.0338 |
| MLP_S       |                       0.0819 |                     -0.0455 |                      0.0235 |                     0.0214 |
| RESMLP      |                       0.0735 |                     -0.0038 |                      0.013  |                     0.0431 |
| SET         |                       0.1195 |                      0.1022 |                      0.0132 |                     0.0087 |
| TRANSFORMER |                       0.1129 |                      0.0623 |                      0.015  |                     0.021  |

## Table 8. H4. Cross-provenance transfer, noise-ceiling normalised.

_Averaged over architectures and seeds. Rows train, columns evaluate; 1.0 would mean everything explainable was explained._

| train      |   c13k_exp |   c13k_desc |   cpc18_exp |   cpc18_desc |
|:-----------|-----------:|------------:|------------:|-------------:|
| c13k_exp   |     0.8507 |      0.8419 |      0.6535 |       0.7581 |
| c13k_desc  |     0.7109 |      0.8354 |      0.3452 |       0.653  |
| cpc18_exp  |     0.2545 |      0.2039 |      0.3855 |       0.4193 |
| cpc18_desc |     0.367  |      0.425  |      0.2013 |       0.4796 |

## Table 9. Controls. Size-matched and feature-reweighted c13k.

_c13k sub-sampled to CPC18's 182 problems, and reweighted to CPC18's feature distribution, to separate provenance from sheer sample size._

|                                |   alpha |   beta |    lam |   gamma_pos |   gamma_neg |    tau |   rare_event_index |
|:-------------------------------|--------:|-------:|-------:|------------:|------------:|-------:|-------------------:|
| ('c13k_desc', 182.0, nan)      |  0.7883 | 0.9527 | 1.1802 |      0.5892 |      0.5379 | 5.2763 |             0.0847 |
| ('c13k_exp', 182.0, nan)       |  0.8459 | 0.9667 | 1.2154 |      0.7672 |      0.9032 | 6.5679 |             0.0323 |
| ('c13k_exp', nan, 'cpc18_exp') |  0.7498 | 0.6462 | 0.9836 |      0.81   |      0.7906 | 7.8394 |             0.0028 |

## Table 10. Population arm. Two labs, same problems, same protocol.

_An upper bound on how far respondent population alone moves the readout, for comparison against the provenance gaps above._

| dataset            |   ('alpha', 'mean') |   ('alpha', 'std') |   ('beta', 'mean') |   ('beta', 'std') |   ('lam', 'mean') |   ('lam', 'std') |   ('gamma_pos', 'mean') |   ('gamma_pos', 'std') |   ('gamma_neg', 'mean') |   ('gamma_neg', 'std') |   ('tau', 'mean') |   ('tau', 'std') |   ('rare_event_index', 'mean') |   ('rare_event_index', 'std') |   ('certainty_index', 'mean') |   ('certainty_index', 'std') |   ('reflection_index', 'mean') |   ('reflection_index', 'std') |   ('loss_aversion_kappa', 'mean') |   ('loss_aversion_kappa', 'std') |   ('ev_consistency', 'mean') |   ('ev_consistency', 'std') |
|:-------------------|--------------------:|-------------------:|-------------------:|------------------:|------------------:|-----------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|------------------:|-----------------:|-------------------------------:|------------------------------:|------------------------------:|-----------------------------:|-------------------------------:|------------------------------:|----------------------------------:|---------------------------------:|-----------------------------:|----------------------------:|
| cpc18_rehovot_exp  |              0.7416 |             0.1379 |             0.7716 |            0.2093 |            0.9909 |           0.2794 |                  0.9125 |                 0.2822 |                  0.9475 |                 0.2397 |           10.6129 |           5.9038 |                        -0.0418 |                        0.0472 |                       -0.0549 |                       0.1133 |                         0.1247 |                        0.1108 |                            1.1853 |                           0.4884 |                       0.7964 |                      0.0959 |
| cpc18_technion_exp |              0.7447 |             0.1126 |             0.7984 |            0.2093 |            1.0537 |           0.2813 |                  0.8375 |                 0.1886 |                  1.0097 |                 0.3562 |           14.0153 |           7.1406 |                        -0.0291 |                        0.0411 |                       -0.0033 |                       0.1064 |                         0.1832 |                        0.1571 |                            1.2069 |                           0.6104 |                       0.8177 |                      0.1032 |

## Table 11. Where the contest is decided: variance shares by abstraction level.

_Distance-based R^2 for each factor, read at three levels of abstraction: the raw 1,742 probe responses, the five behavioural indices, and the six CPT coordinates. Every level is standardised before the distances, so shares are comparable across very different dimensionalities, and all three are read off the same runs. The second block repeats the two levels that survive the looser screen on all of it._

|                                                     |   dim |   r2_dataset |   r2_arch |   ratio |
|:----------------------------------------------------|------:|-------------:|----------:|--------:|
| ('theory screen (n=276)', 'raw probe responses')    |  1742 |       0.1954 |    0.4342 |  0.4501 |
| ('theory screen (n=276)', 'behavioural indices')    |     5 |       0.2407 |    0.3481 |  0.6915 |
| ('theory screen (n=276)', 'CPT coordinates')        |     6 |       0.2044 |    0.3131 |  0.6528 |
| ('behaviour screen (n=309)', 'raw probe responses') |  1742 |       0.1489 |    0.3873 |  0.3845 |
| ('behaviour screen (n=309)', 'behavioural indices') |     5 |       0.2312 |    0.2955 |  0.7821 |

## Table 12. Probe-level decomposition by problem family.

_The two-way decomposition run separately on each of the 1,742 probes, then averaged within family. Positive diff means provenance explains more of that family's variance than architecture does._

| group       |   omega2_dataset_mean |   omega2_dataset_count |   omega2_arch_mean |   diff_mean |
|:------------|----------------------:|-----------------------:|-------------------:|------------:|
| ce_gain     |                0.3411 |                    840 |             0.5734 |     -0.2323 |
| ce_loss     |                0.3423 |                    840 |             0.6245 |     -0.2822 |
| la_ladder   |                0.3313 |                     12 |             0.6328 |     -0.3015 |
| ladder_gain |                0.4394 |                     15 |             0.4259 |      0.0134 |
| ladder_loss |                0.4457 |                     15 |             0.4761 |     -0.0304 |
| mixed       |                0.3718 |                     20 |             0.6254 |     -0.2536 |

## Table 13. Experience gradient: is each readout monotone in block, and does the trend replicate?

_Spearman rho between block index and readout, computed within architecture and averaged; p from permuting block labels within architecture, so each architecture keeps its own level and spread and only the ordering is tested. CPC18 blocks are within-problem, choices13k blocks are between-problem; the two share no problems, respondents or country._

|                                  |   mean_rho |   n_arch |   min_rho |   max_rho |   same_sign |   p_value |
|:---------------------------------|-----------:|---------:|----------:|----------:|------------:|----------:|
| ('c13k', 'alpha')                |    -0.0567 |        8 |   -0.698  |    0.9695 |           5 |    0.4857 |
| ('c13k', 'beta')                 |     0.1529 |        8 |   -0.8    |    0.5894 |           6 |    0.0612 |
| ('c13k', 'lam')                  |     0.0822 |        8 |   -0.4    |    0.4731 |           4 |    0.3155 |
| ('c13k', 'gamma_pos')            |    -0.1842 |        8 |   -0.8    |    0.38   |           6 |    0.0236 |
| ('c13k', 'gamma_neg')            |     0.3209 |        8 |   -0.0634 |    0.8221 |           6 |    0.0002 |
| ('c13k', 'tau')                  |    -0.0509 |        8 |   -0.2947 |    0.4    |           4 |    0.5387 |
| ('c13k', 'rare_event_index')     |    -0.2582 |        8 |   -0.636  |    0.2404 |           6 |    0.0016 |
| ('c13k', 'certainty_index')      |     0.056  |        8 |   -0.9695 |    0.5662 |           5 |    0.4865 |
| ('c13k', 'reflection_index')     |    -0.178  |        8 |   -0.6282 |    0.6    |           6 |    0.025  |
| ('c13k', 'loss_aversion_kappa')  |     0.269  |        8 |   -0.4    |    0.7445 |           5 |    0.002  |
| ('c13k', 'ev_consistency')       |     0.1204 |        8 |   -0.6    |    0.4853 |           5 |    0.1352 |
| ('cpc18', 'alpha')               |     0.2685 |        7 |   -0.0393 |    0.5897 |           6 |    0.002  |
| ('cpc18', 'beta')                |     0.0928 |        7 |   -0.6214 |    0.7    |           4 |    0.2703 |
| ('cpc18', 'lam')                 |     0.0542 |        7 |   -0.688  |    1      |           4 |    0.5273 |
| ('cpc18', 'gamma_pos')           |     0.4629 |        7 |   -0.0566 |    0.9829 |           6 |    0.0002 |
| ('cpc18', 'gamma_neg')           |     0.5361 |        7 |    0.1512 |    0.9829 |           7 |    0.0002 |
| ('cpc18', 'tau')                 |     0.3643 |        7 |   -0.2315 |    0.8846 |           6 |    0.0002 |
| ('cpc18', 'rare_event_index')    |    -0.3764 |        8 |   -0.9911 |   -0.034  |           8 |    0.0002 |
| ('cpc18', 'certainty_index')     |    -0.3962 |        8 |   -0.6901 |   -0.0453 |           8 |    0.0002 |
| ('cpc18', 'reflection_index')    |    -0.1904 |        8 |   -0.887  |    0.4023 |           4 |    0.0102 |
| ('cpc18', 'loss_aversion_kappa') |    -0.0389 |        8 |   -0.2236 |    0.2241 |           4 |    0.6163 |
| ('cpc18', 'ev_consistency')      |    -0.1639 |        8 |   -1      |    0.2825 |           6 |    0.033  |

## Table 14. Learning-rate calibration.

_Validation MSE normalised by each arm's variance, averaged over a large arm and a small one and over two seeds, under the same 600-epoch budget the reported runs get. Calibrating every architecture the same way keeps tuning differences from being scored as architecture differences: a single shared rate left SET and MLP_L short of their own optimum and made RESMLP collapse to a constant._

| arch        |   0.0003 |   0.001 |   0.003 |   0.01 |   0.03 |   selected_lr |
|:------------|---------:|--------:|--------:|-------:|-------:|--------------:|
| CPT_NET     |   0.5575 |  0.5103 |  0.3981 | 0.3642 | 0.3632 |        0.03   |
| LINEAR      |   0.7654 |  0.7197 |  0.7171 | 0.7183 | 0.726  |        0.003  |
| MLP_L       |   0.4749 |  0.4265 |  0.4774 | 0.5081 | 0.4973 |        0.001  |
| MLP_S       |   0.5429 |  0.5463 |  0.5572 | 0.4586 | 0.4165 |        0.03   |
| RESMLP      |   0.4194 |  0.4781 |  4.212  | 3.8399 | 5.0659 |        0.0003 |
| SET         |   0.6137 |  0.6165 |  0.4628 | 0.3424 | 0.623  |        0.01   |
| TRANSFORMER |   0.611  |  0.399  |  0.4556 | 0.6105 | 4.53   |        0.001  |

## Table 15. Controlled factorial: effect of each component of provenance.

_2^4 design against a known generating agent -- problem pool x agent weighting function x respondents per problem x arm size -- with architecture entered as a fifth factor so the comparison sits on one scale. Every factor is fully crossed, so unlike the observational arms each share is free of confounding. The two agents differ only in their probability weighting (Prelec gamma 0.65 vs 1.0), which is the one property the rare-event index is built to detect._

| value               |   n |   omega2_problems |   omega2_agent |   omega2_n_subj |   omega2_n_problems |   omega2_arch |
|:--------------------|----:|------------------:|---------------:|----------------:|--------------------:|--------------:|
| alpha               | 518 |            0.0617 |         0.7446 |          0.1341 |              0.4083 |        0.8942 |
| beta                | 518 |            0.0552 |         0      |          0.0047 |              0.0541 |        0.8367 |
| lam                 | 518 |            0.0477 |         0.0787 |          0.0031 |              0.0184 |        0.7284 |
| gamma_pos           | 518 |            0      |         0.8073 |          0.028  |              0.4379 |        0.5104 |
| gamma_neg           | 518 |            0.0785 |         0.7375 |          0      |              0.2087 |        0.8615 |
| tau                 | 518 |            0.1313 |         0      |          0.0019 |              0.7417 |        0.9698 |
| rare_event_index    | 640 |            0.2301 |         0.8585 |          0      |              0      |        0.6532 |
| certainty_index     | 640 |            0.0632 |         0.8236 |          0      |              0.0229 |        0.4469 |
| reflection_index    | 640 |            0.0022 |         0.102  |          0.0182 |              0      |        0.7788 |
| loss_aversion_kappa | 640 |            0      |         0.0076 |          0.0239 |              0.544  |        0.427  |
| ev_consistency      | 640 |            0.1137 |         0.4666 |          0.2345 |              0.6356 |        0.7679 |

## Table 16. Design sufficiency: recovery error by arm size and respondents per problem.

_Known generating agent, only the design varies. Error is |recovered - true| / |true| averaged over the six CPT coordinates so parameters on different scales are comparable. This is the prior question the rest of the paper depends on: without it, 'CPC18 models encode a different theory' cannot be told apart from '182 problems is too few to tell'._

|             |   n_runs |   err_alpha |   err_beta |   err_lam |   err_gamma_pos |   err_gamma_neg |   err_tau |   mean_rel_err |   max_rel_err |
|:------------|---------:|------------:|-----------:|----------:|----------------:|----------------:|----------:|---------------:|--------------:|
| (8, 60)     |       11 |      0.1375 |     0.3534 |    0.0459 |          0.0019 |          0.091  |    0.1957 |         0.1376 |        0.3534 |
| (8, 120)    |       16 |      0.2707 |     0.3036 |    0.1101 |          0.3154 |          0.137  |    0.4276 |         0.2607 |        0.4276 |
| (8, 250)    |       14 |      0.1384 |     0.4034 |    0.2313 |          0.3108 |          0.0695 |    0.4445 |         0.2663 |        0.4445 |
| (8, 500)    |       16 |      0.1154 |     0.3884 |    0.055  |          0.1512 |          0.4295 |    0.4568 |         0.266  |        0.4568 |
| (8, 1000)   |       18 |      0.1527 |     0.4351 |    0.1776 |          0.2009 |          0.0844 |    0.32   |         0.2285 |        0.4351 |
| (16, 60)    |       12 |      0.0305 |     0.526  |    0.1013 |          0.386  |          0.0544 |    0.4184 |         0.2527 |        0.526  |
| (16, 120)   |       10 |      0.1608 |     0.1686 |    0.0082 |          0.1004 |          0.3202 |    0.537  |         0.2159 |        0.537  |
| (16, 250)   |       13 |      0.0898 |     0.1754 |    0.3157 |          0.2367 |          0.0344 |    0.4863 |         0.223  |        0.4863 |
| (16, 500)   |       24 |      0.1041 |     0.4053 |    0.0018 |          0.0464 |          0.2767 |    0.3096 |         0.1907 |        0.4053 |
| (16, 1000)  |       23 |      0.1531 |     0.4399 |    0.4528 |          0.1558 |          0.0563 |    0.3025 |         0.2601 |        0.4528 |
| (32, 60)    |        8 |      0.0061 |     0.17   |    0.1823 |          0.1906 |          0.5579 |    0.6024 |         0.2849 |        0.6024 |
| (32, 120)   |       19 |      0.1329 |     0.542  |    0.0132 |          0.3527 |          0.0161 |    0.3532 |         0.235  |        0.542  |
| (32, 250)   |       22 |      0.1631 |     0.5522 |    0.08   |          0.247  |          0.0736 |    0.3284 |         0.2407 |        0.5522 |
| (32, 500)   |       21 |      0.1202 |     0.4147 |    0.146  |          0.263  |          0.0635 |    0.3225 |         0.2216 |        0.4147 |
| (32, 1000)  |       24 |      0.0885 |     0.3772 |    0.2163 |          0.1155 |          0.0629 |    0.3171 |         0.1962 |        0.3772 |
| (128, 60)   |       13 |      0.296  |     0.5986 |    0.2803 |          0.2397 |          0.0277 |    0.4908 |         0.3222 |        0.5986 |
| (128, 120)  |        7 |      0.1783 |     0.1802 |    0.1302 |          0.2701 |          0.1009 |    0.4786 |         0.223  |        0.4786 |
| (128, 250)  |       23 |      0.0932 |     0.478  |    0.3228 |          0.2586 |          0.0885 |    0.3603 |         0.2669 |        0.478  |
| (128, 500)  |       23 |      0.1136 |     0.3582 |    0.272  |          0.1698 |          0.0802 |    0.3266 |         0.2201 |        0.3582 |
| (128, 1000) |       24 |      0.0603 |     0.337  |    0.2488 |          0.1203 |          0.0758 |    0.3208 |         0.1938 |        0.337  |

## Table 17. Smallest arm size reaching 10% recovery error.

_NaN means no swept size reached the tolerance at that respondent count -- which is the answer, not a gap._

|   n_subj |   min_n_problems_within_0.1 |   best_error |
|---------:|----------------------------:|-------------:|
|        8 |                         nan |       0.1376 |
|       16 |                         nan |       0.1907 |
|       32 |                         nan |       0.1962 |
|      128 |                         nan |       0.1938 |

## Table 18. Artifact floor: observed effects against what the readout manufactures when nothing differs.

_On the fixed-agent arms every run shares the same true coordinates, so any omega^2 still reported is produced by the readout itself. Each observed effect is compared with the floor for its own factor, because architecture perturbs the prediction surface directly and has a much higher floor than swapping the problem pool does._

| value     |   arch_observed |   arch_artifact_floor |   arch_excess | arch_survives   |   dataset_observed |   dataset_artifact_floor |   dataset_excess | dataset_survives   |
|:----------|----------------:|----------------------:|--------------:|:----------------|-------------------:|-------------------------:|-----------------:|:-------------------|
| alpha     |          0.8028 |                0.7506 |        0.0522 | True            |             0.3667 |                   0.0617 |           0.3051 | True               |
| beta      |          0.4869 |                0.7825 |       -0.2956 | False           |             0.3596 |                   0.0552 |           0.3043 | True               |
| lam       |          0.2113 |                0.6673 |       -0.456  | False           |             0.3139 |                   0.0477 |           0.2662 | True               |
| gamma_pos |          0.453  |                0.5965 |       -0.1434 | False           |             0.5194 |                   0      |           0.5194 | True               |
| gamma_neg |          0.0438 |                0.4902 |       -0.4464 | False           |             0.2874 |                   0.0785 |           0.209  | True               |
| tau       |          0.7643 |                0.9804 |       -0.2161 | False           |             0.628  |                   0.1313 |           0.4967 | True               |

## Table 19. Bootstrap 95% intervals for each effect and its artifact floor.

_400 resamples within design cells, so every replicate keeps a balanced design. `clears` is True only when the observed interval lies strictly above the floor interval -- comparing point estimates alone would not support the claim._

| value     |   arch_lo |   arch_hi |   arch_floor_lo |   arch_floor_hi | arch_clears   |   ds_lo |   ds_hi |   ds_floor_lo |   ds_floor_hi | ds_clears   |
|:----------|----------:|----------:|----------------:|----------------:|:--------------|--------:|--------:|--------------:|--------------:|:------------|
| alpha     |    0.7834 |    0.8603 |          0.7653 |          0.8943 | False         |  0.2882 |  0.5101 |        0.0264 |        0.1228 | True        |
| beta      |    0.385  |    0.654  |          0.6959 |          0.8103 | False         |  0.2861 |  0.5413 |        0.0256 |        0.1006 | True        |
| lam       |    0.1523 |    0.3773 |          0.677  |          0.8378 | False         |  0.2585 |  0.4624 |        0.011  |        0.0846 | True        |
| gamma_pos |    0.3851 |    0.6482 |          0.5745 |          0.8432 | False         |  0.4567 |  0.6774 |        0      |        0.0094 | True        |
| gamma_neg |    0      |    0.2694 |          0.4668 |          0.898  | False         |  0.2368 |  0.4512 |        0.0501 |        0.2173 | True        |
| tau       |    0.7061 |    0.8631 |          0.9837 |          0.9913 | False         |  0.519  |  0.7834 |        0.0878 |        0.2563 | True        |
