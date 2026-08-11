# Provenance vs architecture - summary

## Identifiability screens (runs kept / total)

Two screens, because the two kinds of readout fail differently.
  behaviour  probe SD >= 0.02: the network responds to the battery at all. Used for the behavioural indices.
  theory     the above AND no CPT parameter pinned to its box constraint. Used for the CPT coordinates.

### identifiable
arch       CPT_NET    GBT LINEAR  MLP_L  MLP_S RESMLP    SET TRANSFORMER
dataset                                                                 
c13k_desc    10/10  10/10  10/10  10/10  10/10  10/10  10/10       10/10
c13k_exp     10/10  10/10  10/10  10/10  10/10  10/10  10/10       10/10
cpc18_desc   10/10  10/10   9/10   8/10   5/10  10/10   9/10       10/10
cpc18_exp    10/10  10/10   9/10  10/10  10/10  10/10  10/10        9/10
kept 309/320 main-grid runs

### theory_identifiable
arch       CPT_NET    GBT LINEAR  MLP_L  MLP_S RESMLP    SET TRANSFORMER
dataset                                                                 
c13k_desc    10/10  10/10  10/10  10/10  10/10  10/10   9/10       10/10
c13k_exp     10/10  10/10  10/10  10/10  10/10  10/10  10/10        7/10
cpc18_desc   10/10  10/10   1/10   7/10   3/10   6/10   9/10        9/10
cpc18_exp    10/10  10/10   0/10  10/10  10/10   9/10  10/10        6/10
kept 276/320 main-grid runs

## H1: partial omega^2 (dataset vs architecture)

              value              screen   n dropped_archs  omega2_dataset  omega2_arch  omega2_interaction
              alpha theory_identifiable 255        LINEAR           0.367        0.803               0.290
               beta theory_identifiable 255        LINEAR           0.360        0.487               0.293
                lam theory_identifiable 255        LINEAR           0.314        0.211               0.443
          gamma_pos theory_identifiable 255        LINEAR           0.519        0.453               0.505
          gamma_neg theory_identifiable 255        LINEAR           0.287        0.044               0.360
                tau theory_identifiable 255        LINEAR           0.628        0.764               0.412
   rare_event_index        identifiable 309                         0.694        0.497               0.351
    certainty_index        identifiable 309                         0.689        0.485               0.560
   reflection_index        identifiable 309                         0.103        0.560               0.533
loss_aversion_kappa        identifiable 309                         0.231        0.244               0.072
     ev_consistency        identifiable 309                         0.230        0.695               0.434

CPT coordinates (theory_identifiable, 6 readouts): median dataset 0.363 vs architecture 0.470; dataset wins 3/6

behavioural indices (identifiable, 5 readouts): median dataset 0.231 vs architecture 0.497; dataset wins 2/5

All 11 readouts pooled: median dataset 0.360 vs architecture 0.487; dataset wins 5/11

### Sensitivity of the behavioural indices to the identifiability threshold

 probe_sd_threshold  n_runs  median_omega2_dataset  median_omega2_arch  n_readouts_dataset_wins dropped_archs
               0.00     320                  0.261               0.482                        3              
               0.01     317                  0.256               0.506                        3              
               0.02     309                  0.231               0.497                        2              
               0.05     290                  0.340               0.502                        2        LINEAR
               0.10     285                  0.350               0.496                        2        LINEAR

## H1 multivariate (behavioural fingerprints)

[identifiable, n=309] PERMANOVA dataset: F=17.8, p=0.0005 | architecture: F=27.2, p=0.0005
[identifiable] Silhouette by dataset 0.058 vs by architecture 0.136

[all_runs, n=320] PERMANOVA dataset: F=19.5, p=0.0005 | architecture: F=24.1, p=0.0005
[all_runs] Silhouette by dataset 0.057 vs by architecture 0.116

## H2: same agent, different problem distributions

(identifiable runs only: 145/160)
True theta: {"alpha": 0.88, "beta": 0.88, "lam": 2.25, "gamma_pos": 0.65, "gamma_neg": 0.65, "tau": 6.0}
          alpha          beta           lam        gamma_pos        gamma_neg           tau       
           mean    std   mean    std   mean    std      mean    std      mean    std   mean    std
dataset                                                                                           
syn_c13k  0.803  0.062  1.110  0.285  2.999  1.311     0.603  0.083     0.707  0.142  4.486  1.692
syn_cpc   0.786  0.112  1.169  0.274  2.437  1.218     0.597  0.130     0.765  0.316  3.507  1.684

Recovered-parameter gap (syn_c13k - syn_cpc):
alpha        0.017
beta        -0.059
lam          0.562
gamma_pos    0.006
gamma_neg   -0.059
tau          0.979

## H3: rare-event index, matched problems (desc vs exp)

(identifiable runs only: 80/80)
                              mean     std
dataset        arch                       
c13k_pair_desc CPT_NET      0.0568  0.0003
               GBT          0.0083  0.0000
               LINEAR       0.0516  0.0039
               MLP_L        0.0773  0.0107
               MLP_S        0.0819  0.0235
               RESMLP       0.0735  0.0130
               SET          0.1195  0.0132
               TRANSFORMER  0.1129  0.0150
c13k_pair_exp  CPT_NET     -0.0005  0.0004
               GBT         -0.0198  0.0000
               LINEAR       0.0207  0.0050
               MLP_L       -0.0005  0.0338
               MLP_S       -0.0455  0.0214
               RESMLP      -0.0038  0.0431
               SET          0.1022  0.0087
               TRANSFORMER  0.0623  0.0210

Architectures showing the description>0 / experience<0 sign flip: 5/8

## Experience gradient (blocks as an ordered factor)

Spearman rho between block index and each readout, computed within architecture and averaged; p from permuting block labels within architecture.

source               value  mean_rho  n_arch  min_rho  max_rho  same_sign  p_value
  c13k               alpha    -0.057       8   -0.698    0.969          5    0.486
  c13k                beta     0.153       8   -0.800    0.589          6    0.061
  c13k                 lam     0.082       8   -0.400    0.473          4    0.316
  c13k           gamma_pos    -0.184       8   -0.800    0.380          6    0.024
  c13k           gamma_neg     0.321       8   -0.063    0.822          6    0.000
  c13k                 tau    -0.051       8   -0.295    0.400          4    0.539
  c13k    rare_event_index    -0.258       8   -0.636    0.240          6    0.002
  c13k     certainty_index     0.056       8   -0.969    0.566          5    0.487
  c13k    reflection_index    -0.178       8   -0.628    0.600          6    0.025
  c13k loss_aversion_kappa     0.269       8   -0.400    0.745          5    0.002
  c13k      ev_consistency     0.120       8   -0.600    0.485          5    0.135
 cpc18               alpha     0.269       7   -0.039    0.590          6    0.002
 cpc18                beta     0.093       7   -0.621    0.700          4    0.270
 cpc18                 lam     0.054       7   -0.688    1.000          4    0.527
 cpc18           gamma_pos     0.463       7   -0.057    0.983          6    0.000
 cpc18           gamma_neg     0.536       7    0.151    0.983          7    0.000
 cpc18                 tau     0.364       7   -0.232    0.885          6    0.000
 cpc18    rare_event_index    -0.376       8   -0.991   -0.034          8    0.000
 cpc18     certainty_index    -0.396       8   -0.690   -0.045          8    0.000
 cpc18    reflection_index    -0.190       8   -0.887    0.402          4    0.010
 cpc18 loss_aversion_kappa    -0.039       8   -0.224    0.224          4    0.616
 cpc18      ev_consistency    -0.164       8   -1.000    0.282          6    0.033

Readouts whose trend has the same sign in both sources: 5/11

## Controlled factorial: which component of provenance moves the learned theory?

2^4 design against a known generating agent (problem pool x agent weighting x respondents per problem x arm size), with architecture entered as a fifth factor. Every factor is crossed, so each share is free of the confounding that limits the observational arms.

              value   n  omega2_problems  omega2_agent  omega2_n_subj  omega2_n_problems  omega2_arch
              alpha 518            0.062         0.745          0.134              0.408        0.894
               beta 518            0.055         0.000          0.005              0.054        0.837
                lam 518            0.048         0.079          0.003              0.018        0.728
          gamma_pos 518            0.000         0.807          0.028              0.438        0.510
          gamma_neg 518            0.078         0.737          0.000              0.209        0.861
                tau 518            0.131         0.000          0.002              0.742        0.970
   rare_event_index 640            0.230         0.859          0.000              0.000        0.653
    certainty_index 640            0.063         0.824          0.000              0.023        0.447
   reflection_index 640            0.002         0.102          0.018              0.000        0.779
loss_aversion_kappa 640            0.000         0.008          0.024              0.544        0.427
     ev_consistency 640            0.114         0.467          0.234              0.636        0.768

Median across readouts:
omega2_arch          0.768
omega2_agent         0.467
omega2_n_problems    0.209
omega2_problems      0.062
omega2_n_subj        0.005

Largest median effect: arch

## Design sufficiency: how much data before the readout can be trusted?

Known generating agent; only arm size and respondents per problem vary. Error is |recovered - true| / |true| averaged over the six CPT coordinates, so parameters on different scales are comparable. Identifiable runs only.

Mean relative recovery error:
n_problems   60     120    250    500    1000
n_subj                                       
8           0.138  0.261  0.266  0.266  0.228
16          0.253  0.216  0.223  0.191  0.260
32          0.285  0.235  0.241  0.222  0.196
128         0.322  0.223  0.267  0.220  0.194

Smallest swept arm size reaching 10% error (NaN = unreachable at that respondent count):
 n_subj  min_n_problems_within_0.1  best_error
      8                        NaN       0.138
     16                        NaN       0.191
     32                        NaN       0.196
    128                        NaN       0.194

## Artifact floor: does either effect survive it?

On the fixed-agent arms every run shares the same true coordinates, so any omega^2 the readout still reports is manufactured by the readout. Architecture perturbs the prediction surface directly and has a high floor; swapping the problem pool barely moves it, so each observed effect is compared with the floor for its own factor.

              value  arch_observed  arch_artifact_floor  arch_excess  arch_survives  dataset_observed  dataset_artifact_floor  dataset_excess  dataset_survives
              alpha          0.803                0.751        0.052           True             0.367                   0.062           0.305              True
               beta          0.487                0.782       -0.296          False             0.360                   0.055           0.304              True
                lam          0.211                0.667       -0.456          False             0.314                   0.048           0.266              True
          gamma_pos          0.453                0.596       -0.143          False             0.519                   0.000           0.519              True
          gamma_neg          0.044                0.490       -0.446          False             0.287                   0.078           0.209              True
                tau          0.764                0.980       -0.216          False             0.628                   0.131           0.497              True
   rare_event_index          0.497                0.751       -0.254          False             0.694                   0.230           0.464              True
    certainty_index          0.485                0.371        0.114           True             0.689                   0.063           0.626              True
   reflection_index          0.560                0.731       -0.171          False             0.103                   0.002           0.100              True
loss_aversion_kappa          0.244                0.724       -0.480          False             0.231                   0.000           0.231              True
     ev_consistency          0.695                0.800       -0.105          False             0.230                   0.114           0.117              True

Architecture effects clearing their floor: 2/11
Provenance effects clearing their floor: 11/11

### Bootstrap 95% intervals (400 resamples)

              value            family  arch_lo  arch_hi  arch_floor_lo  arch_floor_hi  arch_clears  ds_lo  ds_hi  ds_floor_lo  ds_floor_hi  ds_clears
              alpha    CPT coordinate    0.783    0.860          0.765          0.894        False  0.288  0.510        0.026        0.123       True
               beta    CPT coordinate    0.385    0.654          0.696          0.810        False  0.286  0.541        0.026        0.101       True
                lam    CPT coordinate    0.152    0.377          0.677          0.838        False  0.258  0.462        0.011        0.085       True
          gamma_pos    CPT coordinate    0.385    0.648          0.575          0.843        False  0.457  0.677        0.000        0.009       True
          gamma_neg    CPT coordinate    0.000    0.269          0.467          0.898        False  0.237  0.451        0.050        0.217       True
                tau    CPT coordinate    0.706    0.863          0.984          0.991        False  0.519  0.783        0.088        0.256       True
   rare_event_index behavioural index    0.450    0.613          0.778          0.873        False  0.681  0.769        0.197        0.361       True
    certainty_index behavioural index    0.379    0.619          0.349          0.507        False  0.641  0.797        0.041        0.129       True
   reflection_index behavioural index    0.425    0.615          0.756          0.836        False  0.010  0.167        0.000        0.009       True
loss_aversion_kappa behavioural index    0.241    0.489          0.660          0.818        False  0.272  0.503        0.000        0.014       True
     ev_consistency behavioural index    0.672    0.795          0.804          0.891        False  0.089  0.241        0.098        0.225      False

Architecture: interval strictly above its floor in 0/11 coordinates
Provenance:   interval strictly above its floor in 10/11 coordinates

## Controls (size-matched / feature-reweighted)

(identifiable runs only: 107/120)
                                   alpha   beta    lam  gamma_pos  gamma_neg    tau  rare_event_index
dataset   subsample_n reweight_to                                                                    
c13k_desc 182.0       NaN          0.788  0.953  1.180      0.589      0.538  5.276             0.085
c13k_exp  182.0       NaN          0.846  0.967  1.215      0.767      0.903  6.568             0.032
          NaN         cpc18_exp    0.750  0.646  0.984      0.810      0.791  7.839             0.003

## Population arm (two labs, same problems/protocol)

(identifiable runs only: 61/80)
                    alpha   beta    lam  gamma_pos  gamma_neg     tau  rare_event_index
dataset                                                                                
cpc18_rehovot_exp   0.742  0.772  0.991      0.912      0.947  10.613            -0.042
cpc18_technion_exp  0.745  0.798  1.054      0.837      1.010  14.015            -0.029
