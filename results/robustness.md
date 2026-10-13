# Robustness of the floor-clearing result

## 1. Which floor, and does the result depend on it?

The provenance floor used in the main analysis is the problem-pool share of the controlled factorial. The same factorial also measures the arm-size and respondent shares; 'largest component' takes the max of the three per bootstrap replicate, 'composite' enters all three as one 8-level factor so their interactions count too. Architecture is given its usual floor (sufficiency suite) and a second one from the controlled suite.

Clears: point = observed > floor; interval = observed 95% interval strictly above the floor interval (the rule used in the paper); Holm = Holm-adjusted bootstrap p < 0.05 across the 11 read-outs for H0: observed <= floor.

      factor                             floor_def  n_readouts  clears_point  clears_interval  clears_holm  median_floor  median_obs
architecture                     sufficiency suite          11             2                0            0         0.724       0.487
architecture                      controlled suite          11             1                0            0         0.768       0.487
  provenance                          problem pool          11            11               10           11         0.047       0.360
  provenance                              arm size          11             7                5            5         0.185       0.360
  provenance                           respondents          11            10                9           10         0.002       0.360
  provenance                     largest component          11             7                4            4         0.230       0.360
  provenance composite (pool x size x respondents)          11             5                3            3         0.430       0.360

## 2. Leave one architecture out

Interval test with 200 resamples; provenance floor = problem pool (as in the paper) and, separately, the composite floor; architecture floor = sufficiency suite. 'none' is the full design.

   left_out  n_readouts  archs_in_design  median_omega2_dataset  median_omega2_arch  provenance_clears  provenance_clears_composite  architecture_clears
       none          11                8                  0.360               0.487                 10                            3                    0
     LINEAR          11                7                  0.360               0.478                  9                            3                    0
      MLP_S          11                7                  0.398               0.538                 10                            4                    0
      MLP_L          11                7                  0.314               0.470                  9                            2                    0
     RESMLP          11                7                  0.323               0.484                 10                            3                    0
        SET          11                7                  0.466               0.494                 10                            3                    0
TRANSFORMER          11                7                  0.414               0.540                  9                            3                    0
    CPT_NET          11                7                  0.360               0.470                  8                            3                    0
        GBT          11                7                  0.309               0.483                 11                            2                    1

## 3. H5: do the distilled expressions sort by provenance or by architecture?

Each run's gain and loss expressions are evaluated on the 7 x 12 (x/100, p) grid the regression was fitted on, giving a 168-dimensional function fingerprint per run; PERMANOVA permutes within strata of the other factor (2,000 permutations). The syntactic view uses operator counts and program length instead, which is what Fig. S2c shows.

                       view  n_runs  dim  permanova_F_dataset  permanova_p_dataset  permanova_F_arch  permanova_p_arch  silhouette_dataset  silhouette_arch  r2_dataset  r2_arch
       functional (CE grid)      96  168                2.300                0.000             3.355             0.000              -0.100           -0.166       0.070    0.211
syntactic (operator counts)      96   14                0.805                0.761             1.424             0.012              -0.101           -0.205       0.026    0.102

(7 of 16128 evaluated values clipped to [-3, 3])
