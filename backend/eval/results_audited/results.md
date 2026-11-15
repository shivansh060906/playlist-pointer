Retrieval eval: 50 questions, tolerance +/-30s, 2000 bootstrap resamples.

Intervals are 95% bootstrap CIs over questions. When two intervals overlap heavily, treat the configurations as indistinguishable; see the paired tables below for direct comparisons.

| chunking | configuration | hit@1 (95% CI) | hit@3 | hit@5 | MRR (95% CI) | median seek err (s, 95% CI) | avg chunk (s) | p50 latency (ms) |
|---|---|---|---|---|---|---|---|---|
| fixed | dense only | 0.88 (0.78-0.96) | 0.96 | 1.00 | 0.926 (0.864-0.980) | 23 (17-28) | 58 | 18 |
| fixed | bm25 only | 0.66 (0.54-0.78) | 0.84 | 0.92 | 0.760 (0.666-0.852) | 28 (25-30) | 58 | 0 |
| fixed | bm25 + glossary | 0.66 (0.52-0.78) | 0.76 | 0.92 | 0.746 (0.641-0.845) | 29 (25-31) | 58 | 1 |
| fixed | hybrid (RRF) | 0.84 (0.74-0.94) | 0.96 | 1.00 | 0.906 (0.843-0.960) | 27 (22-29) | 58 | 18 |
| fixed | hybrid RRF 2:1 | 0.82 (0.70-0.92) | 1.00 | 1.00 | 0.900 (0.837-0.957) | 25 (18-29) | 58 | 19 |
| fixed | hybrid RRF 3:1 | 0.82 (0.70-0.92) | 1.00 | 1.00 | 0.903 (0.840-0.957) | 26 (20-28) | 58 | 19 |
| fixed | hybrid + glossary | 0.78 (0.66-0.88) | 0.94 | 1.00 | 0.875 (0.805-0.935) | 27 (23-30) | 58 | 20 |
| fixed | hybrid 2:1 + glossary | 0.84 (0.74-0.94) | 1.00 | 1.00 | 0.907 (0.843-0.967) | 25 (18-28) | 58 | 20 |
| fixed | hybrid + rerank | 0.86 (0.76-0.94) | 0.94 | 0.96 | 0.895 (0.817-0.963) | 24 (16-28) | 58 | 613 |
| fixed | hybrid + glossary + rerank | 0.86 (0.76-0.94) | 0.94 | 0.96 | 0.897 (0.820-0.963) | 24 (16-28) | 58 | 612 |
| sliding | dense only | 0.94 (0.88-1.00) | 1.00 | 1.00 | 0.967 (0.927-1.000) | 14 (11-19) | 58 | 16 |
| sliding | bm25 only | 0.68 (0.54-0.80) | 0.90 | 0.92 | 0.793 (0.703-0.879) | 21 (15-25) | 58 | 1 |
| sliding | bm25 + glossary | 0.66 (0.52-0.80) | 0.86 | 0.92 | 0.769 (0.668-0.860) | 23 (15-28) | 58 | 1 |
| sliding | hybrid (RRF) | 0.88 (0.78-0.96) | 0.96 | 0.98 | 0.927 (0.868-0.980) | 19 (13-23) | 58 | 18 |
| sliding | hybrid RRF 2:1 | 0.90 (0.82-0.98) | 0.98 | 0.98 | 0.940 (0.885-0.987) | 19 (13-25) | 58 | 18 |
| sliding | hybrid RRF 3:1 | 0.90 (0.82-0.98) | 0.98 | 1.00 | 0.944 (0.894-0.990) | 19 (13-25) | 58 | 17 |
| sliding | hybrid + glossary | 0.80 (0.70-0.90) | 0.92 | 0.98 | 0.875 (0.801-0.941) | 20 (13-24) | 58 | 18 |
| sliding | hybrid 2:1 + glossary | 0.86 (0.76-0.94) | 0.98 | 0.98 | 0.913 (0.850-0.970) | 19 (14-24) | 58 | 18 |
| sliding | hybrid + rerank | 0.82 (0.72-0.92) | 0.94 | 0.98 | 0.881 (0.805-0.950) | 13 (10-19) | 58 | 591 |
| sliding | hybrid + glossary + rerank | 0.82 (0.72-0.92) | 0.94 | 0.98 | 0.885 (0.811-0.950) | 13 (10-19) | 58 | 622 |
| semantic | dense only | 0.86 (0.76-0.96) | 0.96 | 1.00 | 0.917 (0.853-0.970) | 12 (9-27) | 62 | 17 |
| semantic | bm25 only | 0.68 (0.56-0.80) | 0.88 | 0.90 | 0.782 (0.690-0.871) | 19 (9-34) | 62 | 0 |
| semantic | bm25 + glossary | 0.72 (0.60-0.84) | 0.86 | 0.92 | 0.799 (0.703-0.888) | 24 (11-37) | 62 | 1 |
| semantic | hybrid (RRF) | 0.84 (0.74-0.94) | 0.98 | 1.00 | 0.905 (0.843-0.963) | 22 (10-30) | 62 | 18 |
| semantic | hybrid RRF 2:1 | 0.88 (0.80-0.96) | 0.98 | 1.00 | 0.928 (0.872-0.980) | 12 (8-26) | 62 | 17 |
| semantic | hybrid RRF 3:1 | 0.86 (0.76-0.94) | 1.00 | 1.00 | 0.920 (0.863-0.970) | 13 (9-27) | 62 | 18 |
| semantic | hybrid + glossary | 0.90 (0.82-0.98) | 0.98 | 1.00 | 0.938 (0.885-0.987) | 25 (9-30) | 62 | 18 |
| semantic | hybrid 2:1 + glossary | 0.84 (0.74-0.94) | 0.98 | 1.00 | 0.908 (0.847-0.965) | 14 (9-28) | 62 | 17 |
| semantic | hybrid + rerank | 0.80 (0.68-0.90) | 0.94 | 0.98 | 0.877 (0.802-0.945) | 17 (9-27) | 62 | 806 |
| semantic | hybrid + glossary + rerank | 0.80 (0.68-0.90) | 0.94 | 0.98 | 0.877 (0.802-0.945) | 17 (9-27) | 62 | 808 |

## Configurations vs "dense only" (same chunking)

d = configuration minus baseline. Paired bootstrap on per-question differences, 95% CI; `*` = CI excludes 0. With this many comparisons, roughly 1 in 20 starred results can be chance.

| chunking | configuration | d hit@1 | d MRR |
|---|---|---|---|
| fixed | bm25 only | -0.220 [-0.360, -0.080]* | -0.166 [-0.267, -0.070]* |
| fixed | bm25 + glossary | -0.220 [-0.360, -0.100]* | -0.180 [-0.288, -0.084]* |
| fixed | hybrid (RRF) | -0.040 [-0.140, +0.060] | -0.020 [-0.093, +0.056] |
| fixed | hybrid RRF 2:1 | -0.060 [-0.160, +0.040] | -0.026 [-0.087, +0.039] |
| fixed | hybrid RRF 3:1 | -0.060 [-0.140, +0.020] | -0.022 [-0.075, +0.031] |
| fixed | hybrid + glossary | -0.100 [-0.220, +0.000] | -0.051 [-0.125, +0.020] |
| fixed | hybrid 2:1 + glossary | -0.040 [-0.120, +0.040] | -0.019 [-0.070, +0.032] |
| fixed | hybrid + rerank | -0.020 [-0.100, +0.060] | -0.031 [-0.106, +0.033] |
| fixed | hybrid + glossary + rerank | -0.020 [-0.100, +0.060] | -0.029 [-0.101, +0.033] |
| sliding | bm25 only | -0.260 [-0.400, -0.140]* | -0.174 [-0.261, -0.092]* |
| sliding | bm25 + glossary | -0.280 [-0.420, -0.140]* | -0.198 [-0.296, -0.109]* |
| sliding | hybrid (RRF) | -0.060 [-0.160, +0.040] | -0.040 [-0.098, +0.020] |
| sliding | hybrid RRF 2:1 | -0.040 [-0.120, +0.060] | -0.027 [-0.085, +0.030] |
| sliding | hybrid RRF 3:1 | -0.040 [-0.120, +0.060] | -0.023 [-0.078, +0.031] |
| sliding | hybrid + glossary | -0.140 [-0.260, -0.020]* | -0.091 [-0.166, -0.015]* |
| sliding | hybrid 2:1 + glossary | -0.080 [-0.180, +0.020] | -0.054 [-0.119, +0.013] |
| sliding | hybrid + rerank | -0.120 [-0.240, -0.020]* | -0.085 [-0.160, -0.019]* |
| sliding | hybrid + glossary + rerank | -0.120 [-0.240, -0.020]* | -0.082 [-0.154, -0.017]* |
| semantic | bm25 only | -0.180 [-0.320, -0.020]* | -0.135 [-0.229, -0.038]* |
| semantic | bm25 + glossary | -0.140 [-0.300, +0.000] | -0.117 [-0.219, -0.023]* |
| semantic | hybrid (RRF) | -0.020 [-0.140, +0.100] | -0.012 [-0.085, +0.068] |
| semantic | hybrid RRF 2:1 | +0.020 [-0.080, +0.120] | +0.012 [-0.050, +0.078] |
| semantic | hybrid RRF 3:1 | +0.000 [-0.080, +0.100] | +0.003 [-0.055, +0.067] |
| semantic | hybrid + glossary | +0.040 [-0.080, +0.160] | +0.022 [-0.047, +0.097] |
| semantic | hybrid 2:1 + glossary | -0.020 [-0.120, +0.100] | -0.008 [-0.073, +0.065] |
| semantic | hybrid + rerank | -0.060 [-0.200, +0.060] | -0.040 [-0.118, +0.033] |
| semantic | hybrid + glossary + rerank | -0.060 [-0.200, +0.060] | -0.040 [-0.118, +0.033] |

## Chunking strategies vs fixed (same configuration)

d = strategy minus fixed. Paired bootstrap on per-question differences, 95% CI; `*` = CI excludes 0. With this many comparisons, roughly 1 in 20 starred results can be chance.

| chunking | configuration | d hit@1 | d MRR |
|---|---|---|---|
| sliding | dense only | +0.060 [+0.000, +0.140] | +0.041 [+0.006, +0.085]* |
| sliding | bm25 only | +0.020 [-0.060, +0.100] | +0.033 [-0.021, +0.088] |
| sliding | bm25 + glossary | +0.000 [-0.100, +0.100] | +0.023 [-0.033, +0.079] |
| sliding | hybrid (RRF) | +0.040 [-0.060, +0.140] | +0.021 [-0.039, +0.085] |
| sliding | hybrid RRF 2:1 | +0.080 [-0.020, +0.180] | +0.040 [-0.018, +0.103] |
| sliding | hybrid RRF 3:1 | +0.080 [-0.020, +0.200] | +0.041 [-0.017, +0.103] |
| sliding | hybrid + glossary | +0.020 [-0.080, +0.120] | +0.001 [-0.052, +0.058] |
| sliding | hybrid 2:1 + glossary | +0.020 [-0.100, +0.140] | +0.006 [-0.057, +0.076] |
| sliding | hybrid + rerank | -0.040 [-0.120, +0.040] | -0.014 [-0.063, +0.035] |
| sliding | hybrid + glossary + rerank | -0.040 [-0.120, +0.040] | -0.012 [-0.061, +0.036] |
| semantic | dense only | -0.020 [-0.120, +0.080] | -0.009 [-0.071, +0.045] |
| semantic | bm25 only | +0.020 [-0.060, +0.100] | +0.022 [-0.031, +0.077] |
| semantic | bm25 + glossary | +0.060 [-0.040, +0.160] | +0.053 [-0.001, +0.111] |
| semantic | hybrid (RRF) | +0.000 [-0.120, +0.100] | -0.001 [-0.063, +0.061] |
| semantic | hybrid RRF 2:1 | +0.060 [-0.060, +0.180] | +0.028 [-0.037, +0.097] |
| semantic | hybrid RRF 3:1 | +0.040 [-0.080, +0.160] | +0.017 [-0.053, +0.090] |
| semantic | hybrid + glossary | +0.120 [+0.020, +0.220]* | +0.063 [+0.008, +0.125]* |
| semantic | hybrid 2:1 + glossary | +0.000 [-0.120, +0.120] | +0.002 [-0.068, +0.075] |
| semantic | hybrid + rerank | -0.060 [-0.160, +0.020] | -0.018 [-0.075, +0.037] |
| semantic | hybrid + glossary + rerank | -0.060 [-0.160, +0.020] | -0.020 [-0.075, +0.034] |

## Glossary expansion on vs off

d = variant minus reference; the two differ in only this one factor. Paired bootstrap on per-question differences, 95% CI; `*` = CI excludes 0. With this many comparisons, roughly 1 in 20 starred results can be chance.

| chunking | configuration | d hit@1 | d MRR |
|---|---|---|---|
| fixed | bm25 + glossary vs bm25 only | +0.000 [-0.100, +0.100] | -0.014 [-0.084, +0.049] |
| fixed | hybrid + glossary vs hybrid (RRF) | -0.060 [-0.140, +0.000] | -0.031 [-0.079, +0.009] |
| fixed | hybrid 2:1 + glossary vs hybrid RRF 2:1 | +0.020 [-0.040, +0.080] | +0.007 [-0.030, +0.040] |
| fixed | hybrid + glossary + rerank vs hybrid + rerank | +0.000 [+0.000, +0.000] | +0.002 [+0.000, +0.006] |
| sliding | bm25 + glossary vs bm25 only | -0.020 [-0.140, +0.100] | -0.024 [-0.091, +0.038] |
| sliding | hybrid + glossary vs hybrid (RRF) | -0.080 [-0.160, +0.000] | -0.051 [-0.104, +0.000] |
| sliding | hybrid 2:1 + glossary vs hybrid RRF 2:1 | -0.040 [-0.120, +0.040] | -0.027 [-0.070, +0.010] |
| sliding | hybrid + glossary + rerank vs hybrid + rerank | +0.000 [+0.000, +0.000] | +0.003 [+0.000, +0.009] |
| semantic | bm25 + glossary vs bm25 only | +0.040 [-0.060, +0.160] | +0.018 [-0.050, +0.085] |
| semantic | hybrid + glossary vs hybrid (RRF) | +0.060 [-0.020, +0.160] | +0.033 [-0.010, +0.083] |
| semantic | hybrid 2:1 + glossary vs hybrid RRF 2:1 | -0.040 [-0.100, +0.000] | -0.020 [-0.050, +0.003] |
| semantic | hybrid + glossary + rerank vs hybrid + rerank | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

## Weighted RRF vs equal-weight RRF

d = variant minus reference; the two differ in only this one factor. Paired bootstrap on per-question differences, 95% CI; `*` = CI excludes 0. With this many comparisons, roughly 1 in 20 starred results can be chance.

| chunking | configuration | d hit@1 | d MRR |
|---|---|---|---|
| fixed | hybrid RRF 2:1 vs hybrid (RRF) | -0.020 [-0.080, +0.040] | -0.006 [-0.046, +0.030] |
| fixed | hybrid RRF 3:1 vs hybrid (RRF) | -0.020 [-0.080, +0.040] | -0.002 [-0.045, +0.036] |
| sliding | hybrid RRF 2:1 vs hybrid (RRF) | +0.020 [+0.000, +0.060] | +0.013 [+0.000, +0.036] |
| sliding | hybrid RRF 3:1 vs hybrid (RRF) | +0.020 [+0.000, +0.060] | +0.018 [+0.000, +0.043] |
| semantic | hybrid RRF 2:1 vs hybrid (RRF) | +0.040 [-0.040, +0.120] | +0.023 [-0.017, +0.067] |
| semantic | hybrid RRF 3:1 vs hybrid (RRF) | +0.020 [-0.060, +0.100] | +0.015 [-0.030, +0.062] |

Seek error is the distance from where the top-1 link lands to the labelled moment, over the questions whose top-1 result was correct (so the sample differs per row; read its interval as wide).
