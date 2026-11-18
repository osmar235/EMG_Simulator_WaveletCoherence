# Uploaded V3.2 full-run result note

The uploaded CSV is a complete V3.2 factorial output: **2,880 band rows = 720 bilateral trials x 4 bands** (15 trials/cell, 8 crosstalk levels, 3 mixtures, 2 cortical bandwidths).

Important audit limitation: this V3.2 CSV does **not** contain an `analysis_taper` or `matviyenko_K` column, so the exact taper setting used for this completed run cannot be recovered from the CSV alone. V3.3 adds those columns.

## Descriptive findings

Mean coherence rises monotonically with R->L crosstalk:

| R->L crosstalk | Mean coherence |
|---:|---:|
| 0.00 | 0.3483 |
| 0.05 | 0.3582 |
| 0.10 | 0.3651 |
| 0.15 | 0.3740 |
| 0.20 | 0.3876 |
| 0.25 | 0.3998 |
| 0.30 | 0.4127 |
| 0.35 | 0.4310 |

Across all conditions, mean coherence by mixture was:

- `50_50`: 0.3929
- `75_25`: 0.3823
- `100_0`: 0.3785

This overall ordering is **50/50 > 75/25 > 100/0**, which differs from the published fixed-common-drive interpretation and is one reason not to treat the V3.2 run as the final corrected result before V3.3 primary preprocessing/statistics are applied.

Across all conditions, mean coherence by band was:
- `5-13`: 0.4465
- `30-60`: 0.3728
- `13-30`: 0.3622
- `60-100`: 0.3568

Selected bandwidth x band x mixture means:

| Cortical high | Band | 100/0 | 75/25 | 50/50 |
|---:|---|---:|---:|---:|
| 30 | 13-30 | 0.3834 | 0.3796 | 0.3620 |
| 30 | 30-60 | 0.3665 | 0.3729 | 0.3918 |
| 30 | 5-13 | 0.4298 | 0.4384 | 0.4687 |
| 30 | 60-100 | 0.3737 | 0.3734 | 0.3673 |
| 60 | 13-30 | 0.3507 | 0.3502 | 0.3470 |
| 60 | 30-60 | 0.3547 | 0.3620 | 0.3890 |
| 60 | 5-13 | 0.4301 | 0.4422 | 0.4701 |
| 60 | 60-100 | 0.3392 | 0.3400 | 0.3472 |

These are descriptive summaries only. V3.3 changes the primary analysis to no taper, same-condition cross-trial nulls, and mixed ART with a random trial intercept. The completed V3.2 CSV remains useful as an audit/comparison dataset.
