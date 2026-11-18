# EMG Simulator - Wavelet Coherence, Version 3.3.0

Version 3.3 is the corrected analysis baseline built from the V2 implementation and the V3 scientific/reproducibility fixes. V1 and V2 remain under `legacy/` for audit and comparison.

## V3.3 primary decisions

1. **Fixed common-drive magnitude.** The cortical/subcortical mixture is composed first, then the combined common drive is RMS-normalized to 6.0 in every 100/0, 75/25, and 50/50 condition. This removes the old 6.0 -> 5.14 -> 4.60 common-drive-strength confound.
2. **Mixture labels are amplitude weights.** The historical 4.875:1.625 coefficients are a 3:1 amplitude ratio. For independent equal-variance sources, 75/25 amplitude corresponds to approximately 90/10 variance contribution. Do not describe 75/25 as 75/25 power.
3. **SNR is explicit in dB.** A requested value of 40 means 40 dB.
4. **Exact usable duration.** `T_end=18` returns 18 s after a separate burn-in.
5. **Continuous LIF state.** Membrane, refractory, and adaptation state do not reset every 100 ms.
6. **Bounded intent variability.** `Intent_variability=0.01` is approximately 1% RMS low-frequency variation, not 1% growth per second.
7. **Primary coherence preprocessing uses NO full-record taper.** The 5-499 Hz non-rectified bipolar EMG is detrended and z-scored; wavelet band summaries exclude the cone of influence (COI).
8. **Matviyenko remains available only as a sensitivity analysis.** The exact user-supplied MATLAB implementation is preserved in `reference_matlab/matviyenko.m` and ported exactly. Historical K is unknown.
9. **Primary null inference uses same-condition cross-trial pairing/derangements.** Phase randomization is retained only as a legacy diagnostic because it destroys the task-locked burst structure.
10. **Primary factorial statistics use mixed ART** with `(1 | trial_id)` because the four frequency-band values from one bilateral trial are repeated/correlated observations. The fixed-effects ART is retained as a manuscript-reproduction analysis.
11. **Independent per-MU synaptic noise remains deferred.** It is a future model extension/sensitivity study, not part of the corrected baseline.

## Files

- `functions.py` - V3.3 simulator, preprocessing, wavelet coherence, Matviyenko port, cached CWT helpers, derangement helper
- `run_single_demo.py` - one representative simulation; V3.3 primary defaults to no taper
- `run_factorial_v3.py` - full 720-trial factorial runner; defaults to no taper and saves per-cell signals for matched null analysis
- `trial_shuffle_null_v3.py` - same-condition cross-trial/derangement null analysis
- `analysis_art_v3.R` - primary mixed ART plus secondary fixed-effects manuscript-reproduction ART
- `validate_v3.py` - V3.3 validation suite, including restored V2 LIF continuity/window-reset checks
- `MODEL_DECISIONS_V3.md` - scientific decisions and rationale
- `EMG_SIMULATION_VERSION_HISTORY_MEMORY_LOG.md` - cumulative V1 -> V3.3 decision log
- `WINDOWS_TEST_COMMANDS.md` - Windows CMD commands for the user's venv
- `V3_2_FULL_RUN_RESULT_NOTE.md` - descriptive note on the uploaded completed V3.2 factorial CSV
- `reference_matlab/matviyenko.m` - supplied MATLAB source

## Validate first

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe validate_v3.py
```

Expected V3.3 result:

```text
RESULT: 30/30 checks passed
```

If `pycwt` is installed, the validation script also checks that cached-CWT pair coherence exactly matches direct coherence.

## Representative V3.3 demo

Primary analysis: no taper, no phase-surrogate inference.

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_single_demo.py --taper none --surrogates 0 --outdir results\demo_v3_3_primary
```

The time-domain export/plot is always untapered. `single_demo_wavelet_input_1k.csv` records the actual CWT input.

Matviyenko is sensitivity only:

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_single_demo.py --taper matviyenko --matviyenko-k 2 --surrogates 0 --outdir results\demo_sensitivity_K2
```

## Full corrected factorial

Dry run:

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --dry-run
```

Smoke test:

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 1 --taper none --outdir results\factorial_v3_3_smoke
```

Primary full run:

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 15 --base-seed 20260926 --taper none --outdir results\factorial_v3_3_primary
```

The primary CSV is:

```text
results\factorial_v3_3_primary\factorial_v3_3_band_medians.csv
```

The runner also saves 48 compressed per-condition signal files under:

```text
results\factorial_v3_3_primary\null_signals\
```

These are 1-kHz, 5-499 Hz, detrended, z-scored, untapered trial signals used by the matched cross-trial null.

## Same-condition cross-trial null

Smoke test on one condition:

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe trial_shuffle_null_v3.py results\factorial_v3_3_primary --permutations 99 --max-cells 1 --taper none
```

Full null analysis:

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe trial_shuffle_null_v3.py results\factorial_v3_3_primary --permutations 999 --seed 20260926 --taper none
```

For each factorial condition, the script pairs `L_i` with `R_j` for `i != j`, caches all trial CWTs, forms all off-diagonal band-coherence values, and then samples random derangements of trial labels. It exports the observed condition median, null median, null 95th percentile, observed-minus-null value, empirical one-sided p, and BH-FDR q.

This null analysis is intended to interpret coherence above the matched task/burst floor. It does **not** replace the factorial ART model.

## ART analysis

```bat
Rscript analysis_art_v3.R results\factorial_v3_3_primary\factorial_v3_3_band_medians.csv
```

Primary model:

```r
coherence_median ~ crosstalk * mixture * cortical_high * band + (1 | trial_id)
```

A fixed-effects ART is also generated only for comparison with the published analysis.

## Matviyenko sensitivity

The supplied MATLAB function is an optimized symmetric full-record window, not a short edge taper. It changes the entire record, especially for larger K. Therefore V3.3 does not apply it in the primary analysis. It can be tested separately:

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 3 --base-seed 20260926 --taper matviyenko --matviyenko-k 8 --outdir results\factorial_sensitivity_K8
```

The same base seed makes taper-sensitivity comparisons paired at the simulation level.

## Deferred future model extension

Independent per-motor-unit synaptic noise before LIF spike generation may reduce artificial within-pool synchronization, but it changes the model architecture. It is deliberately **not** part of V3.3 and should be tested in a future study/sensitivity analysis.
