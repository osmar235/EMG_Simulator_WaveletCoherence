# Changelog

## v3.3.0 (2026-09-26) - corrected primary inference and preprocessing

1. **Primary coherence analysis now uses no full-record taper.**
   - `run_single_demo.py` and `run_factorial_v3.py` default to `--taper none`.
   - Matviyenko remains available only as a sensitivity analysis.
   - Band medians continue to exclude the cone of influence.
2. **Primary null inference changed from phase randomization to same-condition cross-trial pairing.**
   - Added `trial_shuffle_null_v3.py`.
   - The factorial runner saves untapered, preprocessed 1-kHz trial signals by condition unless `--no-save-null-signals` is specified.
   - The null pairs `L_i` with `R_j` for `i != j`, preserving task/burst structure while breaking within-trial bilateral pairing.
   - Random derangements generate condition-level null distributions, empirical one-sided p-values, and BH-FDR q-values.
   - Phase-randomization functions remain only as legacy diagnostics; the incorrect `IAAFT-lite` description was removed.
3. **Mixed ART promoted to the primary factorial model.**
   - `analysis_art_v3.R` now uses `(1 | trial_id)` in the primary model to account for repeated band values within each bilateral trial.
   - The old fixed-effects ART is retained as a manuscript-reproduction analysis.
4. **Restored neuron regression checks from V2.**
   - Single-call LIF update rule comparison.
   - 100-ms chunking with carried state equals continuous integration exactly.
   - Explicit check that resetting state every 100 ms is not the V3 path.
   - Population-firing smoke diagnostic for a 100-ms burst artifact.
5. **Documentation and audit metadata updated.**
   - Factorial rows now record `trial_uid`, `analysis_taper`, and `matviyenko_K`.
   - 75/25 is explicitly documented as an amplitude weighting; for independent equal-variance sources it is approximately 90/10 in variance contribution.
   - Removed stale statements that global-max phase-surrogate contours or Matviyenko K=8 are current defaults.
6. **Independent per-MU synaptic noise remains deliberately deferred** as a future model extension/sensitivity test.

## v3.2.0 (2026-09-26) - separate time-domain EMG from wavelet-only tapering

1. **Matviyenko taper is now strictly analysis-only.**
   - The physiological/reporting bipolar EMG is filtered, detrended, z-scored, saved, and plotted **without tapering**.
   - A separate analysis copy is created for CWT; Matviyenko (or another selected taper) is applied only to that copy immediately before wavelet analysis.
   - `single_demo_emg_bipolar_1k.csv` is therefore always untapered.
   - `single_demo_wavelet_input_1k.csv` records the actual CWT input for audit/debugging.
2. **Removed global-max surrogate contours from the default wavelet map.**
   - The K-sensitivity runs showed global-max 95% thresholds saturating near 1.0, making that display uninformative for dense time-frequency maps.
   - The default map now displays squared wavelet coherence and the cone of influence only, with no pixelwise significance claim.
   - Optional phase-randomized surrogates are retained as a **frequency-median diagnostic** and are saved to `single_demo_frequency_median_surrogate_threshold.csv`; these values are not drawn as pixelwise map contours.
3. **Validation expanded to verify branch separation.**
   - Added checks that taper preprocessing does not mutate the source EMG and that the untapered reporting signal remains distinct from the tapered CWT input.
4. **No simulator architecture change.**
   - LIF dynamics, common-drive normalization, SNR handling, duration/burn-in, MUAP synthesis, bipolar projection, and the decision to defer independent per-MU synaptic noise are unchanged.

## v3.1.0 (2026-09-26) - exact Matviyenko taper integration

1. **Ported the supplied MATLAB Matviyenko taper exactly into NumPy.**
   - Added `matviyenko(K, n)`, reproducing the MATLAB definition and coefficient table for integer `K=2..13`.
   - Added `matviyenko_window(length, K=8)` for direct signal preprocessing. The original formula returns `2*n` samples, so odd signal lengths are rejected rather than silently altered.
   - Preserved the supplied original source at `reference_matlab/matviyenko.m`.
2. **Coherence preprocessing now supports `taper='matviyenko'`.**
   - `taper_kwargs={'K': ...}` selects the order.
   - `run_single_demo.py` and `run_factorial_v3.py` now use Matviyenko by default.
3. **K is explicit in metadata and CLI.**
   - The supplied MATLAB function defines K as a free parameter but does not contain the historical call site; therefore the exact K used in the original published analysis cannot be inferred from that file alone.
   - `K=8` is used as a transparent, reproducible **V3 preset**, not as a claim about the historical analysis. It can be changed with `--matviyenko-k`.
4. **Validation expanded to 21 checks.**
   - Added a literal numerical reference test from the supplied MATLAB equations, symmetry/length checks, all supported K values, odd-length rejection, and deterministic Matviyenko preprocessing.

## v3.0.0 (2026-09-26) - scientific/reproducibility revision

Built from v2.0.0. Version 1 and Version 2 are preserved under `legacy/`.

### Scientific fixes

1. **Common-drive strength is fixed across source mixtures.**
   - V2 directly used coefficient sums `6+0`, `4.875+1.625`, and `3.25+3.25`.
   - Because cortical and subcortical signals are independent standardized processes, their combined RMS was approximately 6.00, 5.14, and 4.60 respectively.
   - V3 infers the intended cortical fraction from those coefficient pairs, composes the mixture, and RMS-normalizes the *combined* common drive to `common_drive_target_rms=6.0` by default.
   - `common_drive_normalization='legacy_coefficients'` remains available for diagnostic comparison.

2. **SNR is explicit and defaults to dB.**
   - V1/V2 scaled noise as `signal_sd/sqrt(SNR)`, which interprets the number as a linear power ratio.
   - V3 default `snr_units='db'` uses `signal_sd/10**(SNR_dB/20)`.
   - `snr_units='linear_power'` preserves the old interpretation when required for comparison.

3. **`T_end` now means usable returned duration.**
   - V1/V2 simulated `T_end` then trimmed the beginning, so `T_end=18` returned about 17 s.
   - V3 simulates an explicit `burn_in_s` before time zero and returns exactly `T_end` seconds after burn-in.

4. **Intent variability is bounded instead of secular.**
   - V2's `intent_drift='trial'` implemented `1 + Intent_variability*t`, so 0.01 became ~18% change over 18 s.
   - V3 default `intent_drift='bounded'` uses smooth low-frequency variation whose RMS is approximately `Intent_variability`.
   - Legacy `linear_per_second`, `window_legacy`, and `none` modes remain available.

5. **Coherence preprocessing is explicit.**
   - Added `preprocess_emg_for_coherence()` for band-pass filtering, linear detrending, optional tapering, and z-scoring.
   - At the time of v3.0.0, the exact historical Matviyenko taper implementation was absent from the supplied v1/v2 archives, so no substitute was mislabeled as Matviyenko. **Resolved in v3.1.0 when the MATLAB source was supplied.**

6. **Surrogate inference now matches the statistic being displayed.**
   - Added `phase_randomize()` which preserves the exact Fourier periodogram.
   - Added `surrogate_wavelet_threshold()` with:
     - `global_max` for a map-wide maximum-statistic threshold,
     - `pointwise_frequency` for descriptive per-frequency pointwise thresholds,
     - `frequency_median` for inference on time-median coherence.
   - The v2 Gaussian `mc_null_threshold()` is retained only for backward compatibility.

### Reproducibility / workflow

7. `pycwt` is now optional at import time; generator-only workflows run without it.
8. Generator `info` metadata reports actual common-drive RMS, cortical fraction, burn-in/usable duration, SNR units, and measured drive-noise SNR.
9. Added `validate_v3.py` with deterministic numerical checks.
10. Added `run_factorial_v3.py` with deterministic cell/trial seeds and checkpointed long-format output.
11. Added `configs/paper_factorial_v3.json` with the complete planned corrected factorial configuration.
12. Added a V3 metadata JSON output to the single-trial demo.

### Deliberately not changed by default

- All MUs within a limb still receive the same limb-level synaptic current apart from recruitment threshold differences. Per-MU independent synaptic noise remains a model extension, not a silent correction to the published architecture.
- The spatial bipolar projection remains a post-summation global fractional delay rather than an MU-specific electrode/conduction model.
- No exact Matviyenko taper is claimed until the historical implementation is recovered.

## v2.0.0 (2026-09) - major bug-fix release

See `legacy/v2/README_V2.md` and `legacy/v2/functions_v2.py`. V2 fixed the wavelet scale-smoothing axis, preserved the complex cross-spectrum, eliminated the 100-ms LIF resets/input discontinuities, fixed MUAP edge truncation and seed handling, and made the generator reproducible.

## v1 - published legacy implementation

Preserved in `legacy/functions_v1.py` and `legacy/run_single_demo_v1.py` for historical comparison.
