# EMG Simulation Code Review and Version History Memory Log

**Project:** Bilateral FDI EMG simulation and EMG-EMG wavelet coherence framework  
**Associated manuscript:** *EMG Simulation and Coherence Analysis: A Computational Framework for EMG Simulation and Coherence-Based Biomarker Analysis on Neural Crosstalk in Bimanual Coordination*  
**Review / revision date:** September 26, 2026  
**Purpose of this log:** Preserve the technical decisions made during the V1/V2 code review and the construction of Version 3 so the rationale is not lost between future analyses.

---

## 1. Starting point

We reviewed two code archives, referred to here as **Version 1 (V1)** and **Version 2 (V2)**, against the simulation and coherence-analysis pipeline described in the associated manuscript.

The manuscript-level target model contains:

- bilateral FDI EMG simulation;
- 72 motor units per limb at the main setting;
- leaky integrate-and-fire motor-unit dynamics with spike-frequency adaptation;
- 1:2 force envelopes (0.5 Hz and 1.0 Hz);
- cortical common drive with an upper cutoff of either 30 or 60 Hz;
- subcortical common drive in approximately 5-13 Hz;
- asymmetric cross-limb mixing (R->L varied 0-35%; L->R fixed at 1%);
- MUAP synthesis with fiber-level variability;
- bipolar spatial filtering;
- non-rectified EMG for wavelet coherence;
- Morlet CWT and EMG-EMG coherence summaries in 5-13, 13-30, 30-60, and 60-100 Hz bands;
- 8 x 3 x 2 x 15 factorial design = 720 bilateral trials = 1,440 simulated EMG signals.

---

## 2. Major problems identified in Version 1

### 2.1 Wavelet scale smoothing was applied along the wrong axis

V1 effectively applied both smoothing operations along time. The second operation should smooth across wavelet scales.

**Consequence:** the wavelet-coherence estimator did not implement the intended time-plus-scale smoothing.

**Decision:** retain the V2 correction, in which the scale kernel is oriented across the scale dimension.

### 2.2 Complex cross-wavelet information was discarded

V1 converted the cross-wavelet quantity to its real component during smoothing.

**Consequence:** phase-bearing imaginary information was removed before the magnitude-squared coherence calculation.

**Decision:** V2/V3 preserve the complex cross-spectrum through smoothing and take magnitude only at the appropriate final coherence step.

### 2.3 LIF state was reset every 100 ms

The V1 generator processed 100-ms windows independently. Adaptation and refractory state were recreated at each window, and the right-side membrane voltage was also reset across windows.

**Why this matters:** the model describes adaptation with a time constant around 250 ms, which cannot be represented correctly if adaptation state is reset every 100 ms. A 100-ms artificial segmentation also corresponds to a 10-Hz periodic structure, which falls inside the 5-13 Hz band used in the paper.

**Decision:** V2/V3 simulate LIF population state continuously across the full trial.

### 2.4 MUAP activity could be truncated at 100-ms window boundaries

Spikes close to a window edge could lose part of their MUAP waveform.

**Decision:** V2/V3 perform full-trial spike-to-MUAP convolution so the waveform is not truncated by artificial simulation windows.

### 2.5 The seed did not fully determine the generated signal

V1 mixed seeded and unseeded random-number generation, including new RNG instances and global NumPy random calls.

**Consequence:** the same nominal seed did not reliably reproduce identical EMG signals.

**Decision:** V2/V3 derive stochastic components from controlled seeded RNGs. V3 validation explicitly checks bitwise reproducibility.

### 2.6 Some motor-unit gain arguments were not actually used

V1 exposed parameters that were effectively ignored in the generator.

**Decision:** V2 corrected the relevant parameter usage and V3 retains those fixes.

---

## 3. Important problems that remained after Version 2

Version 2 was judged clearly preferable to Version 1, but several scientific/reproducibility issues remained.

### 3.1 Common-drive magnitude was not fixed across 100/0, 75/25, and 50/50 mixtures

The manuscript states that total common drive is fixed while the cortical/subcortical composition changes. The stated coefficient pairs were:

- 100/0: cortical 6.0, subcortical 0.0;
- 75/25: cortical 4.875, subcortical 1.625;
- 50/50: cortical 3.25, subcortical 3.25.

Even the coefficient sums differ (6.0 versus 6.5). More importantly, for independent standardized cortical and subcortical processes, the standard deviation/RMS of a mixture is approximately proportional to:

`sqrt(a^2 + b^2)`

rather than `a + b`.

The legacy combinations therefore produced approximate common-drive RMS values:

- 100/0: 6.00;
- 75/25: about 5.14;
- 50/50: about 4.60.

This created a potential confound between **spectral/source composition** and **total shared-input magnitude**.

**V3 decision:** use the legacy coefficient pairs to infer the intended cortical fraction, construct the cortical/subcortical mixture, and then RMS-normalize the **combined common drive** to a common target (`common_drive_target_rms=6.0`).

The corrected default therefore gives:

| Mixture | Cortical fraction | Combined common-drive RMS |
|---|---:|---:|
| 100/0 | 1.00 | 6.0 |
| 75/25 | 0.75 | 6.0 |
| 50/50 | 0.50 | 6.0 |

The default semantic interpretation of 75/25 is an **amplitude ratio** because 4.875:1.625 = 3:1. A power-ratio interpretation remains scientifically possible but must be treated as a different model and rerun consistently.

### 3.2 SNR values were labeled as dB but treated as linear power ratios

Legacy scaling effectively used:

`noise_sd = signal_sd / sqrt(SNR)`

This treats `SNR` as a linear power ratio. For example, `SNR=40` corresponds to about 16 dB rather than 40 dB.

**V3 decision:** the corrected default treats the numeric parameter as dB:

`noise_sd = signal_sd / 10**(SNR_dB/20)`

A legacy linear-power option is retained only for explicit backward comparisons.

### 3.3 Requested 18-s simulations returned only about 17 s

Legacy versions simulated 18 s and then trimmed approximately 1 s from the beginning.

**V3 decision:** simulate an explicit burn-in period in addition to the requested duration. `T_end=18` now returns exactly 18.000 s of analyzable output after burn-in.

### 3.4 Intent variability was implemented as an unbounded trial ramp

In V2, an `Intent_variability` of 0.01 could behave approximately like `1 + 0.01*t`, yielding around an 18% change over an 18-s trial.

**V3 decision:** the default is a smooth, bounded low-frequency stochastic variation with RMS approximately equal to the requested fractional variability. Thus `0.01` means approximately 1% RMS variability rather than 1% growth per second.

Legacy modes remain available for comparison.

### 3.5 The manuscript's detrending step needed to be made explicit

**V3 decision:** `preprocess_emg_for_coherence()` explicitly performs the analysis preprocessing, including band-pass filtering and linear detrending before tapering/coherence.

### 3.6 The surrogate statistic used for a time-frequency map did not match the map-level inference

A threshold based on the 95th percentile of a frequency-specific **time-median** coherence statistic had been used to outline individual time-frequency pixels.

**Problem:** inference on `median_t(R^2(f,t))` is not the same as pointwise or family-wise inference on `R^2(f,t)`.

**V3 decision:**

- phase-randomized surrogates preserve each signal's Fourier magnitude/power spectrum;
- a **global maximum statistic** was evaluated in early V3 work, but this was superseded in V3.2/V3.3 because it saturated near 1.0 and is not a current default;
- separate `frequency_median` and `pointwise_frequency` modes remain available when those are the intended statistics.

### 3.7 Cone-of-influence handling

V3 band-median summaries can exclude time-frequency samples outside the CWT cone of influence, rather than allowing edge-contaminated values into the frequency-band summary.

---

## 4. Version 3.0 implementation decisions

Version 3.0 was built from Version 2, not Version 1.

### Corrected baseline defaults

- continuous full-trial LIF state;
- corrected wavelet scale smoothing;
- preserved complex cross-spectrum;
- deterministic seeded RNG;
- full-trial MUAP convolution;
- combined common-drive RMS fixed to 6.0 across source mixtures;
- amplitude-fraction interpretation for the published 100/0, 75/25, 50/50 labels;
- SNR interpreted in dB;
- explicit burn-in and exact returned analyzable duration;
- bounded low-frequency intent variability;
- explicit band-pass + detrend coherence preprocessing;
- legacy phase-randomized surrogate diagnostics (superseded for primary inference in V3.3);
- historical global-max map threshold option (not a V3.3 default);
- cone-of-influence-aware band summaries;
- pycwt optional at import time so simulation-only workflows can run without the wavelet dependency.

### Reproducibility additions

V3 added:

- `validate_v3.py`;
- `run_single_demo.py`;
- `run_factorial_v3.py`;
- `configs/paper_factorial_v3.json`;
- deterministic trial/cell seed generation;
- checkpointed factorial CSV output;
- `analysis_art_v3.R` for the factorial ART analysis; as of V3.3 the repeated-trial mixed ART is primary and fixed ART is manuscript-reproduction only;
- `MODEL_DECISIONS_V3.md`;
- `CHANGELOG.md`;
- legacy V1 and V2 code retained under `legacy/`.

---

## 5. Version 3.1 - Matviyenko taper integration

On September 26, 2026 the historical MATLAB function `matviyenko.m` was supplied.

### What the MATLAB source establishes

The function:

`matviyenko(K, n)`

uses a fixed coefficient table for integer `K=2..13` and returns a symmetric window with exactly `2*n` samples. The comments state that increasing K makes the window steeper and lowers Fourier sidelobes.

### What was added to Python

`functions.py` now contains:

- `matviyenko(K, n)` - direct NumPy port of the supplied MATLAB equations and coefficient table;
- `matviyenko_window(length, K=8)` - wrapper for direct application to an even-length signal;
- `preprocess_emg_for_coherence(..., taper='matviyenko', taper_kwargs={'K': ...})`.

The original MATLAB file is preserved unchanged at:

`reference_matlab/matviyenko.m`

### Exact even-length behavior is preserved

The MATLAB algorithm is intrinsically defined as a `2*n`-sample window. The Python wrapper therefore **rejects odd lengths** rather than silently padding, truncating, or modifying the historical formula.

The main paper-like coherence record is 18 s at 1000 Hz = 18,000 samples, so the exact function applies without modification.

### K remains historically unresolved

The supplied file contains the **function definition**, but not the historical script line that called it. Therefore the file cannot reveal which K was used in the original analysis.

**Decision:**

- make K explicit in all V3.1 CLI/config/metadata;
- use **K=8 only as a reproducible V3 preset**;
- do **not** state that K=8 was necessarily the value used in the published paper;
- if the old caller is later recovered, substitute that K and rerun the corrected factorial grid.

### Validation of the port

Version 3.1 validation currently reports **21/21 checks passed**.

Matviyenko-specific checks include:

- literal numerical reference values from the supplied MATLAB equations for K=8, n=8;
- symmetry and exact `2*n` length;
- valid operation for every K=2..13;
- deliberate rejection of odd lengths;
- deterministic preprocessing using the Matviyenko taper.

---

## 6. Deliberately deferred future model test: independent per-MU synaptic noise

**Do not add independent per-MU synaptic noise to the V3 default.**

Current baseline motor units within one limb receive the same limb-level temporal input, with recruitment differences generated by motor-unit properties/thresholds. Adding an independent synaptic-noise component to each motor unit *before LIF spike generation* could reduce artificial within-pool synchronization and alter absolute coherence.

However, this is a genuine **model architecture extension**, not a repair of an implementation bug.

### Future test plan

Treat per-MU independent synaptic noise as a separate sensitivity/model-extension analysis, for example:

`I_i(t) = I_common(t) + I_unique,i(t)`

with the common versus independent fractions controlled explicitly.

Compare the extension against the corrected V3 baseline for:

- mean discharge rate;
- recruitment profile;
- within-limb spike synchronization;
- EMG PSD;
- absolute intermuscular coherence;
- crosstalk slope;
- beta versus low-gamma source-mixture effects;
- robustness of the paper's qualitative conclusions.

Do not silently mix these results into the baseline V3 rerun.

---

## 7. Other architectural extension deliberately deferred

The bipolar spatial projection remains a global post-summation fractional-delay model. A future muscle/electrode model may project each MU separately using MU-specific depth, conduction velocity, and electrode geometry, but that also changes the architecture and should be a separate extension.

---

## 8. Current V3.2 factorial rerun specification

Corrected baseline:

- 8 R->L crosstalk values: 0, 5, 10, 15, 20, 25, 30, 35%;
- L->R = 1%;
- common-drive mixtures: 100/0, 75/25, 50/50;
- combined common-drive RMS = 6.0 in every mixture;
- cortical upper cutoff = 30 or 60 Hz;
- 15 trials per factorial cell;
- deterministic base seed = 20260926 by default;
- 72 MUs per limb;
- 346 fibers per MU;
- `T_end=18` analyzable seconds plus 1-s burn-in;
- drive SNR = 40 dB;
- MU-level SNR = 8 dB;
- EMG-level SNR = 40 dB;
- bipolar delay about 1.2 ms;
- bipolar low-pass about 180 Hz in the default rerun;
- resample to 1000 Hz;
- non-rectified 5-499 Hz EMG;
- linear detrend;
- Matviyenko taper, K explicit, V3 preset K=8;
- Morlet CWT, w0=6, dj=1/8, fmax=128 Hz;
- coherence summaries: 5-13, 13-30, 30-60, 60-100 Hz;
- cone-of-influence-aware band medians.

---

## 9. What must be compared against the published results

After the corrected full factorial run, compare V3.2 to the published/legacy results for:

1. monotonic coherence increase with modeled crosstalk;
2. crosstalk thresholds that become distinguishable from baseline;
3. 100/0 versus 75/25 versus 50/50 ordering;
4. beta discrimination under 13-30 Hz cortical drive;
5. low-gamma discrimination under 13-60 Hz cortical drive;
6. alpha-band behavior;
7. high-gamma behavior;
8. all major ART main effects and interactions;
9. effect sizes and post-hoc contrast directions;
10. absolute coherence magnitude versus the legacy implementation.

Do not decide whether a journal correction is needed until these corrected outputs are compared with the published statistical conclusions.

---

## 10. Windows execution commands for the current machine

Existing virtual environment:

`C:\Users\osmar\projects\.venv`

Assume V3.2 is extracted to:

`C:\Users\osmar\projects\Version3_EMGSim`

### Direct use of the venv Python - safest option

```bat
cd /d C:\Users\osmar\projects\Version3_EMGSim
C:\Users\osmar\projects\.venv\Scripts\python.exe --version
C:\Users\osmar\projects\.venv\Scripts\python.exe -c "import numpy, scipy, pandas, matplotlib, pycwt; print('Python dependencies OK')"
C:\Users\osmar\projects\.venv\Scripts\python.exe validate_v3.py
C:\Users\osmar\projects\.venv\Scripts\python.exe run_single_demo.py --taper matviyenko --matviyenko-k 8 --surrogates 100
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --dry-run
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 1 --taper matviyenko --matviyenko-k 8 --outdir results\factorial_v3_smoke
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 15 --base-seed 20260926 --taper none --outdir results\factorial_v3_3_primary
```

### Optional CMD activation

```bat
C:\Users\osmar\projects\.venv\Scripts\activate.bat
cd /d C:\Users\osmar\projects\Version3_EMGSim
python validate_v3.py
```

### Optional PowerShell activation

```powershell
& C:\Users\osmar\projects\.venv\Scripts\Activate.ps1
Set-Location C:\Users\osmar\projects\Version3_EMGSim
python validate_v3.py
```

### ART analysis after the full CSV is complete

If R/Rscript and the required R packages are on PATH:

```bat
Rscript analysis_art_v3.R results\factorial_v3_K8\factorial_v3_band_medians.csv
```

---

## 11. Current artifact status

**Use for future work:** Version 3.3.0  
**Keep for historical comparison only:** Version 1 and Version 2 under `legacy/`  
**Matviyenko source of truth:** `reference_matlab/matviyenko.m`  
**Scientific decisions:** `MODEL_DECISIONS_V3.md`  
**Software changes:** `CHANGELOG.md`  
**Numerical checks:** `results/v3_validation_report.txt`

This log should be updated whenever the corrected factorial rerun changes a scientific decision, whenever the historical Matviyenko K is recovered, or when a future model extension (especially independent per-MU synaptic noise) is formally tested.


---

## 12. V3.2 correction after representative K-sensitivity tests

The representative runs with no taper and Matviyenko K=2, K=8, and K=13 exposed a reporting-path mistake in V3.1: `run_single_demo.py` was saving and plotting the already-tapered wavelet-analysis copy as if it were the bipolar EMG. This produced artificial amplitude falloff toward the record edges in the time-domain figure.

Decision: the Matviyenko window is used **only immediately before wavelet/CWT analysis**. It must never modify the simulated/exported/plotted bipolar EMG. V3.2 therefore separates the branches:

1. bipolar EMG -> resample -> 5-499 Hz filter -> detrend -> z-score -> save/plot as the time-domain EMG;
2. a separate analysis copy -> selected taper (Matviyenko K explicit) -> CWT -> coherence.

V3.2 saves the two branches separately as `single_demo_emg_bipolar_1k.csv` (untapered) and `single_demo_wavelet_input_1k.csv` (actual CWT input).

The same K-sensitivity runs showed global-max 95% phase-surrogate thresholds near the theoretical coherence ceiling (~0.985-1.000). This method is therefore no longer the default map-level display. V3.2 wavelet figures show the coherence field and cone of influence without a pixelwise significance contour. Optional surrogates use a frequency-median diagnostic threshold and are saved separately, not overlaid on the map.

No generator/model architecture was changed by this V3.2 correction. In particular, independent per-MU synaptic noise remains deliberately deferred for future tests.

**Current baseline for future work: Version 3.3.0.**


---

## 8. Version 3.3 - corrected primary inference and preprocessing (September 26, 2026)

V3.3 supersedes the V3.2 analysis defaults while preserving the simulator fixes.

### Primary taper decision
The primary corrected coherence analysis uses **no full-record taper**. The exact Matviyenko MATLAB implementation remains available as a sensitivity analysis only. This decision follows the K-sensitivity runs showing that the supplied Matviyenko function weights the full record rather than only the edges, and stronger K values measurably increase lower-frequency band medians. Band summaries continue to exclude samples outside the cone of influence.

### Primary null/significance decision
Phase randomization is no longer used for primary inference because it preserves the periodogram but destroys the task-locked burst/nonstationary structure of the simulated EMG. Zero-coupling tests showed this can miscalibrate band-specific null levels, especially 5-13 Hz.

V3.3 uses **same-condition cross-trial pairing/derangements**:

- left trial `i` is paired with right trial `j`, with `i != j`;
- task timing, burst structure, spectra, duration, and preprocessing are preserved;
- within-trial bilateral common input/crosstalk is broken;
- random derangements create a null distribution for the condition-level median coherence statistic.

`trial_shuffle_null_v3.py` implements this analysis. Phase-randomization code is retained only as a legacy diagnostic and is labeled accordingly. The old `IAAFT-lite` description was corrected because the implementation is plain Fourier phase randomization.

### Statistical model decision
The four frequency-band values from a bilateral trial are correlated repeated observations. V3.3 therefore promotes the mixed ART model with random trial intercept to primary:

`coherence_median ~ crosstalk * mixture * cortical_high * band + (1 | trial_id)`

The original fixed-effects ART is retained as a manuscript-reproduction comparison.

### Restored validation tests
V3.3 ports back the V2 neuron regression checks:

- the continuous single-call LIF rule matches the legacy local update rule;
- 100-ms chunks with carried state are exactly identical to continuous integration;
- resetting state every 100 ms is explicitly shown to produce a different result and is not the V3 path;
- a population-firing smoke diagnostic checks for a conspicuous 100-ms burst artifact.

The V3.3 validation suite passes **30/30 core checks** in the build environment. If `pycwt` is installed, an additional runtime check verifies that cached-CWT cross-trial pairing produces the same coherence as direct computation.

### Mixture interpretation clarification
The labels 100/0, 75/25, and 50/50 are amplitude weightings. For independent equal-variance sources, 75/25 amplitude corresponds to approximately **90/10 variance contribution**. This should be stated explicitly in revised Methods and should not be described as 75/25 power.

### Factorial audit metadata
V3.3 factorial rows now record `trial_uid`, `analysis_taper`, and `matviyenko_K`. The runner also saves per-condition untapered 1-kHz analysis signals by default so the matched cross-trial null can be calculated without rerunning the simulator.

### Deferred model extension remains unchanged
**Independent per-MU synaptic noise remains excluded from the V3.3 baseline.** It is reserved for a future model-extension/sensitivity study.

### Current baseline
**Use for future corrected work: Version 3.3.0.**
Primary taper: **none**.
Primary null: **same-condition cross-trial derangement**.
Primary factorial model: **mixed ART with `(1|trial_id)`**.
Matviyenko: **sensitivity only**.
Independent per-MU synaptic noise: **future test only**.
