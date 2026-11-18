# EMG Simulation Model Decisions - V3.3

## Corrected baseline

### Common drive
The old coefficient pairs produced different total common-drive RMS values across mixtures. V3.3 composes cortical and subcortical inputs first and normalizes the combined common drive to RMS = 6.0 for every condition. This separates source composition from common-drive magnitude.

The historical labels 100/0, 75/25, and 50/50 are interpreted as **amplitude weights**. Under independence and equal source variance, 75/25 amplitude corresponds to approximately 90/10 variance contribution. The manuscript/correction should not describe these labels as power fractions.

### SNR
All SNR parameters are explicit in dB. Noise amplitude is scaled as `signal_sd / 10**(snr_db/20)`.

### Duration and LIF state
`T_end` is the usable returned duration after a separate burn-in. LIF membrane, refractory, and adaptation state remain continuous; the historical 100-ms reset artifact is not present.

### Intent variability
The default is bounded low-frequency stochastic variation with RMS approximately equal to the requested fraction. `0.01` means approximately 1% RMS.

## Coherence-analysis decisions

### Primary taper decision: no full-record taper
The primary V3.3 coherence analysis uses no full-record taper. Signals are band-pass filtered 5-499 Hz, linearly detrended, z-scored, and analyzed with the Morlet CWT. Band summaries exclude the cone of influence.

The exact supplied Matviyenko MATLAB function is preserved and ported exactly, but it constructs a symmetric full-record window whose rising half occupies half the record. Historical K is unknown. It is therefore retained as a **sensitivity analysis**, not used as the primary corrected preprocessing choice.

Any optional taper is analysis-only and is applied to a copy immediately before CWT. It never modifies the simulated/exported/plotted bipolar EMG.

### Null/significance decision: matched cross-trial null
Phase randomization preserves the periodogram but destroys the force-envelope/burst structure of the simulated EMG. Tests with zero neural coupling showed that this can miscalibrate band-specific null levels, especially 5-13 Hz.

V3.3 primary null inference therefore uses same-condition cross-trial pairing:

`left trial i` vs `right trial j`, with `i != j`.

This preserves task timing, burst/nonstationary structure, marginal spectra, duration, and preprocessing while breaking within-trial bilateral pairing. Random derangements of the trial labels generate a null distribution for the condition-level median coherence statistic.

Phase-randomized functions remain in `functions.py` only for historical/diagnostic comparisons and are explicitly labeled legacy.

### Alpha / 5-13 Hz interpretation
Absolute coherence in 5-13 Hz can have a high matched-null floor. V3.3 therefore interprets alpha-band effects relative to the same-condition cross-trial null and factorial within-band contrasts, not from the absolute coherence magnitude or cross-band ordering alone.

## Statistical decision

The four band values from each bilateral trial are repeated observations from the same simulated signal pair. Therefore the corrected primary ART model includes a random intercept for trial:

`coherence_median ~ crosstalk * mixture * cortical_high * band + (1 | trial_id)`

The fixed-effects ART model is retained only as a manuscript-reproduction comparison.

## Deliberately deferred

**Independent per-MU synaptic noise is not part of V3.3.** It remains a future model-extension/sensitivity experiment because adding unit-specific input before spike generation changes the model architecture rather than merely correcting an implementation error.
