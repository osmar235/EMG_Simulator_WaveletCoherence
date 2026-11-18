# EMG Simulator – Wavelet Coherence

Bilateral EMG simulator with cortical/subcortical common drive, optional cross-limb mixing,
LIF motor units, MUAP synthesis, and wavelet-coherence analysis (Morlet, pycwt).

> **Version 2.0.0 is a bug-fix release.** It corrects the wavelet-coherence smoothing and the
> 100-ms window reset of the motor-neuron pool; see [CHANGELOG.md](CHANGELOG.md). The exact
> code behind the published analyses is kept in `legacy/`.

## Install
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run a single demo
```bash
python run_single_demo.py
```
Outputs are written to `results/`:

- `single_demo_emg_timeseries.png`: stacked left/right bipolar EMG (5–499 Hz, z-scored).
- `single_demo_wavelet_coherence.png`: Morlet wavelet coherence (5–100 Hz) with the COI and Monte-Carlo significance.
- `single_demo_coh_avg.png`: time-averaged coherence spectrum.
- `single_demo_emg_bipolar_1k.csv`: columns time_s, left_emg, right_emg (1 kHz).

### Check the v2 fixes
```bash
python validate_fixes.py        # ~3 min; writes results/validation_report.txt and fig_validation.png
```

## API (quick start)
Everything lives in `functions.py`:
```python
import numpy as np
from functions import (generate_modulated_EMG_physiological_upgraded, simulate_bipolar_emg_spatial,
                       bandpass_emg, zscore, resample_to_fs, compute_wavelet_coherence,
                       band_medians, mc_null_threshold)

EMG_L, EMG_R, fs = generate_modulated_EMG_physiological_upgraded(
    motorunits_max=72, n_fibers=346, FreqInputTotal=100,
    SNR=40, SNRE=40, SNRF=8, T_end=18,
    V_reset=-0.080, V_e=-0.075, Vth_min=-0.055, Vth_max=-0.040,
    Rm=10e6, tau_m=10e-3, dt=0.0002,
    const_current=10.0, freq_left=0.5, freq_right=1.0,
    phase_left=0.0, phase_right=np.pi/2, window_size=0.1,
    Intent_variability=0.01, cross_talk_R2L=0.20, cross_talk_L2R=0.01,
    common_mod_cortical=3.25, common_mod_subcortical=3.25,
    I_scale=1.5e-9, cortical_high_hz=60,
    muap_dur_s=0.018, muap_jitter_s=0.004,
    use_dog_muaps=True, use_lif_adaptation=True, noise_type='pink', seed=43)

bL = simulate_bipolar_emg_spatial(EMG_L, fs, tau_ms=1.2, lp_hz=180)
bR = simulate_bipolar_emg_spatial(EMG_R, fs, tau_ms=1.2, lp_hz=180)
x = zscore(bandpass_emg(resample_to_fs(bL, fs, 1000), 1000))
y = zscore(bandpass_emg(resample_to_fs(bR, fs, 1000), 1000))
freqs, period, coi, Rsq, *_ = compute_wavelet_coherence(x, y, fs=1000)
print(band_medians(Rsq, freqs, period=period, coi=coi))   # medians inside the cone of influence
```
Pass `return_spikes=True` to also get the spike trains. See the docstring of
`generate_modulated_EMG_physiological_upgraded` for all parameters.

## Reproducing the published (v1) numbers
```python
import sys; sys.path.insert(0, "legacy")
import functions_v1 as F   # original code, unmodified
```
v1 draws part of its noise from unseeded generators, so its outputs vary slightly from run to run.
