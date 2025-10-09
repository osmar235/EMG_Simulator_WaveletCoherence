# EMG_Simulator_WaveletCoherence

Bilateral EMG simulator with cortical/subcortical common drive, optional cross-limb mixing, MUAP synthesis, and Morlet wavelet-coherence analysis (pycwt). Includes a one-click demo that plots EMGs/coherence and exports CSV. Please cite the paper + Zenodo DOI.

---

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python run_single_demo.py
from functions import (
  generate_modulated_EMG_physiological_upgraded,
  simulate_bipolar_emg_spatial,
  bandpass_emg, zscore, resample_to_fs,
  compute_wavelet_coherence, mc_null_threshold
)

import numpy as np
from functions import (
  generate_modulated_EMG_physiological_upgraded,
  simulate_bipolar_emg_spatial,
  bandpass_emg, zscore, resample_to_fs,
  compute_wavelet_coherence, mc_null_threshold
)

EMG_L, EMG_R, fs = generate_modulated_EMG_physiological_upgraded(
    motorunits_max=72, n_fibers=346, FreqInputTotal=100,
    SNR=40, SNRE=40, SNRF=8, T_end=18,
    V_reset=-0.080, V_e=-0.075, Vth_min=-0.055, Vth_max=-0.040,
    Rm=10e6, tau_m=10e-3, dt=0.0002,
    const_current=10.0, freq_left=0.5, freq_right=1.0,
    phase_left=0.0, phase_right=np.pi/2,
    window_size=0.1, Intent_variability=0.01,
    cross_talk_R2L=0.20, cross_talk_L2R=0.01,
    common_mod_cortical=3.25, common_mod_subcortical=3.25,
    I_scale=1.5e-9, cortical_high_hz=60,
    muap_dur_s=0.018, muap_jitter_s=0.004,
    use_dog_muaps=True, use_lif_adaptation=True,
    noise_type='pink', shaper_taps_mu=None, shaper_taps_emg=None, seed=43)

bL = simulate_bipolar_emg_spatial(EMG_L, fs, tau_ms=1.2, lp_hz=180)
bR = simulate_bipolar_emg_spatial(EMG_R, fs, tau_ms=1.2, lp_hz=180)
