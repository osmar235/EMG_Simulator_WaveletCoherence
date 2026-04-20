# Supplementary Sensitivity Analyses

Scripts and preview results addressing reviewer requests for the CMPB
revision of *"A Computational Framework for EMG Simulation and
Coherence-Based Biomarker Analysis on Neural Crosstalk in Bimanual
Coordination"* (CMPB-D-25-06796).

## What is in this folder

| File | Reviewer item | What it does |
| --- | --- | --- |
| `sensitivity_tau_m.py` | R1 (alpha-band, tau_m) | Sweeps the LIF membrane time constant tau_m and reports band-median wavelet coherence. Tests whether alpha-band insensitivity is a property of the input structure or a parameter artifact. |
| `sensitivity_nuisance_control.py` | R2 Q5(c) | Sweeps two purely *downstream* nuisance parameters (MUAP duration, EMG SNR) and one *input-structure* parameter (cortical bandwidth), to show the beta <-> low-gamma redistribution arises from input structure, not generic parameter sensitivity. |
| `sensitivity_mu_pool.py` | R1 (pool size) | Sweeps motor-unit pool size (36, 72, 144) at a representative configuration. |
| `stats_emg_features.py` | R3 Q4(c) | Computes temporal and spectral EMG features (RMS, ZCR, WL, MDF, MNF, spectral entropy, beta- and low-gamma-normalized PSD power) for two groups of trials and compares them with Cohen's d, Mann-Whitney U, and Kolmogorov-Smirnov. Runs in `--demo` mode (compares two simulated configs) or with `--data_root` pointing at folders of real and simulated trials. |
| `results/` | -- | Outputs from the preview runs in this sandbox (CSV summaries, PNG/PDF figures, text stats). |

## Re-running locally

All scripts use the same simulator API as `run_single_demo.py` in the
GitHub repo. From a checkout that has `CODES_GITHUB/` and
`Supplementary_Sensitivity/` as siblings:

```bash
# Install once
pip install -r ../CODES_GITHUB/requirements.txt

# Each script accepts --n_seeds, --T_end, --outdir
python sensitivity_tau_m.py            --n_seeds 10 --T_end 18
python sensitivity_nuisance_control.py --n_seeds 10 --T_end 18
python sensitivity_mu_pool.py          --n_seeds 10 --T_end 18

# Stats: demo mode (no experimental data needed)
python stats_emg_features.py --demo

# Stats: real vs simulated, once you have the experimental CSVs in place
#   <DATA_ROOT>/real/*.csv         <-- columns: time_s, left_emg, right_emg
#   <DATA_ROOT>/simulated/*.csv
python stats_emg_features.py --data_root ./my_data
```

