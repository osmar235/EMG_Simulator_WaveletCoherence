# EMG_Simulator_WaveletCoherence

Bilateral EMG simulation and EMG-EMG wavelet coherence.

This repository is a Python toolbox for two tasks:

1. **Simulating realistic bilateral surface EMG.** The model covers cortical and subcortical common drive, cross-limb neural crosstalk, leaky integrate-and-fire motor-unit pools with adaptation, MUAP synthesis and a bipolar electrode model.
2. **Measuring EMG-EMG (intermuscular) coherence with the Morlet wavelet transform.** This works on simulated signals or on your own recordings, and includes a matched cross-trial significance test.

It is the software of:

> Pinto Neto O, Pinho TOR, Wang Y, Balbinot G, Kennedy DM. **A Computational Framework for EMG Simulation and Coherence-Based Biomarker Analysis on Neural Crosstalk in Bimanual Coordination.** *Computer Methods and Programs in Biomedicine* (2026) 109657. https://doi.org/10.1016/j.cmpb.2026.109657

## How to cite

If you use this code, please cite the article above **and** the archived software. This applies whether you use the simulator, the coherence analysis, or both.

> Pinto Neto O, Pinho TOR, Wang Y, Balbinot G, Kennedy DM. EMG_Simulator_WaveletCoherence (software). Zenodo. https://doi.org/10.5281/zenodo.19703513

GitHub's **"Cite this repository"** button, in the right-hand panel, gives both references in APA and BibTeX from `CITATION.cff`.

## Installation

You need Python 3.9 or newer.

```bash
git clone https://github.com/osmar235/EMG_Simulator_WaveletCoherence.git
cd EMG_Simulator_WaveletCoherence
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python tests/validate.py        # quick self-test, ~5 s -> "RESULT: 13/13 checks passed"
```

Everything lives in one file, `functions.py`. Scripts either run from the repository folder, or add that folder to `sys.path`.

---

## 1. Simulate bilateral EMG

### Quick start

```bash
python run_single_demo.py                     # one-click demo (same as examples/simulate_demo.py)
python examples/simulate_demo.py --crosstalk 0.35 --mixture 50_50 --cortical-high 30 --seed 7
```

Outputs go to `results/demo/`:

- `demo_emg_1k.csv`: the two EMG signals.
- Figures of the EMG signals and of the coherence map.
- Band-median coherence.

### In Python

```python
import functions as F

# Representative configuration of the article:
# 1:2 bimanual task (0.5 / 1.0 Hz force envelopes), 72 motor units per limb,
# common drive 75/25 cortical (13-60 Hz) / subcortical (5-13 Hz),
# 20 % right-to-left crosstalk, 18 s
left, right, fs = F.simulate_bilateral_emg(seed=1, fs_out=1000.0)

# Change anything you like
left, right, fs, info = F.simulate_bilateral_emg(
    seed=2, T_end=30, cross_talk_R2L=0.0,            # no crosstalk
    common_mod_cortical=6.0, common_mod_subcortical=0.0, cortical_high_hz=30,
    freq_left=0, phase_left=1.5708, freq_right=0, phase_right=1.5708,  # constant force
    return_spikes=True)                              # info['spikes_L'], info['spikes_R']
```

`simulate_bilateral_emg` returns **bipolar** surface EMG by default. Pass `bipolar=False` for monopolar signals. For full control, use `generate_modulated_EMG_physiological_upgraded(...)`; every argument has a documented default.

### Main parameters

| Parameter | Default | Meaning |
|---|---|---|
| `T_end` | 18 | Returned duration (s). A separate 1-s burn-in is simulated first. |
| `dt` | 0.0002 | Simulation step (s), i.e. 5 kHz. |
| `motorunits_max` | 72 | Motor units per limb. |
| `const_current` | 10 | Amplitude of each limb's unique drive. |
| `common_mod_cortical`, `common_mod_subcortical` | 4.875, 1.625 | Relative amplitude weights of the common drive: 6/0 = 100/0, 4.875/1.625 = 75/25, 3.25/3.25 = 50/50. |
| `common_drive_target_rms` | 6 | Total amplitude (SD) of the common drive, the same for every mixture. |
| `cortical_high_hz` | 60 | Upper edge of the cortical band (13–30 or 13–60 Hz). |
| `cross_talk_R2L`, `cross_talk_L2R` | 0.20, 0.01 | Fraction of the other limb's input mixed in (0–1). |
| `freq_left`, `freq_right` | 0.5, 1.0 | Force-envelope frequencies (Hz). Use `freq=0, phase=pi/2` for a constant force. |
| `Intent_variability` | 0.01 | Slow (<0.25 Hz) envelope variability, as a fractional RMS. |
| `SNR`, `SNRF`, `SNRE` | 40, 8, 40 | SNR in dB of the drive noise, motor-unit-level noise and EMG-level noise. |
| `muap_dur_s`, `muap_jitter_s` | 0.018, 0.004 | Mean MUAP duration and per-unit jitter (s). |
| `min_isi_ms`, `dVth`, `tau_adapt_ms` | 18, 0.003, 250 | Refractory period and threshold adaptation. |
| `seed` | 1234 | The same seed gives identical signals. |

The docstrings of `generate_modulated_EMG_physiological_upgraded` and `simulate_bilateral_emg` list all the parameters.

---

## 2. Wavelet coherence of your own EMG

### Command line

Each file holds one trial: a CSV or TXT file with one column per channel, and optionally a time column.

```bash
# one trial, 2 kHz, channels by name, remove 60-Hz mains
python examples/analyze_emg_coherence.py my_trial.csv --fs 2000 --left FDI_L --right FDI_R --notch 60

# several trials of the same condition -> also tests coherence against a matched null
python examples/analyze_emg_coherence.py "data/condition1_*.csv" --fs 2000 --notch 50
```

The processing steps are:

1. Raw, **non-rectified** EMG.
2. Optional mains notch.
3. Resampling to 1 kHz.
4. 5–499 Hz band-pass, linear detrend and z-score.
5. Squared Morlet wavelet coherence (w0 = 6, 1/8-octave resolution, 4–128 Hz).
6. Median coherence **inside the cone of influence** in 5–13, 13–30, 30–60 and 60–100 Hz. Other bands can be set with `--bands "8-12,15-35"`.

Outputs go to `results/coherence/`:

- `band_medians_per_trial.csv`
- `coherence_map_<trial>.png`
- `cross_trial_null.csv`, when two or more trials are given.

**Significance.** With two or more trials of one condition, the left signal of trial *i* is paired with the right signal of trial *j* ≠ *i*. This keeps each signal's spectrum, amplitude modulation and task timing, but removes within-trial coupling. The p-value is one-sided and is computed from derangements of the trial labels. With few trials it is exact; use at least 6 trials for a useful p-value resolution. Because the null is matched this way, it also absorbs edge and task-related coherence floors, which can be high below ~13 Hz, where surface EMG has little power.

### In Python

```python
import functions as F

# emg_left, emg_right: raw 1-D arrays recorded at fs_raw Hz
x = F.preprocess_emg_for_coherence(F.resample_to_fs(emg_left, fs_raw, 1000), 1000, notch_hz=60)  # or 50 / None
y = F.preprocess_emg_for_coherence(F.resample_to_fs(emg_right, fs_raw, 1000), 1000, notch_hz=60)
freqs, period, coi, Rsq, *_ = F.compute_wavelet_coherence(x, y, fs=1000)
print(F.band_medians(Rsq, freqs, period=period, coi=coi))      # {'5-13': ..., '13-30': ..., ...}
F.plot_coherence_map(Rsq, freqs, period, coi, 1000, "coherence.png")

# several trials (lists of preprocessed, equal-length arrays)
res = F.cross_trial_null(left_trials, right_trials, fs=1000, n_perm=999)
print(res["bands"], res["observed"], res["null_median"], res["p"])
```

---

## 3. Reproduce the factorial simulation study of the article

The design is 8 crosstalk levels (0–35 %) × 3 common-drive mixtures × 2 cortical bandwidths × 15 trials, which gives 720 bilateral trials.

```bash
python paper_analysis/run_factorial.py --dry-run
python paper_analysis/run_factorial.py                                   # ~40-60 min on a laptop
python paper_analysis/cross_trial_null_factorial.py results/factorial    # coherence vs matched null
Rscript paper_analysis/art_analysis.R results/factorial/factorial_band_medians.csv   # ART ANOVA + post hocs
```

The R step needs R with `install.packages(c("ARTool","lme4","emmeans","dplyr","readr"))`. Seeds are fixed (`--base-seed 20260926`), so the study is reproduced exactly.

---

## Repository contents

```
functions.py                          simulator + coherence toolbox (single file)
run_single_demo.py                    one-click demo
examples/simulate_demo.py             simulate one EMG pair, plot, coherence
examples/analyze_emg_coherence.py     coherence of your own recordings (+ matched null)
paper_analysis/run_factorial.py       factorial simulation study
paper_analysis/cross_trial_null_factorial.py
paper_analysis/art_analysis.R         aligned-rank-transform ANOVA
tests/validate.py                     self-test
requirements.txt  CITATION.cff  LICENSE
```

## Model summary

Each limb's input is its unique low-pass drive plus a common drive shared by both limbs. The common drive has a cortical component (13–30 or 13–60 Hz) and a subcortical component (5–13 Hz), mixed at a fixed total amplitude. This input is scaled by the task force envelope.

Cross-limb crosstalk mixes a fraction of one limb's input into the other. The resulting currents drive leaky integrate-and-fire motor neurons, which have graded thresholds, an 18-ms refractory period and after-spike threshold adaptation; they are integrated continuously over the whole trial.

Spike trains are convolved with difference-of-Gaussian MUAPs, dispersed across muscle fibres, and summed with motor-unit-level and EMG-level noise. Bipolar surface EMG is obtained by differencing the signal with a copy delayed by 1.2 ms and low-pass filtering it at 180 Hz. Coherence follows Torrence & Compo (1998) and Grinsted et al. (2004), with Gaussian smoothing in time and 0.6-octave smoothing in scale.

## License

MIT License (see `LICENSE`). The article is open access under CC BY 4.0.
