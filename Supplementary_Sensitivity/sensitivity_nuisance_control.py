"""
Control sensitivity analysis: non-input (nuisance) parameters vs. band-median coherence.

Purpose
-------
The main paper's central non-trivial claim is that the beta<->low-gamma
redistribution depends on *input structure* (narrow vs broad cortical drive).
Here we show the opposite direction: when we sweep two purely *downstream*
(post-LIF, post-input) nuisance parameters while holding all input structure
fixed, the beta<->low-gamma redistribution does NOT arise.

We sweep, independently:
  - MUAP duration (muap_dur_s)   : purely waveform-shape parameter
  - EMG-level noise SNR (SNRE)   : sensor-level noise floor

For comparison, we also run the same format sweep over cortical_high_hz
(the input-structure parameter from the main paper), which IS expected to
redistribute beta <-> low-gamma. This gives the reviewer a visual control.

Outputs
-------
- results/nuisance_control.csv : raw per-trial
- results/nuisance_control_summary.csv : mean/SE per (param, level)
- results/nuisance_control.png : 3-panel figure comparing sweeps
- results/nuisance_control_stats.txt : KW per band per sweep

Usage
-----
    python sensitivity_nuisance_control.py --n_seeds 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "CODES_GITHUB"))

from functions import (
    generate_modulated_EMG_physiological_upgraded,
    simulate_bipolar_emg_spatial, band_medians, bandpass_emg,
    zscore, resample_to_fs, compute_wavelet_coherence,
)


BASE_CONFIG = dict(
    motorunits_max=72, n_fibers=346, FreqInputTotal=100,
    SNR=40, SNRE=40, SNRF=8,
    V_reset=-0.080, V_e=-0.075, Vth_min=-0.055, Vth_max=-0.040,
    Rm=10e6, tau_m=10e-3,
    dt=0.0002, const_current=10.0,
    freq_left=0.5, freq_right=1.0,
    phase_left=0.0, phase_right=np.pi/2,
    window_size=0.1, Intent_variability=0.01,
    cross_talk_R2L=0.20, cross_talk_L2R=0.01,
    common_mod_cortical=4.875, common_mod_subcortical=1.625,
    I_scale=1.5e-9,
    cortical_high_hz=60,
    muap_dur_s=0.018, muap_jitter_s=0.004,
    use_dog_muaps=True, use_lif_adaptation=True,
    noise_type="pink",
)


def run_single(override, seed, T_end, base=BASE_CONFIG):
    cfg = dict(base); cfg.update(override)
    cfg.update(dict(T_end=T_end, seed=int(seed)))
    EMG_L, EMG_R, fs_sim = generate_modulated_EMG_physiological_upgraded(**cfg)
    bL = simulate_bipolar_emg_spatial(EMG_L, fs_sim, tau_ms=1.2, lp_hz=180.0)
    bR = simulate_bipolar_emg_spatial(EMG_R, fs_sim, tau_ms=1.2, lp_hz=180.0)
    fs_w = 1000.0
    bL_rs = resample_to_fs(bL, fs_sim, fs_w)
    bR_rs = resample_to_fs(bR, fs_sim, fs_w)
    bL_bp = zscore(bandpass_emg(bL_rs, fs_w))
    bR_bp = zscore(bandpass_emg(bR_rs, fs_w))
    freqs, _, _, Rsq, *_ = compute_wavelet_coherence(
        bL_bp, bR_bp, fs=fs_w, fmax=128.0, dj=1/8, s0=None, J=None, w0=6
    )
    return band_medians(Rsq, freqs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--T_end", type=float, default=18.0)
    ap.add_argument("--outdir", type=str, default=str(HERE / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Sweep definitions: (name_of_sweep, param_key, values, xlabel)
    sweeps = [
        ("MUAP duration (nuisance)", "muap_dur_s",
         [0.010, 0.014, 0.018, 0.022, 0.026], "MUAP duration (ms)"),
        ("EMG-level SNR (nuisance)", "SNRE",
         [20.0, 30.0, 40.0, 50.0, 60.0], "SNR_EMG (dB)"),
        ("Cortical bandwidth (input)", "cortical_high_hz",
         [20.0, 30.0, 45.0, 60.0, 80.0], "Cortical upper cutoff (Hz)"),
    ]

    rows = []
    total = sum(len(v) * args.n_seeds for _, _, v, _ in sweeps)
    done = 0
    t0 = time.time()
    print(f"Running 3 sweeps x {args.n_seeds} seeds = {total} trials")
    for sweep_name, key, values, _xlab in sweeps:
        print(f"\n-- Sweep: {sweep_name} (param={key}) --")
        for v in values:
            for k in range(args.n_seeds):
                seed = 2000 + hash((key, v)) % 10000 + k
                r = run_single({key: v}, seed, args.T_end)
                rows.append({"sweep": sweep_name, "param": key,
                             "value": (v * 1e3 if key == "muap_dur_s" else float(v)),
                             "seed": int(seed), **r})
                done += 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] {key}={v}  seed={seed}  "
                      f"a={r['5-13']:.3f}  b={r['13-30']:.3f}  "
                      f"lg={r['30-60']:.3f}  hg={r['60-100']:.3f}  "
                      f"(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "nuisance_control.csv", index=False)

    bands = ["5-13", "13-30", "30-60", "60-100"]
    band_labels = ["alpha", "beta", "low-gamma", "high-gamma"]
    colors = {"5-13": "#4477AA", "13-30": "#EE6677",
              "30-60": "#228833", "60-100": "#CCBB44"}

    # Summary
    summary = df.groupby(["sweep", "value"])[bands].agg(["mean", "sem"])
    summary.to_csv(outdir / "nuisance_control_summary.csv")

    # Per-sweep stats: does any sweep produce a beta <-> low-gamma redistribution?
    stats_lines = ["Kruskal-Wallis across levels, per sweep, per band",
                   "-" * 60]
    for sweep_name, _, values, _ in sweeps:
        stats_lines.append(f"\n{sweep_name}:")
        for b, lab in zip(bands, band_labels):
            groups = [df.loc[(df.sweep == sweep_name) & (df.value == (
                v * 1e3 if 'MUAP' in sweep_name else float(v))), b].values
                      for v in values]
            if all(len(g) >= 2 for g in groups):
                H, p = stats.kruskal(*groups)
                stats_lines.append(f"  {lab:11s}: H = {H:6.3f}, p = {p:.4f}")
            else:
                stats_lines.append(f"  {lab:11s}: insufficient data")
    stats_txt = "\n".join(stats_lines)
    print("\n" + stats_txt)
    (outdir / "nuisance_control_stats.txt").write_text(stats_txt + "\n")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    for ax, (sweep_name, key, values, xlab) in zip(axes, sweeps):
        sub = df[df.sweep == sweep_name]
        for b, lab in zip(bands, band_labels):
            g = sub.groupby("value")[b].agg(["mean", "sem"]).reset_index()
            ax.errorbar(g["value"], g["mean"], yerr=g["sem"],
                        marker="o", lw=2, capsize=3, label=lab, color=colors[b])
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_title(sweep_name, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Band-median coherence", fontsize=11)
    axes[0].legend(loc="best", fontsize=9, frameon=False)
    fig.suptitle("Control sensitivity: nuisance parameters (left, middle) do not "
                 "reproduce the $\\beta \\leftrightarrow$ low-$\\gamma$ redistribution "
                 "seen when input structure is varied (right)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "nuisance_control.png", dpi=300)
    fig.savefig(outdir / "nuisance_control.pdf")

    print(f"\nDone. Outputs in {outdir.resolve()}")


if __name__ == "__main__":
    main()
