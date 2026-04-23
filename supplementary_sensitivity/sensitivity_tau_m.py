"""
Sensitivity analysis: Membrane time constant tau_m vs. band-median wavelet coherence.

Purpose
-------

We hold all input structure fixed at a representative configuration (broad
cortical bandwidth 13-60 Hz, mixed 75/25 common drive, asymmetric coupling
R->L = 20%, L->R = 1%) and sweep the LIF membrane time constant tau_m over
a physiologically reasonable range. If alpha-band coherence remains at floor
across the sweep, the finding is robust to tau_m (i.e., not an artifact of
the default choice).

Outputs
-------
- results/tau_m_sensitivity.csv : raw per-trial band medians
- results/tau_m_sensitivity_summary.csv : mean/SE per tau_m value
- results/tau_m_sensitivity.png : supplementary figure (band medians vs tau_m)
- results/tau_m_sensitivity_stats.txt : text summary (Friedman / Kruskal-Wallis)

Usage
-----
    python sensitivity_tau_m.py --n_seeds 5 --T_end 18 --outdir results

Rerun locally with more seeds for publication-quality error bars:
    python sensitivity_tau_m.py --n_seeds 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Import the simulator from the adjacent CODES_GITHUB package
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "CODES_GITHUB"))

from functions import (
    generate_modulated_EMG_physiological_upgraded,
    simulate_bipolar_emg_spatial,
    band_medians,
    bandpass_emg,
    zscore,
    resample_to_fs,
    compute_wavelet_coherence,
)


# -------- Representative input configuration (held constant across the sweep) --------
BASE_CONFIG = dict(
    motorunits_max=72,
    n_fibers=346,
    FreqInputTotal=100,
    SNR=40, SNRE=40, SNRF=8,
    V_reset=-0.080, V_e=-0.075,
    Vth_min=-0.055, Vth_max=-0.040,
    Rm=10e6,
    dt=0.0002,
    const_current=10.0,
    freq_left=0.5, freq_right=1.0,
    phase_left=0.0, phase_right=np.pi/2,
    window_size=0.1,
    Intent_variability=0.01,
    cross_talk_R2L=0.20,      # representative non-zero coupling
    cross_talk_L2R=0.01,
    common_mod_cortical=4.875, # 75/25 mix
    common_mod_subcortical=1.625,
    I_scale=1.5e-9,
    cortical_high_hz=60,       # broad cortical bandwidth
    muap_dur_s=0.018,
    muap_jitter_s=0.004,
    use_dog_muaps=True,
    use_lif_adaptation=True,
    noise_type="pink",
)


def run_single(tau_m_s, seed, T_end, base=BASE_CONFIG):
    """Run one trial, return dict of band medians + meta."""
    cfg = dict(base)
    cfg.update(dict(tau_m=tau_m_s, T_end=T_end, seed=int(seed)))
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
    med = band_medians(Rsq, freqs)
    return {"tau_m_ms": tau_m_s * 1e3, "seed": int(seed), **med}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau_m_ms", type=float, nargs="+",
                    default=[5.0, 10.0, 15.0, 20.0, 30.0],
                    help="Membrane time constants to sweep (ms).")
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Number of independent seeds per tau_m value.")
    ap.add_argument("--T_end", type=float, default=18.0,
                    help="Simulated trial duration (s).")
    ap.add_argument("--outdir", type=str, default=str(HERE / "results"))
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Run sweep ----
    rows = []
    total = len(args.tau_m_ms) * args.n_seeds
    done = 0
    t_start = time.time()
    print(f"Running tau_m sweep: {len(args.tau_m_ms)} values x {args.n_seeds} seeds = {total} trials")
    for tau_ms in args.tau_m_ms:
        for k in range(args.n_seeds):
            seed = 1000 + int(tau_ms * 10) + k
            r = run_single(tau_ms * 1e-3, seed, args.T_end)
            rows.append(r)
            done += 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done)
            print(f"  [{done}/{total}] tau_m={tau_ms:5.1f} ms  seed={seed}  "
                  f"alpha={r['5-13']:.3f}  beta={r['13-30']:.3f}  "
                  f"low-gamma={r['30-60']:.3f}  high-gamma={r['60-100']:.3f}  "
                  f"(elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min)")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tau_m_sensitivity.csv", index=False)

    # ---- Summary ----
    bands = ["5-13", "13-30", "30-60", "60-100"]
    band_labels = ["alpha (5-13)", "beta (13-30)", "low-gamma (30-60)", "high-gamma (60-100)"]
    summary = df.groupby("tau_m_ms")[bands].agg(["mean", "std", "sem"])
    summary.to_csv(outdir / "tau_m_sensitivity_summary.csv")
    print("\nSummary (mean +- SEM):")
    print(summary.round(4))

    # ---- Nonparametric test across tau_m per band (Kruskal-Wallis) ----
    stats_lines = ["Kruskal-Wallis across tau_m values, per frequency band",
                   "-------------------------------------------------------"]
    for b, lab in zip(bands, band_labels):
        groups = [df.loc[df.tau_m_ms == v, b].values for v in args.tau_m_ms]
        if all(len(g) >= 2 for g in groups) and len(args.tau_m_ms) >= 2:
            H, p = stats.kruskal(*groups)
            stats_lines.append(f"  {lab:20s}: H = {H:.3f},  p = {p:.4f}")
        else:
            stats_lines.append(f"  {lab:20s}: insufficient data for Kruskal-Wallis")
    stats_txt = "\n".join(stats_lines)
    print("\n" + stats_txt)
    (outdir / "tau_m_sensitivity_stats.txt").write_text(stats_txt + "\n")

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = {"5-13": "#4477AA", "13-30": "#EE6677", "30-60": "#228833", "60-100": "#CCBB44"}
    for b, lab in zip(bands, band_labels):
        g = df.groupby("tau_m_ms")[b].agg(["mean", "sem"]).reset_index()
        ax.errorbar(g["tau_m_ms"], g["mean"], yerr=g["sem"],
                    marker="o", lw=2, capsize=3, label=lab, color=colors[b])
    ax.set_xlabel("Membrane time constant $\\tau_m$ (ms)", fontsize=12)
    ax.set_ylabel("Band-median wavelet coherence", fontsize=12)
    ax.set_title("Sensitivity of band-median coherence to $\\tau_m$\n"
                 "(representative config: broad 13-60 Hz cortical, 75/25 drive, "
                 "R->L = 20%)", fontsize=11)
    ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax.set_ylim(0, None)
    ax.legend(loc="best", fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "tau_m_sensitivity.png", dpi=300)
    fig.savefig(outdir / "tau_m_sensitivity.pdf")

    print(f"\nDone. Outputs in {outdir.resolve()}")
    print(f"  - tau_m_sensitivity.csv          (raw per-trial)")
    print(f"  - tau_m_sensitivity_summary.csv  (mean / std / sem per tau_m)")
    print(f"  - tau_m_sensitivity_stats.txt    (Kruskal-Wallis per band)")
    print(f"  - tau_m_sensitivity.png / .pdf   (supplementary figure)")


if __name__ == "__main__":
    main()
