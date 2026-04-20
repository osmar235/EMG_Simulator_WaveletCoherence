"""
Sensitivity analysis: Motor-unit pool size vs. band-median wavelet coherence.

Purpose
-------
Addresses Reviewer 1's comment: "Discuss how larger motor unit pools (e.g.,
tibialis anterior) would affect coherence resolution." The reviewer explicitly
accepted "discussion OR a short additional simulation" - this is the short
simulation.

We hold all input structure fixed at the representative configuration and
sweep motorunits_max over three physiologically plausible sizes:
  - 36 MUs : small intrinsic hand muscle (lower bound)
  - 72 MUs : FDI baseline (study default)
  - 144 MUs : larger hand / intrinsic-pool upper bound or small proximal

This brackets the range between intrinsic and small proximal pools without
claiming to model the specific TA/biceps architecture (which have distinct
MUAP and recruitment statistics beyond pool size alone).

Outputs
-------
- results/mu_pool_sensitivity.csv
- results/mu_pool_sensitivity_summary.csv
- results/mu_pool_sensitivity.png
- results/mu_pool_sensitivity_stats.txt

Usage
-----
    python sensitivity_mu_pool.py --n_seeds 5
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
    n_fibers=346, FreqInputTotal=100,
    SNR=40, SNRE=40, SNRF=8,
    V_reset=-0.080, V_e=-0.075, Vth_min=-0.055, Vth_max=-0.040,
    Rm=10e6, tau_m=10e-3,
    dt=0.0002, const_current=10.0,
    freq_left=0.5, freq_right=1.0,
    phase_left=0.0, phase_right=np.pi/2,
    window_size=0.1, Intent_variability=0.01,
    cross_talk_R2L=0.20, cross_talk_L2R=0.01,
    common_mod_cortical=4.875, common_mod_subcortical=1.625,
    I_scale=1.5e-9, cortical_high_hz=60,
    muap_dur_s=0.018, muap_jitter_s=0.004,
    use_dog_muaps=True, use_lif_adaptation=True, noise_type="pink",
)


def run_single(n_mu, seed, T_end, base=BASE_CONFIG):
    cfg = dict(base)
    cfg.update(dict(motorunits_max=int(n_mu), T_end=T_end, seed=int(seed)))
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
    ap.add_argument("--pool_sizes", type=int, nargs="+",
                    default=[36, 72, 144])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--T_end", type=float, default=18.0)
    ap.add_argument("--outdir", type=str, default=str(HERE / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(args.pool_sizes) * args.n_seeds
    done = 0
    t0 = time.time()
    print(f"Running pool-size sweep: {len(args.pool_sizes)} sizes x {args.n_seeds} seeds")
    for n_mu in args.pool_sizes:
        for k in range(args.n_seeds):
            seed = 4000 + n_mu + k
            r = run_single(n_mu, seed, args.T_end)
            rows.append({"n_mu": int(n_mu), "seed": int(seed), **r})
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{done}/{total}] n_mu={n_mu}  seed={seed}  "
                  f"a={r['5-13']:.3f}  b={r['13-30']:.3f}  "
                  f"lg={r['30-60']:.3f}  hg={r['60-100']:.3f}  "
                  f"(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "mu_pool_sensitivity.csv", index=False)

    bands = ["5-13", "13-30", "30-60", "60-100"]
    band_labels = ["alpha", "beta", "low-gamma", "high-gamma"]
    colors = {"5-13": "#4477AA", "13-30": "#EE6677",
              "30-60": "#228833", "60-100": "#CCBB44"}

    summary = df.groupby("n_mu")[bands].agg(["mean", "std", "sem"])
    summary.to_csv(outdir / "mu_pool_sensitivity_summary.csv")
    print("\nSummary:\n", summary.round(4))

    lines = ["Kruskal-Wallis across pool sizes, per band", "-" * 50]
    for b, lab in zip(bands, band_labels):
        groups = [df.loc[df.n_mu == v, b].values for v in args.pool_sizes]
        if all(len(g) >= 2 for g in groups):
            H, p = stats.kruskal(*groups)
            lines.append(f"  {lab:11s}: H = {H:.3f}, p = {p:.4f}")
    txt = "\n".join(lines)
    print("\n" + txt)
    (outdir / "mu_pool_sensitivity_stats.txt").write_text(txt + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for b, lab in zip(bands, band_labels):
        g = df.groupby("n_mu")[b].agg(["mean", "sem"]).reset_index()
        ax.errorbar(g["n_mu"], g["mean"], yerr=g["sem"],
                    marker="o", lw=2, capsize=3, label=lab, color=colors[b])
    ax.set_xlabel("Motor-unit pool size", fontsize=12)
    ax.set_ylabel("Band-median wavelet coherence", fontsize=12)
    ax.set_title("Sensitivity of band-median coherence to motor-unit pool size\n"
                 "(representative config; 36 = smaller intrinsic, 72 = FDI baseline, "
                 "144 = larger pool)", fontsize=11)
    ax.set_ylim(0, None)
    ax.legend(loc="best", fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "mu_pool_sensitivity.png", dpi=300)
    fig.savefig(outdir / "mu_pool_sensitivity.pdf")
    print(f"\nDone. Outputs in {outdir.resolve()}")


if __name__ == "__main__":
    main()
