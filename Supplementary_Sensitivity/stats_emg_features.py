"""
Supplementary statistical comparison of simulated vs. experimental EMG features.

Purpose
-------
Addresses Reviewer 3, Q4(c): "the comparison of the electromyographic signals
is presented in the Supplementary Data section, without any statistical
analysis of their temporal and spectral properties."

This script computes, for each trial (real or simulated), a set of
standard temporal and spectral EMG features on bipolar, band-pass filtered,
z-scored signals:

  Temporal
  --------
  RMS_z           : root-mean-square of the z-scored EMG (dimensionless).
  ZCR             : zero-crossing rate (per second).
  WL_z            : waveform length / total variation of the z-scored EMG.

  Spectral (power spectrum via Welch)
  -----------------------------------
  MDF             : median frequency (Hz).
  MNF             : mean frequency (Hz).
  SE              : normalized spectral entropy (0..1).
  P_beta_norm     : normalized PSD power in 13-30 Hz band.
  P_lowgamma_norm : normalized PSD power in 30-60 Hz band.

For each feature we compute:
  - mean +/- SD for each group
  - Cohen's d effect size (pooled SD, Hedges correction)
  - Mann-Whitney U test
  - Two-sample Kolmogorov-Smirnov test

Input formats
-------------
Three ways to supply experimental trials:

1. --real_txt_dir <folder> : one .txt file per trial, tab-separated with
   header "EMG_R\tEMG_L" (this project's Delsys export format; assumes fs
   from --fs_real, default 2000 Hz).

2. --data_root <root>      : <root>/real/ and <root>/simulated/ with CSV
   files containing columns time_s, left_emg, right_emg, OR NPZ files
   with arrays 'left', 'right', 'fs'.

3. --demo                  : no experimental data; compares two simulated
   configurations to illustrate the output format.

When --real_txt_dir is provided and --simulated_dir is not, the script
simulates a *matched* number of trials at the best-matching configuration
from the main paper (broad 13-60 Hz cortical bandwidth, mixed 75/25
cortical-subcortical drive, asymmetric bilateral coupling R->L = 20%).

Usage
-----
    # Recommended for this project's Delsys export
    python stats_emg_features.py --real_txt_dir ../P&D_testing_trial/txt_export \
                                 --fs_real 2000

    # Real vs simulated, if trials are already in CSV/NPZ layout
    python stats_emg_features.py --data_root ./my_data

    # Demo (no experimental data needed)
    python stats_emg_features.py --demo
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "CODES_GITHUB"))

from functions import (
    generate_modulated_EMG_physiological_upgraded,
    simulate_bipolar_emg_spatial, bandpass_emg, zscore, resample_to_fs,
)


# --------------------------- Feature computation ---------------------------

def compute_features(x, fs):
    """Return dict of temporal and spectral features for a 1-D EMG trace x at fs."""
    x = np.asarray(x, dtype=float)
    # Temporal
    rms = float(np.sqrt(np.mean(x ** 2)))
    zcr = float(np.sum(np.diff(np.signbit(x)) != 0) / (len(x) / fs))
    wl = float(np.sum(np.abs(np.diff(x))))
    # Spectral via Welch
    nperseg = int(min(len(x), 2 * fs))
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    # restrict to 5-500 Hz (EMG band)
    m = (f >= 5) & (f <= 500)
    f_m, P_m = f[m], Pxx[m]
    cumP = np.cumsum(P_m) / np.sum(P_m)
    mdf = float(f_m[np.searchsorted(cumP, 0.5)])
    mnf = float(np.sum(f_m * P_m) / np.sum(P_m))
    # Spectral entropy (normalized)
    p = P_m / np.sum(P_m)
    p = p[p > 0]
    se = float(-np.sum(p * np.log2(p)) / np.log2(len(p)))
    # Band-normalized power
    def band_pow(lo, hi):
        bm = (f_m >= lo) & (f_m <= hi)
        return float(np.sum(P_m[bm]) / np.sum(P_m))
    return {
        "RMS_z": rms, "ZCR": zcr, "WL_z": wl,
        "MDF": mdf, "MNF": mnf, "SE": se,
        "P_beta_norm": band_pow(13, 30),
        "P_lowgamma_norm": band_pow(30, 60),
    }


def cohens_d(a, b):
    """Hedges-corrected Cohen's d for two independent samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    d = (np.mean(a) - np.mean(b)) / sp
    # Hedges' small-sample correction
    J = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return float(d * J)


# --------------------------- Trial loading ---------------------------

def load_trial(path, fs_txt=2000.0):
    """Return (left, right, fs) from a CSV, NPZ, or TXT trial file."""
    p = Path(path)
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        if "time_s" in cols and "left_emg" in cols and "right_emg" in cols:
            t = df[cols["time_s"]].values
            fs = 1.0 / np.median(np.diff(t))
            return (df[cols["left_emg"]].values,
                    df[cols["right_emg"]].values, float(fs))
        raise ValueError(f"CSV {p} missing time_s/left_emg/right_emg columns")
    if p.suffix == ".npz":
        z = np.load(p)
        fs = float(z["fs"].item()) if "fs" in z.files else 1000.0
        return z["left"], z["right"], fs
    if p.suffix == ".txt":
        # Tab-separated with header "EMG_R\tEMG_L" (this project's Delsys export)
        df = pd.read_csv(p, sep=r"\s+", engine="python")
        cols = {c.upper(): c for c in df.columns}
        if "EMG_L" not in cols or "EMG_R" not in cols:
            raise ValueError(f"TXT {p} missing EMG_L/EMG_R columns")
        L = df[cols["EMG_L"]].values.astype(float)
        R = df[cols["EMG_R"]].values.astype(float)
        return L, R, float(fs_txt)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def prep_emg(x, fs_in, fs_out=1000.0):
    x_rs = resample_to_fs(x, fs_in, fs_out) if abs(fs_in - fs_out) > 1 else x
    x_bp = zscore(bandpass_emg(x_rs, fs_out))
    return x_bp, fs_out


def load_group(folder, fs_txt=2000.0):
    """Load all trials from a folder; return list of (left_bp, right_bp, fs, name)."""
    folder = Path(folder)
    files = sorted([*folder.glob("*.csv"), *folder.glob("*.npz"),
                    *folder.glob("*.txt")])
    trials = []
    for f in files:
        try:
            L, R, fs = load_trial(f, fs_txt=fs_txt)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue
        Lz, fs_w = prep_emg(L, fs); Rz, _ = prep_emg(R, fs)
        trials.append((Lz, Rz, fs_w, f.name))
    return trials


# --------------------------- Demo mode ---------------------------

DEMO_CONFIG_A = dict(
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
    I_scale=1.5e-9, cortical_high_hz=60,
    muap_dur_s=0.018, muap_jitter_s=0.004,
    use_dog_muaps=True, use_lif_adaptation=True, noise_type="pink",
)
# "Alternative" config (e.g., narrow-cortical, pure cortical) - only slightly different
DEMO_CONFIG_B = dict(DEMO_CONFIG_A)
DEMO_CONFIG_B.update(dict(
    cortical_high_hz=30,
    common_mod_cortical=6.0, common_mod_subcortical=0.0,
    cross_talk_R2L=0.10,
))


def simulate_trials(config, n_trials, seed_base, T_end=18.0):
    trials = []
    for k in range(n_trials):
        cfg = dict(config); cfg.update(dict(T_end=T_end, seed=seed_base + k))
        EMG_L, EMG_R, fs_sim = generate_modulated_EMG_physiological_upgraded(**cfg)
        bL = simulate_bipolar_emg_spatial(EMG_L, fs_sim, tau_ms=1.2, lp_hz=180.0)
        bR = simulate_bipolar_emg_spatial(EMG_R, fs_sim, tau_ms=1.2, lp_hz=180.0)
        Lz, fs_w = prep_emg(bL, fs_sim); Rz, _ = prep_emg(bR, fs_sim)
        trials.append((Lz, Rz, fs_w, f"sim_{seed_base + k}"))
    return trials


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real_txt_dir", type=str, default=None,
                    help="Folder with project Delsys *.txt exports "
                         "(tab-separated, header EMG_R\\tEMG_L).")
    ap.add_argument("--fs_real", type=float, default=2000.0,
                    help="Sampling rate for --real_txt_dir files (Hz).")
    ap.add_argument("--simulated_dir", type=str, default=None,
                    help="Optional folder with pre-generated simulated trials "
                         "(CSV/NPZ/TXT). If omitted with --real_txt_dir, the "
                         "script simulates a matched number at the paper's "
                         "best-matching configuration.")
    ap.add_argument("--data_root", type=str, default=None,
                    help="Alternative: folder with real/ and simulated/ subfolders.")
    ap.add_argument("--demo", action="store_true",
                    help="Run in demo mode (no experimental data).")
    ap.add_argument("--n_demo_trials", type=int, default=8)
    ap.add_argument("--T_end_sim", type=float, default=20.0,
                    help="Simulated trial duration (s); match the real-trial length.")
    ap.add_argument("--outdir", type=str, default=str(HERE / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.real_txt_dir and not args.demo:
        real = load_group(args.real_txt_dir, fs_txt=args.fs_real)
        if not real:
            print(f"No .txt/.csv/.npz trials found in {args.real_txt_dir}")
            return
        print(f"Loaded {len(real)} experimental trials from {args.real_txt_dir}")
        if args.simulated_dir:
            sim = load_group(args.simulated_dir, fs_txt=args.fs_real)
        else:
            print(f"Simulating {len(real)} matched trials at paper's best-matching "
                  f"configuration (broad 13-60 Hz cortical, 75/25 drive, "
                  f"R->L = 20%, T_end = {args.T_end_sim}s)...")
            sim = simulate_trials(DEMO_CONFIG_A, len(real), 9000,
                                  T_end=args.T_end_sim)
        label_A, label_B = "Real (experimental FDI)", "Simulated (best-matching cfg)"
    elif args.data_root and not args.demo:
        root = Path(args.data_root)
        real = load_group(root / "real")
        sim = load_group(root / "simulated")
        label_A, label_B = "Real", "Simulated"
        if not real or not sim:
            print("No trials loaded in one of the groups; check --data_root layout.")
            return
    else:
        print("Running in DEMO mode: group A = broad cortical, 75/25 drive; "
              "group B = narrow cortical, 100/0 drive.")
        real = simulate_trials(DEMO_CONFIG_A, args.n_demo_trials, 7000)
        sim = simulate_trials(DEMO_CONFIG_B, args.n_demo_trials, 8000)
        label_A, label_B = "Config A (paper baseline)", "Config B (narrow/pure cortical)"

    # Compute features for every trial (we pool left + right traces)
    def feats(group, group_name):
        rows = []
        for Lz, Rz, fs, name in group:
            for side, x in (("L", Lz), ("R", Rz)):
                f = compute_features(x, fs)
                f.update({"trial": name, "side": side, "group": group_name})
                rows.append(f)
        return rows

    rows = feats(real, label_A) + feats(sim, label_B)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "emg_features.csv", index=False)

    feature_names = ["RMS_z", "ZCR", "WL_z", "MDF", "MNF", "SE",
                     "P_beta_norm", "P_lowgamma_norm"]
    lines = [f"Feature comparison: {label_A}  vs  {label_B}",
             "=" * 70,
             f"n({label_A}) = {sum(df['group']==label_A)}    "
             f"n({label_B}) = {sum(df['group']==label_B)}",
             ""]
    stat_rows = []
    for name in feature_names:
        a = df.loc[df["group"] == label_A, name].dropna().values
        b = df.loc[df["group"] == label_B, name].dropna().values
        d = cohens_d(a, b)
        U, pU = stats.mannwhitneyu(a, b, alternative="two-sided")
        K, pK = stats.ks_2samp(a, b)
        stat_rows.append({
            "feature": name,
            f"{label_A}_mean": float(np.mean(a)),
            f"{label_A}_sd": float(np.std(a, ddof=1)),
            f"{label_B}_mean": float(np.mean(b)),
            f"{label_B}_sd": float(np.std(b, ddof=1)),
            "cohen_d": d,
            "MW_U": float(U), "MW_p": float(pU),
            "KS_D": float(K), "KS_p": float(pK),
        })
        lines.append(
            f"{name:17s}  A: {np.mean(a):7.3f}+-{np.std(a, ddof=1):.3f}   "
            f"B: {np.mean(b):7.3f}+-{np.std(b, ddof=1):.3f}   "
            f"d = {d:+.2f}   MW p = {pU:.4f}   KS p = {pK:.4f}"
        )

    pd.DataFrame(stat_rows).to_csv(outdir / "emg_features_stats.csv", index=False)
    txt = "\n".join(lines)
    print("\n" + txt)
    (outdir / "emg_features_stats.txt").write_text(txt + "\n")

    # ---- Supplementary comparison figure: boxplots per feature ----
    import matplotlib.pyplot as plt
    ncol = 4
    nrow = int(np.ceil(len(feature_names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
    axes = np.atleast_2d(axes).ravel()
    for ax, name in zip(axes, feature_names):
        a = df.loc[df["group"] == label_A, name].dropna().values
        b = df.loc[df["group"] == label_B, name].dropna().values
        bp = ax.boxplot([a, b], labels=["Real", "Sim"], widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], ["#4477AA", "#EE6677"]):
            patch.set_facecolor(col); patch.set_alpha(0.5)
        # strip plot overlay
        for i, vals in enumerate([a, b]):
            x = np.random.normal(i + 1, 0.04, size=len(vals))
            ax.plot(x, vals, ".", ms=3, color="k", alpha=0.4)
        d = cohens_d(a, b)
        _, pK = stats.ks_2samp(a, b)
        ax.set_title(f"{name}\nd = {d:+.2f}, KS p = {pK:.3f}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(feature_names):]:
        ax.axis("off")
    fig.suptitle(f"{label_A} vs. {label_B}  (n = {sum(df['group']==label_A)} / {sum(df['group']==label_B)})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "emg_features_comparison.png", dpi=300)
    fig.savefig(outdir / "emg_features_comparison.pdf")

    print(f"\nDone. Outputs in {outdir.resolve()}:")
    print(f"  - emg_features.csv         (per-trial per-side features)")
    print(f"  - emg_features_stats.csv   (summary with Cohen's d, MW, KS)")
    print(f"  - emg_features_stats.txt   (human-readable table)")
    print(f"  - emg_features_comparison.png / .pdf  (supplementary figure)")


if __name__ == "__main__":
    main()
