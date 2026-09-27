"""Simulate one bilateral EMG pair and compute its left-right wavelet coherence.

Examples (run from the repository folder):
    python examples/simulate_demo.py
    python examples/simulate_demo.py --crosstalk 0.35 --mixture 50_50 --cortical-high 30 --seed 7

Outputs (in --outdir, default results/demo):
    demo_emg_1k.csv             time_s, left_emg, right_emg (bipolar, 1 kHz, 5-499 Hz, z-scored)
    demo_emg_timeseries.png     the two EMG signals
    demo_wavelet_coherence.png  time-frequency coherence map with cone of influence (COI)
    demo_band_medians.csv       median coherence per band (inside the COI)
The CSV can be fed directly to examples/analyze_emg_coherence.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions as F

MIXTURES = {"100_0": (6.0, 0.0), "75_25": (4.875, 1.625), "50_50": (3.25, 3.25)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--duration", type=float, default=18.0, help="seconds (default 18)")
    ap.add_argument("--crosstalk", type=float, default=0.20, help="right-to-left crosstalk, 0-1 (default 0.20)")
    ap.add_argument("--crosstalk-l2r", type=float, default=0.01, help="left-to-right crosstalk (default 0.01)")
    ap.add_argument("--mixture", choices=list(MIXTURES), default="75_25",
                    help="cortical/subcortical common-drive mixture (default 75_25)")
    ap.add_argument("--cortical-high", type=float, default=60, help="upper cortical band edge, Hz (30 or 60)")
    ap.add_argument("--outdir", type=Path, default=Path("results/demo"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    cw, sw = MIXTURES[a.mixture]
    print(f"Simulating {a.duration:g} s: crosstalk R->L={a.crosstalk:.2f}, mixture={a.mixture}, "
          f"cortical 13-{a.cortical_high:g} Hz, seed={a.seed}")
    left, right, fs = F.simulate_bilateral_emg(
        seed=a.seed, T_end=a.duration, cross_talk_R2L=a.crosstalk, cross_talk_L2R=a.crosstalk_l2r,
        common_mod_cortical=cw, common_mod_subcortical=sw, cortical_high_hz=a.cortical_high,
        bipolar=True, fs_out=1000.0)

    x = F.preprocess_emg_for_coherence(left, fs)
    y = F.preprocess_emg_for_coherence(right, fs)
    t = np.arange(x.size)/fs
    pd.DataFrame({"time_s": t, "left_emg": x, "right_emg": y}).to_csv(a.outdir/"demo_emg_1k.csv", index=False)

    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(t, x, lw=0.6); ax[0].set_ylabel("Left EMG (z)")
    ax[1].plot(t, y, lw=0.6, color="C1"); ax[1].set_ylabel("Right EMG (z)"); ax[1].set_xlabel("Time (s)")
    ax[0].set_title("Simulated bipolar EMG (5-499 Hz, z-scored)")
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(a.outdir/"demo_emg_timeseries.png", dpi=200); plt.close(fig)

    freqs, period, coi, Rsq, *_ = F.compute_wavelet_coherence(x, y, fs)
    F.plot_coherence_map(Rsq, freqs, period, coi, fs, a.outdir/"demo_wavelet_coherence.png")

    meds = F.band_medians(Rsq, freqs, period=period, coi=coi)
    pd.DataFrame([{"band_hz": k, "coherence_median": v} for k, v in meds.items()]).to_csv(
        a.outdir/"demo_band_medians.csv", index=False)
    print("Median coherence inside the COI:", {k: round(v, 3) for k, v in meds.items()})
    print("Saved outputs to", a.outdir.resolve())


if __name__ == "__main__":
    main()
