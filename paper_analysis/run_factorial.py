"""Factorial simulation study of the article.

Design: 8 right-to-left crosstalk levels (0-35 %) x 3 common-drive mixtures
(100/0, 75/25, 50/50 cortical/subcortical) x 2 cortical bandwidths (13-30, 13-60 Hz)
x 15 trials = 720 bilateral trials (1,440 EMG signals). Left-to-right crosstalk 1 %.
For each trial: bipolar EMG -> 1 kHz -> 5-499 Hz, detrend, z-score -> Morlet wavelet
coherence -> median coherence inside the cone of influence in 5-13, 13-30, 30-60
and 60-100 Hz.

Usage (from the repository folder):
    python paper_analysis/run_factorial.py --dry-run
    python paper_analysis/run_factorial.py --trials 2 --outdir results/factorial_test     # quick test
    python paper_analysis/run_factorial.py                                                # full study

Outputs (in --outdir, default results/factorial):
    factorial_band_medians.csv   one row per trial x band
    factorial_config.json        settings
    trial_signals/*.npz          preprocessed 1-kHz signals per condition (for the
                                 cross-trial null; disable with --no-save-signals)
Next: python paper_analysis/cross_trial_null_factorial.py results/factorial
      Rscript paper_analysis/art_analysis.R results/factorial/factorial_band_medians.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions as F

BANDS = F.DEFAULT_BANDS
MIXTURES = {"100_0": (6.0, 0.0), "75_25": (4.875, 1.625), "50_50": (3.25, 3.25)}
CORTICAL_HIGHS = (30, 60)
CROSSTALK_R2L = tuple(np.round(np.arange(0.0, 0.351, 0.05), 2))
FS_ANALYSIS = 1000.0


def trial_seed(base_seed, bw_i, mix_i, ct_i, trial):
    ss = np.random.SeedSequence([int(base_seed), bw_i, mix_i, ct_i, trial])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def cell_name(bw, mix, ct):
    return f"bw{int(bw):02d}_mix{mix}_ct{int(round(ct*100)):02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--base-seed", type=int, default=20260926)
    ap.add_argument("--outdir", type=Path, default=Path("results/factorial"))
    ap.add_argument("--no-save-signals", dest="save_signals", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    n_cells = len(CORTICAL_HIGHS)*len(MIXTURES)*len(CROSSTALK_R2L)
    if a.dry_run:
        print(f"{n_cells} conditions x {a.trials} trials = {n_cells*a.trials} bilateral trials "
              f"({2*n_cells*a.trials} EMG signals), {4*n_cells*a.trials} output rows")
        return
    a.outdir.mkdir(parents=True, exist_ok=True)
    sig_dir = a.outdir/"trial_signals"
    if a.save_signals:
        sig_dir.mkdir(exist_ok=True)
    config = dict(code_version=F.__version__, trials_per_condition=a.trials, base_seed=a.base_seed,
                  cortical_high_hz=list(CORTICAL_HIGHS), crosstalk_R2L=list(CROSSTALK_R2L),
                  crosstalk_L2R=0.01, mixtures=MIXTURES, common_drive_rms=6.0,
                  duration_s=18.0, burn_in_s=1.0, snr_db=dict(drive=40, motor_unit=8, emg=40),
                  analysis=dict(fs_hz=FS_ANALYSIS, bandpass_hz=[5, 499], detrend=True, taper=None,
                                wavelet="Morlet w0=6, dj=1/8", fmax_hz=128, bands_hz=BANDS,
                                summary="median inside cone of influence"))
    (a.outdir/"factorial_config.json").write_text(json.dumps(config, indent=2))

    rows, t0, done = [], time.time(), 0
    csv_path = a.outdir/"factorial_band_medians.csv"
    for bw_i, bw in enumerate(CORTICAL_HIGHS):
        for mix_i, (mix, (cw, sw)) in enumerate(MIXTURES.items()):
            for ct_i, ct in enumerate(CROSSTALK_R2L):
                Ls, Rs, seeds = [], [], []
                for trial in range(a.trials):
                    seed = trial_seed(a.base_seed, bw_i, mix_i, ct_i, trial)
                    L, R, fs = F.simulate_bilateral_emg(
                        seed=seed, T_end=18.0, cross_talk_R2L=float(ct), cross_talk_L2R=0.01,
                        common_mod_cortical=cw, common_mod_subcortical=sw, cortical_high_hz=bw,
                        bipolar=True, fs_out=FS_ANALYSIS)
                    x = F.preprocess_emg_for_coherence(L, fs)
                    y = F.preprocess_emg_for_coherence(R, fs)
                    freqs, period, coi, Rsq, *_ = F.compute_wavelet_coherence(x, y, fs)
                    for band, v in F.band_medians(Rsq, freqs, BANDS, period=period, coi=coi).items():
                        rows.append(dict(trial_id=f"{cell_name(bw, mix, ct)}_t{trial:02d}", trial=trial,
                                         seed=seed, cortical_high_hz=bw, mixture=mix, crosstalk_R2L=ct,
                                         crosstalk_L2R=0.01, band=band, coherence_median=v))
                    if a.save_signals:
                        Ls.append(x.astype(np.float32)); Rs.append(y.astype(np.float32)); seeds.append(seed)
                done += 1
                pd.DataFrame(rows).to_csv(csv_path, index=False)   # checkpoint
                if a.save_signals:
                    np.savez_compressed(sig_dir/f"{cell_name(bw, mix, ct)}.npz", left=np.stack(Ls),
                                        right=np.stack(Rs), seeds=np.array(seeds, np.uint32),
                                        fs_hz=FS_ANALYSIS, cortical_high_hz=bw, mixture=mix,
                                        crosstalk_R2L=ct, crosstalk_L2R=0.01)
                el = time.time() - t0
                print(f"[{done}/{n_cells}] cortical 13-{bw} Hz, mixture {mix}, R->L {ct:.2f}  "
                      f"({el/60:.1f} min elapsed, ~{el/done*(n_cells-done)/60:.1f} min left)")
    print("Saved", csv_path.resolve())


if __name__ == "__main__":
    main()
