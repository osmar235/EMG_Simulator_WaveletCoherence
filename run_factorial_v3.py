"""Run the full Version 3.3 factorial EMG/coherence experiment.

Primary V3.3 analysis:
  * no full-record taper before CWT;
  * band medians exclude the cone of influence (COI);
  * same-condition cross-trial nulls are computed separately by
    ``trial_shuffle_null_v3.py`` from optionally saved per-cell signals.

Default grid matches the manuscript design:
  8 R->L crosstalk levels x 3 common-drive mixtures x 2 cortical bandwidths
  x 15 trials = 720 bilateral trials = 1,440 EMG signals.

The script writes one long-format row per trial x frequency band and checkpoints
at the end of every experimental cell. By default it also saves the untapered,
preprocessed 1-kHz left/right signals for the matched cross-trial null analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import functions as F

BANDS = ((5, 13), (13, 30), (30, 60), (60, 100))
MIXTURES = {
    "100_0": (6.0, 0.0),
    "75_25": (4.875, 1.625),
    "50_50": (3.25, 3.25),
}
CORTICAL_HIGHS = (30, 60)
CROSSTALK_R2L = tuple(np.round(np.arange(0.0, 0.351, 0.05), 2))


def deterministic_seed(base_seed: int, bw_i: int, mix_i: int, ct_i: int, trial: int) -> int:
    ss = np.random.SeedSequence([int(base_seed), bw_i, mix_i, ct_i, trial])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=15)
    ap.add_argument('--base-seed', type=int, default=20260926)
    ap.add_argument('--outdir', type=Path, default=Path('results/factorial_v3_3'))
    ap.add_argument('--taper', choices=['none', 'matviyenko', 'tukey'], default='none',
                    help='Wavelet-analysis taper only. V3.3 primary analysis = none; other choices are sensitivity analyses.')
    ap.add_argument('--matviyenko-k', type=int, default=8,
                    help='Matviyenko K (2..13), used only when --taper matviyenko. Historical K is unknown.')
    ap.add_argument('--no-save-null-signals', dest='save_null_signals', action='store_false',
                    help='Do not save per-cell untapered 1-kHz signals for trial_shuffle_null_v3.py.')
    ap.set_defaults(save_null_signals=True)
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args()


def _cell_stem(cortical_high, mix_name, ct):
    return f"bw{int(cortical_high):02d}_mix{mix_name}_ct{int(round(float(ct)*100)):02d}"


def main():
    args = parse_args()
    total = len(CORTICAL_HIGHS)*len(MIXTURES)*len(CROSSTALK_R2L)*args.trials
    if args.dry_run:
        print(f"V3.3 factorial plan: {total} bilateral trials = {2*total} EMG signals")
        print(f"Rows after four bands/trial: {4*total}")
        print(f"Primary taper: {args.taper}")
        print(f"Save matched-null signals: {args.save_null_signals}")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    null_dir = args.outdir/'null_signals'
    if args.save_null_signals:
        null_dir.mkdir(parents=True, exist_ok=True)

    taper = None if args.taper == 'none' else args.taper
    taper_kwargs = {'K': args.matviyenko_k} if taper == 'matviyenko' else None
    rows = []
    csv_path = args.outdir/'factorial_v3_3_band_medians.csv'

    meta = {
        'code_version': F.__version__,
        'analysis_status': 'V3.3 corrected primary' if args.taper == 'none' else 'V3.3 taper sensitivity',
        'trials_per_cell': args.trials,
        'base_seed': args.base_seed,
        'cortical_high_hz': list(CORTICAL_HIGHS),
        'crosstalk_R2L': list(CROSSTALK_R2L),
        'crosstalk_L2R': 0.01,
        'mixtures_coefficients_used_to_infer_fraction': {k: list(v) for k,v in MIXTURES.items()},
        'common_drive_normalization': 'rms',
        'common_drive_target_rms': 6.0,
        'common_fraction_interpretation': 'amplitude',
        'note_75_25': '75/25 denotes amplitude weighting; for independent equal-variance sources this is approximately 90/10 variance contribution.',
        'snr_units': 'db',
        'SNR_drive_db': 40,
        'SNR_MU_db': 8,
        'SNR_EMG_db': 40,
        'usable_duration_s': 18.0,
        'burn_in_s': 1.0,
        'intent_variability_rms_fraction': 0.01,
        'intent_mode': 'bounded',
        'coherence_preprocessing': {
            'bandpass_hz': [5, 499], 'detrend': True, 'taper': args.taper,
            'matviyenko_K': args.matviyenko_k if args.taper == 'matviyenko' else None,
            'primary_v3_3_taper': 'none', 'band_medians_exclude_COI': True,
        },
        'wavelet': {'mother': 'Morlet', 'w0': 6, 'dj': 1/8, 'fmax_hz': 128},
        'primary_null_inference': 'same-condition cross-trial derangement; run trial_shuffle_null_v3.py',
        'phase_randomization_status': 'legacy diagnostic only; not primary inference',
        'saved_null_signals': bool(args.save_null_signals),
    }
    (args.outdir/'factorial_v3_3_config.json').write_text(json.dumps(meta, indent=2))

    mix_items = list(MIXTURES.items())
    for bw_i, cortical_high in enumerate(CORTICAL_HIGHS):
        for mix_i, (mix_name, (cw, sw)) in enumerate(mix_items):
            for ct_i, ct in enumerate(CROSSTALK_R2L):
                print(f"bandwidth={cortical_high:2d} Hz  mix={mix_name:>6s}  R2L={ct:.2f}")
                cell_left, cell_right, cell_seeds = [], [], []
                for trial in range(args.trials):
                    seed = deterministic_seed(args.base_seed, bw_i, mix_i, ct_i, trial)
                    L, R, fs, info = F.generate_modulated_EMG_physiological_upgraded(
                        motorunits_max=72, n_fibers=346, FreqInputTotal=100,
                        SNR=40, SNRE=40, SNRF=8, T_end=18,
                        V_reset=-0.080, V_e=-0.075, Vth_min=-0.055, Vth_max=-0.040,
                        Rm=10e6, tau_m=10e-3, dt=0.0002, const_current=10.0,
                        freq_left=0.5, freq_right=1.0, phase_left=0.0, phase_right=np.pi/2,
                        window_size=0.1, Intent_variability=0.01,
                        cross_talk_R2L=float(ct), cross_talk_L2R=0.01,
                        common_mod_cortical=cw, common_mod_subcortical=sw,
                        I_scale=1.5e-9, cortical_high_hz=cortical_high,
                        muap_dur_s=0.018, muap_jitter_s=0.004,
                        use_dog_muaps=True, use_lif_adaptation=True, noise_type='pink',
                        seed=seed, snr_units='db', burn_in_s=1.0,
                        common_drive_normalization='rms', common_drive_target_rms=6.0,
                        common_fraction_interpretation='amplitude',
                        intent_drift='bounded', intent_cutoff_hz=0.25,
                        return_spikes=True)

                    bL = F.simulate_bipolar_emg_spatial(L, fs, tau_ms=1.2, lp_hz=180.0)
                    bR = F.simulate_bipolar_emg_spatial(R, fs, tau_ms=1.2, lp_hz=180.0)
                    fsw = 1000.0
                    bL = F.resample_to_fs(bL, fs, fsw)
                    bR = F.resample_to_fs(bR, fs, fsw)

                    # Base signal for reporting/nulls: filtered, detrended, z-scored, untapered.
                    x_base = F.preprocess_emg_for_coherence(bL, fsw, detrend=True, taper=None)
                    y_base = F.preprocess_emg_for_coherence(bR, fsw, detrend=True, taper=None)
                    if args.save_null_signals:
                        cell_left.append(x_base.astype(np.float32, copy=False))
                        cell_right.append(y_base.astype(np.float32, copy=False))
                        cell_seeds.append(seed)

                    # Primary V3.3: no taper. Sensitivity runs apply a taper only here.
                    x = F.apply_analysis_taper(x_base, taper=taper, taper_kwargs=taper_kwargs)
                    y = F.apply_analysis_taper(y_base, taper=taper, taper_kwargs=taper_kwargs)
                    freqs, period, coi, Rsq, *_ = F.compute_wavelet_coherence(
                        x, y, fsw, fmax=128.0, dj=1/8, w0=6)
                    meds = F.band_medians(Rsq, freqs, bands=BANDS, period=period, coi=coi)
                    trial_uid = f"bw{int(cortical_high)}_{mix_name}_ct{ct:.2f}_t{trial:02d}"
                    for (f0, f1), value in zip(BANDS, meds.values()):
                        rows.append({
                            'code_version': F.__version__, 'seed': seed, 'trial': trial,
                            'trial_uid': trial_uid,
                            'cortical_high_hz': cortical_high, 'mixture': mix_name,
                            'cortical_fraction_amplitude': info['common_cortical_fraction'],
                            'common_drive_rms': info['common_drive_rms'],
                            'crosstalk_R2L': ct, 'crosstalk_L2R': 0.01,
                            'analysis_taper': args.taper,
                            'matviyenko_K': args.matviyenko_k if args.taper == 'matviyenko' else np.nan,
                            'band_low_hz': f0, 'band_high_hz': f1,
                            'band': f'{f0}-{f1}', 'coherence_median': value,
                        })

                if args.save_null_signals:
                    stem = _cell_stem(cortical_high, mix_name, ct)
                    np.savez_compressed(
                        null_dir/f'{stem}.npz',
                        left=np.stack(cell_left), right=np.stack(cell_right),
                        seeds=np.asarray(cell_seeds, dtype=np.uint32), fs_hz=np.array([1000.0]),
                        cortical_high_hz=np.array([cortical_high]), mixture=np.array([mix_name]),
                        crosstalk_R2L=np.array([ct]), crosstalk_L2R=np.array([0.01]),
                        signal_definition=np.array(['5-499 Hz, detrended, z-scored, untapered']),
                    )

                # checkpoint after each experimental cell
                pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Saved {len(rows)} rows to {csv_path.resolve()}")
    if args.save_null_signals:
        print(f"Saved per-cell signals for cross-trial nulls to {null_dir.resolve()}")


if __name__ == '__main__':
    main()
