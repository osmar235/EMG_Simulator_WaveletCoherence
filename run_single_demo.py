"""Single-trial Version 3.3 demo.

Key V3.3 reporting rule:
- the saved/plotted bipolar EMG is NEVER Matviyenko-tapered;
- tapering is applied only to an analysis copy immediately before the CWT;
- the wavelet map shows the cone of influence (COI) but no default pixelwise
  significance contour. Optional phase-randomized surrogates are retained only as a legacy diagnostic;
  V3.3 primary null inference is same-condition cross-trial pairing and requires
  multiple trials (see trial_shuffle_null_v3.py).

Requires pycwt for the coherence part.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import functions
from functions import (
    generate_modulated_EMG_physiological_upgraded,
    simulate_bipolar_emg_spatial,
    resample_to_fs,
    preprocess_emg_for_coherence,
    compute_wavelet_coherence,
    band_medians,
    surrogate_wavelet_threshold,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surrogates', type=int, default=0,
                    help='LEGACY phase-randomized frequency-median diagnostic only. V3.3 primary null inference uses trial_shuffle_null_v3.py.')
    ap.add_argument('--taper', choices=['none', 'matviyenko', 'tukey'], default='none',
                    help='Wavelet-analysis taper only. V3.3 primary = none; Matviyenko/Tukey are sensitivity analyses.')
    ap.add_argument('--matviyenko-k', type=int, default=8,
                    help='Matviyenko K (2..13), used only for sensitivity analysis; historical K is unknown.')
    ap.add_argument('--outdir', type=Path, default=Path('results'))
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"EMG simulator functions.py v{functions.__version__}")
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    taper = None if args.taper == 'none' else args.taper
    taper_kwargs = {'K': args.matviyenko_k} if taper == 'matviyenko' else None

    # Paper-like representative condition: broad cortical bandwidth, 75/25 mix,
    # R->L 20%, L->R 1%. In V3 the combined common drive has RMS=6 for every mix.
    EMG_L, EMG_R, fs_sim, info = generate_modulated_EMG_physiological_upgraded(
        motorunits_max=72, n_fibers=346, FreqInputTotal=100,
        SNR=40, SNRE=40, SNRF=8, T_end=18,
        V_reset=-0.080, V_e=-0.075, Vth_min=-0.055, Vth_max=-0.040,
        Rm=10e6, tau_m=10e-3, dt=0.0002,
        const_current=10.0, freq_left=0.5, freq_right=1.0,
        phase_left=0.0, phase_right=np.pi/2, window_size=0.1,
        Intent_variability=0.01,
        cross_talk_R2L=0.20, cross_talk_L2R=0.01,
        common_mod_cortical=4.875, common_mod_subcortical=1.625,
        I_scale=1.5e-9, cortical_high_hz=60,
        muap_dur_s=0.018, muap_jitter_s=0.004,
        use_dog_muaps=True, use_lif_adaptation=True,
        noise_type='pink', seed=43,
        snr_units='db', burn_in_s=1.0,
        common_drive_normalization='rms', common_drive_target_rms=6.0,
        common_fraction_interpretation='amplitude',
        intent_drift='bounded', intent_cutoff_hz=0.25,
        return_spikes=True,
    )

    # Surface bipolar projection and resampling.
    bL = simulate_bipolar_emg_spatial(EMG_L, fs_sim, tau_ms=1.2, lp_hz=180.0)
    bR = simulate_bipolar_emg_spatial(EMG_R, fs_sim, tau_ms=1.2, lp_hz=180.0)
    fs_wavelet = 1000.0
    bL = resample_to_fs(bL, fs_sim, fs_wavelet)
    bR = resample_to_fs(bR, fs_sim, fs_wavelet)

    # IMPORTANT: create the physiological/reporting signal WITHOUT taper.
    # This is the signal that is exported and plotted in the time domain.
    x_emg = preprocess_emg_for_coherence(
        bL, fs_wavelet, detrend=True, taper=None)
    y_emg = preprocess_emg_for_coherence(
        bR, fs_wavelet, detrend=True, taper=None)

    # Create a separate analysis copy for CWT. Matviyenko (or other taper) is
    # applied only here, immediately before wavelet analysis.
    x_wave = preprocess_emg_for_coherence(
        bL, fs_wavelet, detrend=True, taper=taper, taper_kwargs=taper_kwargs)
    y_wave = preprocess_emg_for_coherence(
        bR, fs_wavelet, detrend=True, taper=taper, taper_kwargs=taper_kwargs)

    t = np.arange(x_emg.size)/fs_wavelet
    pd.DataFrame({'time_s': t, 'left_emg': x_emg, 'right_emg': y_emg}).to_csv(
        outdir/'single_demo_emg_bipolar_1k.csv', index=False)

    # Keep the actual CWT input for audit/debugging under an explicit filename.
    pd.DataFrame({'time_s': t, 'left_wavelet_input': x_wave,
                  'right_wavelet_input': y_wave}).to_csv(
        outdir/'single_demo_wavelet_input_1k.csv', index=False)

    meta = {
        'code_version': functions.__version__,
        'fs_sim_hz': fs_sim,
        'fs_wavelet_hz': fs_wavelet,
        'usable_duration_s': info['duration_s'],
        'burn_in_s': info['burn_in_s'],
        'common_drive_rms': info['common_drive_rms'],
        'common_cortical_fraction': info['common_cortical_fraction'],
        'common_fraction_interpretation': info['common_fraction_interpretation'],
        'snr_units': info['snr_units'],
        'drive_noise_snr_db_L': info['drive_noise_snr_db_L'],
        'drive_noise_snr_db_R': info['drive_noise_snr_db_R'],
        'intent_drift': info['intent_drift'],
        'time_domain_export': '5-499 Hz, detrended, z-scored, UNTAPERED',
        'time_domain_taper': 'none',
        'wavelet_input_detrend': True,
        'wavelet_input_taper': args.taper,
        'matviyenko_K': args.matviyenko_k if args.taper == 'matviyenko' else None,
        'matviyenko_K_status': 'sensitivity parameter; historical K not recoverable from supplied MATLAB source' if args.taper == 'matviyenko' else None,
        'wavelet_map_significance_contour': 'none',
        'surrogate_method': 'LEGACY independent phase randomization' if args.surrogates > 0 else None,
        'surrogate_statistic': 'frequency_median diagnostic only' if args.surrogates > 0 else None,
        'n_surrogates': args.surrogates,
    }
    (outdir/'single_demo_metadata.json').write_text(json.dumps(meta, indent=2))

    # Time-domain figure: NEVER plot the tapered CWT input here.
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(t, x_emg, lw=0.8); ax[0].set_ylabel('Left EMG (z)')
    ax[1].plot(t, y_emg, lw=0.8); ax[1].set_ylabel('Right EMG (z)'); ax[1].set_xlabel('Time (s)')
    ax[0].set_title('V3.3 bipolar EMG: 5-499 Hz, detrended, z-scored (untapered)')
    for a in ax:
        a.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir/'single_demo_emg_timeseries.png', dpi=300)
    plt.close(fig)

    # Wavelet coherence is computed from the analysis copy (optionally tapered).
    freqs, period, coi, Rsq, *_ = compute_wavelet_coherence(
        x_wave, y_wave, fs=fs_wavelet, fmax=128.0, dj=1/8, w0=6)

    tw = np.arange(Rsq.shape[1])/fs_wavelet
    fig, ax = plt.subplots(figsize=(12, 6))
    levels = np.linspace(0, 1, 101)
    cf = ax.contourf(tw, np.log2(period), Rsq, levels=levels, cmap='jet', vmin=0, vmax=1)
    ax.plot(tw, np.log2(coi), 'k', lw=2, label='COI')
    ticks = np.array([100, 60, 30, 13, 5])
    ax.set_yticks(np.log2(1.0/ticks)); ax.set_yticklabels(ticks)
    ax.set_ylim(np.log2(1/4.0), np.log2(1/100.0))
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
    taper_title = f'Matviyenko K={args.matviyenko_k} sensitivity' if args.taper == 'matviyenko' else ('no taper (V3.3 primary)' if args.taper == 'none' else args.taper)
    ax.set_title(f'Wavelet coherence ({taper_title}; COI shown; no pixelwise significance contour)')
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Squared wavelet coherence')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(outdir/'single_demo_wavelet_coherence.png', dpi=300)
    plt.close(fig)

    meds = band_medians(Rsq, freqs, period=period, coi=coi)
    print('Band medians inside COI:', {k: round(v, 4) for k,v in meds.items()})

    # LEGACY phase-surrogate diagnostic only. We do NOT draw these thresholds as
    # pixelwise map significance contours. V3.3 primary null inference is the
    # same-condition cross-trial procedure in trial_shuffle_null_v3.py.
    if args.surrogates > 0:
        sf, freq_median_thr, smeta = surrogate_wavelet_threshold(
            x_wave, y_wave, fs=fs_wavelet, alpha=0.05,
            num_simulations=args.surrogates, mode='frequency_median',
            fmax=128.0, dj=1/8,
            rng=np.random.default_rng(20260926), progress=True)
        if not np.allclose(freqs, sf):
            raise RuntimeError('surrogate and data frequency grids differ')
        pd.DataFrame({
            'frequency_hz': sf,
            'frequency_median_coherence_95pct_null': freq_median_thr,
        }).to_csv(outdir/'single_demo_frequency_median_surrogate_threshold.csv', index=False)
        print('Saved LEGACY phase-surrogate diagnostic (not V3.3 primary null inference).')

    print('Saved untapered EMG to:', (outdir/'single_demo_emg_bipolar_1k.csv').resolve())
    print('Saved wavelet-analysis input to:', (outdir/'single_demo_wavelet_input_1k.csv').resolve())
    print('Saved outputs to:', outdir.resolve())


if __name__ == '__main__':
    main()
