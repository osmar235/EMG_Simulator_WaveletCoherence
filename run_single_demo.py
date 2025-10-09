# run_single_demo.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

from functions import (
    generate_modulated_EMG_physiological_upgraded,
    simulate_bipolar_emg_spatial, band_medians,
    bandpass_emg, zscore, resample_to_fs,
    compute_wavelet_coherence, mc_null_threshold
)


def main():
    OUTDIR = Path("results"); OUTDIR.mkdir(parents=True, exist_ok=True)

    # -------- Simulate one pair --------
    EMG_L, EMG_R, fs_sim = generate_modulated_EMG_physiological_upgraded(
        motorunits_max=72, n_fibers=346, FreqInputTotal=100,
        SNR=40, SNRE=40, SNRF=8,T_end=18,
        V_reset=-0.080, V_e=-0.075, Vth_min=-0.055, Vth_max=-0.040,
        Rm=10e6, tau_m=10e-3,
        dt=0.0002, const_current=10.0, freq_left=0.5, freq_right=1.0, phase_left=0.0, phase_right=np.pi/2,
        window_size=0.1, Intent_variability=0.01,
        cross_talk_R2L=0.20, cross_talk_L2R=0.01,
        common_mod_cortical=3.25, common_mod_subcortical=3.25,
        I_scale=1.5e-9, cortical_high_hz=60,
        muap_dur_s=0.018, muap_jitter_s=0.004,
        use_dog_muaps=True, use_lif_adaptation=True, noise_type='pink', 
        shaper_taps_mu=None, shaper_taps_emg=None, seed=43,
          
    )

    """
    Simulate a bilateral EMG pair with cortical/subcortical common drive, optional
    cross-limb mixing, LIF motor units, MUAP summation, and realistic noise.

    Returns
    -------
    EMG_L, EMG_R, fs : np.ndarray, np.ndarray, float
        Left and right **monopolar** EMG time series (after trimming & EMG-level noise),
        and the source sampling rate fs (= 1/dt). Downstream examples use a global
        fractional-delay bipolar projection for realism.

    Core timing
    -----------
    dt : float
        Simulation time step (s). Use 0.0002 for 5 kHz internal rate.
    T_end : float
        Total simulated duration in seconds. (e.g., 18.0)

    Drive & task envelope
    ---------------------
    const_current : float
        Baseline random synaptic drive amplitude (A). Scaled low-pass noise.
    common_mod_cortical : float
        Amplitude of 13–cortical_high_hz Hz common drive (β/γ-like).
    common_mod_subcortical : float
        Amplitude of 5–13 Hz common drive (α/low-β-like).
    cortical_high_hz : float
        Upper cutoff for the “cortical” band (e.g., 30 or 60).
    freq_left, freq_right : float
        Target “force” modulation frequencies (Hz), typically 0.5 vs 1.0.
    phase_left, phase_right : float (rad)
        Phases to create 1:2 Lissajous (e.g., 0 and π/2).
    Intent_variability : float
        Slow linear drift within each window (adds slight ramping of the envelope).
    window_size : float
        Integration window (s) for the LIF loop (e.g., 0.1).

    Cross-limb mixing (“crosstalk” at the source level)
    ---------------------------------------------------
    cross_talk_R2L, cross_talk_L2R : float in [0,1]
        Proportion of the *other* side’s drive added into each limb (0.20 = 20%).
        These are nuisance knobs that emulate shared drive/volume conduction.

    LIF / motor unit pool
    ---------------------
    motorunits_max : int
        Number of simulated MUs per limb (e.g., 72).
    Vth_min, Vth_max : float (V)
        Range of static thresholds across the MU pool (ascending).
    V_reset, V_e : float (V)
        Reset and equilibrium potentials.
    Rm : float (Ohm), tau_m : float (s)
        Membrane resistance and time constant.
    I_scale : float
        Converts drive to current (A).
    use_lif_adaptation : bool
        Enables threshold after-spike adaptation.
    min_isi_ms : float
        Hard refractory (ms). Keeps MDRs in realistic range.
    dVth, tau_adapt_ms : float
        Adaptation step and decay (ms).

    MUAP synthesis & noise
    ----------------------
    use_dog_muaps : bool
        If True, use Difference-of-Gaussians MUAPs (smooth, realistic shapes).
        If False, uses the original biphasic template.
    muap_dur_s, muap_jitter_s : float
        Mean MUAP duration (s) and per-MU jitter (s).
    mu_gain_low, mu_gain_high : float
        Linear amplitude ramp across the MU pool (low→high).
    noise_type : {'pink','shaped','white'}
        MU-level and EMG-level noise coloring. For 'shaped', provide FIR taps via
        shaper_taps_mu / shaper_taps_emg (see `colored_noise_from_shaper`).
    SNR, SNRF, SNRE : float
        SNRs for drive noise, MUAP-level noise, and EMG-level noise, respectively.

    Reproducibility
    ---------------
    seed : int
        Master RNG seed.

    Typical starting point
    ----------------------
    motorunits_max=72, n_fibers=346, FreqInputTotal=100,
    SNR=40, SNRE=40, SNRF=8,
    V_reset=-0.080, V_e=-0.075, Vth_min=-0.055, Vth_max=-0.040,
    Rm=10e6, tau_m=10e-3, dt=0.0002, T_end=18.0,
    freq_left=0.5, freq_right=1.0, phase_left=0.0, phase_right=np.pi/2,
    Intent_variability≈0.01, window_size=0.1,
    cross_talk_R2L≈0.20, cross_talk_L2R≈0.01,
    common_mod_cortical≈3.25, common_mod_subcortical≈3.25,
    I_scale≈1.5e-9, cortical_high_hz=60, use_dog_muaps=True.

    Quick experiments
    -----------------
    • Make β/low-γ stronger: increase `common_mod_cortical` (and/or set `cortical_high_hz=60`).
    • Make α stronger: increase `common_mod_subcortical`.
    • Increase bilateral coupling: raise `cross_talk_R2L` (and/or `cross_talk_L2R`).
    • Smoother envelopes: increase `window_size` or lower `FreqInputTotal`.
    • More “spiky” EMG: reduce SNRs (SNRF or SNRE) or shorten `muap_dur_s`.

    Notes
    -----
    Output is **monopolar** EMG. For sensors, use
    `simulate_bipolar_emg_spatial(x, fs, tau_ms=1.2, lp_hz=180)`.
    """

    # Bipolar projection
    bL = simulate_bipolar_emg_spatial(EMG_L, fs_sim, tau_ms=1.2, lp_hz=180.0)
    bR = simulate_bipolar_emg_spatial(EMG_R, fs_sim, tau_ms=1.2, lp_hz=180.0)

    # Resample → 1 kHz, band-pass, z-score
    fs_wavelet = 1000.0
    bL_rs = resample_to_fs(bL, fs_sim, fs_wavelet)
    bR_rs = resample_to_fs(bR, fs_sim, fs_wavelet)
    bL_bp = zscore(bandpass_emg(bL_rs, fs_wavelet))
    bR_bp = zscore(bandpass_emg(bR_rs, fs_wavelet))

    # ---- CSV: bipolar, band-pass, z-scored @ 1 kHz ----
    t_1k = np.arange(bL_bp.size) / fs_wavelet
    df_bip_1k = pd.DataFrame(
        {"time_s": t_1k, "left_emg": bL_bp.astype(float), "right_emg": bR_bp.astype(float)}
    )
    p_csv_bip = OUTDIR / "single_demo_emg_bipolar_1k.csv"
    df_bip_1k.to_csv(p_csv_bip, index=False)
    print(f"Saved: {p_csv_bip.resolve()}  (fs=1000.0 Hz)")

    # -------- Figure A: EMG time series (stacked) --------
    t = np.arange(bL_bp.size)/fs_wavelet
    fig_ts, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax1.plot(t, bL_bp, lw=1); ax1.set_ylabel("Left EMG (z)")
    ax1.set_title("Bipolar EMG (band-pass 5–499 Hz; z-scored)")
    ax2.plot(t, bR_bp, lw=1, color='C1'); ax2.set_ylabel("Right EMG (z)")
    ax2.set_xlabel("Time (s)")
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig_ts.tight_layout()
    fig_ts.savefig(OUTDIR/"single_demo_emg_timeseries.png", dpi=300)

    # -------- Wavelet coherence (data) --------
    freqs, period, coi, Rsq, dt_w, dj, s0, J, mother = compute_wavelet_coherence(
        bL_bp, bR_bp, fs=fs_wavelet, fmax=128.0, dj=1/8, s0=None, J=None, w0=6
    )
    # build time axis for Rsq
    nT = Rsq.shape[1]
    t_w = np.arange(nT)/fs_wavelet

    # -------- MC null (progress bar inside) --------
    freqs_u, upper = mc_null_threshold(
        sig_len=bL_bp.size, dt=dt_w, dj=dj, s0=s0, J=J, mother=mother,
        alpha=0.05, num_simulations=2,    # tune for speed/precision
        bp_low=5.0, bp_high=499.0, fmax=128.0
    )
    # safety: grids must match
    if not np.allclose(freqs, freqs_u):
        # reconcile by intersecting (paranoid safety)
        f_common = np.intersect1d(np.round(freqs, 10), np.round(freqs_u, 10))
        keep_f   = np.isin(np.round(freqs, 10),   f_common)
        keep_fu  = np.isin(np.round(freqs_u, 10), f_common)
        freqs = freqs[keep_f]; period = period[keep_f]
        Rsq   = Rsq[keep_f, :]
        upper = upper[keep_fu]
    assert upper.shape[0] == freqs.shape[0]

    # -------- Pretty contour with smooth significance --------
    fig_wc, ax = plt.subplots(figsize=(12, 6))
    levels = np.linspace(0, 1, 100)
    cf = ax.contourf(
        t_w, np.log2(period), Rsq,
        levels=levels, cmap='jet', vmin=0, vmax=1,
        antialiased=True, extend='both'
    )
    # single 0-level contour of Rsq - per-freq threshold
    sig_map = Rsq - upper[:, None]
    ax.contour(t_w, np.log2(period), sig_map, levels=[0],
               colors='black', linewidths=2, antialiased=True)

    # COI
    ax.plot(t_w, np.log2(coi), 'k', lw=3, label='COI')

    # Axis formatting
    # ---- Frequency axis that matches the computed grid ----
    f_lo = 4.0                       # bottom of plot (Hz)
    f_hi = float(freqs.max())        # top available from the CWT (≈ fmax)
    nice = [5, 13, 30, 60, 100]
    f_hi_nice = max(v for v in nice if v <= f_hi)

    ax.set_ylim(np.log2([1.0/f_lo, 1.0/f_hi_nice]))
    #ax.invert_yaxis()

    tick_candidates = [100,60,30,13,5]
    ticks = [f for f in tick_candidates if f_lo <= f <= f_hi_nice]
    ax.set_yticks(np.log2(1.0/np.array(ticks)))
    ax.set_yticklabels(ticks, fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Frequency (Hz)', fontsize=14)
    ax.set_title('Wavelet Coherence Spectrum', fontsize=16)

    cbar = fig_wc.colorbar(cf, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Coherence (0–1)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(np.linspace(0, 1, 5))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.legend(loc='upper right', fontsize=10)
    fig_wc.tight_layout()
    fig_wc.savefig(OUTDIR/"single_demo_wavelet_coherence.png", dpi=300)
    print("Saved plots to:", OUTDIR.resolve())

    meds = band_medians(Rsq, freqs)  # dict with '5-13','13-30','30-60','60-100'
    print("Band medians:", meds)

    # quick preview figure (time-averaged spectrum and band marks)
    Rsq_mean = Rsq.mean(axis=1)
    plt.figure(figsize=(7,4))
    plt.plot(freqs, Rsq_mean, lw=2)
    for f0,f1 in ((5,13),(13,30),(30,60),(60,100)):
        plt.axvspan(f0, f1, alpha=0.1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Wavelet coherence (avg over time)")
    plt.title("Sim-only wavelet coherence (time-avg)")
    plt.xlim(4, 100)
    plt.tight_layout()
    out_png = OUTDIR / "single_demo_coh_avg.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
