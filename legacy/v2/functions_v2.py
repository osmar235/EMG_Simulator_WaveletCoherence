# functions.py  —  EMG Simulator – Wavelet Coherence
#
# Version 2.0.0 (bug-fix release). See CHANGELOG.md for the full list of changes.
# The exact code used for the published CMPB analyses is kept, unmodified, in
# legacy/functions_v1.py so that the published numbers remain reproducible.
#
# Summary of fixes relative to v1:
#   1. smoothwavelet(): the scale-smoothing boxcar was applied along TIME
#      (kernel[np.newaxis, :]) instead of along SCALE (kernel[:, np.newaxis]).
#   2. smoothwavelet(): np.real() discarded the imaginary part of the smoothed
#      cross-spectrum Wxy, so coherence ignored quadrature (phase-lagged) coupling.
#   3. Generator: the motor-neuron state (adaptive threshold, refractory clock and,
#      for the right limb, the membrane voltage) was reset every `window_size`
#      window, synchronising all motor units of BOTH limbs at 1/window_size Hz.
#   4. Generator: the unique drive, drive noise, MU-level noise and fibre jitter
#      were generated per window (discontinuities every window); MUAPs were
#      convolved per window (truncated at window edges); MUAP durations were drawn
#      per window from the unseeded global np.random; the master RNG was
#      overwritten inside the loop; mu_gain_low/high were ignored.
#   5. Seeds: every random draw now comes from the `seed` argument (bit-for-bit
#      reproducible runs).
#
# Public function names and signatures are unchanged; new arguments are optional.

__version__ = "2.0.0"

import warnings
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401  (kept for backward compatibility)
from fractions import Fraction
from typing import Tuple  # noqa: F401
from tqdm.auto import tqdm

from scipy.signal import (  # noqa: F401
    butter, filtfilt, resample_poly, welch, firls, convolve as sp_convolve
)
from scipy.signal import convolve2d
import pycwt as wavelet

# ------------------------------ Basics ------------------------------

def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def band_medians(Rsq, freqs, bands=((5, 13), (13, 30), (30, 60), (60, 100)),
                 period=None, coi=None):
    """Median squared wavelet coherence within each frequency band.

    Rsq has shape (n_frequencies, n_times). When `period` and `coi` are both
    given (as returned by compute_wavelet_coherence), time-frequency samples
    outside the cone of influence are excluded (recommended). Without them the
    behaviour is identical to v1 (all samples used).
    """
    Rsq = np.asarray(Rsq, float)
    freqs = np.asarray(freqs, float)
    if Rsq.ndim != 2 or Rsq.shape[0] != freqs.size:
        raise ValueError("Rsq must have shape (len(freqs), n_times)")
    use_coi = period is not None and coi is not None
    if use_coi:
        period = np.asarray(period, float)
        coi = np.asarray(coi, float)
        if period.size != freqs.size or coi.size != Rsq.shape[1]:
            raise ValueError("period/coi dimensions do not match Rsq")
    out = {}
    for (f0, f1) in bands:
        idx = np.where((freqs >= f0) & (freqs < f1))[0]
        if idx.size == 0:
            out[f"{f0}-{f1}"] = np.nan
            continue
        values = Rsq[idx, :]
        values = values[period[idx, None] <= coi[None, :]] if use_coi else values.ravel()
        values = values[np.isfinite(values)]
        out[f"{f0}-{f1}"] = float(np.median(values)) if values.size else np.nan
    return out


def butter_filter_low(x, fc, fs, order=4):
    b, a = butter(order, fc/(fs/2), btype="low")
    return filtfilt(b, a, x)


def butter_filter_high(x, fc, fs, order=4):
    b, a = butter(order, fc/(fs/2), btype="high")
    return filtfilt(b, a, x)


def bandpass_emg(x, fs, low=5, high=499, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, x)


def resample_to_fs(x, fs_in, fs_out, max_den=10000):
    if np.isclose(fs_in, fs_out, rtol=1e-8, atol=1e-12):
        return np.asarray(x, float)
    frac = Fraction(fs_out/fs_in).limit_denominator(max_den)
    return resample_poly(np.asarray(x, float), frac.numerator, frac.denominator)

# --------------------------- Bipolar projection ---------------------------

def _frac_delay(x, D, N=21, pad_mode="reflect"):
    x = np.asarray(x, float)
    n = np.arange(N)
    h = np.sinc(n - (N-1)/2 - D) * np.hamming(N)
    h /= h.sum()
    pad = (N-1)//2 + int(np.ceil(abs(D)))
    xpad = np.pad(x, (pad, pad), mode=pad_mode) if pad > 0 else x
    ypad = np.convolve(xpad, h, mode="same")
    return ypad[pad:-pad] if pad > 0 else ypad


def simulate_bipolar_emg_spatial(x, fs, tau_ms=1.2, lp_hz=180, wA=1.0, wB=1.0, fir_len=21):
    D = tau_ms*1e-3*fs
    chA = x
    chB = _frac_delay(x, D, N=fir_len, pad_mode="reflect")
    y = wA*chA - wB*chB
    if lp_hz is not None:
        b, a = butter(4, lp_hz/(fs/2), btype="low")
        y = filtfilt(b, a, y, padtype="odd", padlen=3*(max(len(a), len(b))-1))
    return y

# ----------------------------- Noise helpers -----------------------------

def pinkish_noise_fft(N, fs, alpha=1.0, f_lo=10, f_hi=400, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    w = rng.standard_normal(N)
    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    shape = np.ones_like(freqs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    shape[~mask] = 0.0
    shape[mask] = np.maximum(freqs[mask], 1e-3)**(-alpha/2.0)
    y = np.fft.irfft(W*shape, n=N)
    return zscore(y)


def colored_noise_from_shaper(N, shaper_taps, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    w = rng.standard_normal(N + len(shaper_taps) - 1)
    y = np.convolve(w, shaper_taps, mode='valid')[:N]
    return zscore(y)


def _noise(N, fs, noise_type, rng, shaper_taps=None):
    """Unit-variance noise of the requested colour, drawn from `rng`."""
    if noise_type == 'pink':
        return pinkish_noise_fft(N, fs, alpha=1.2, f_lo=10, f_hi=450, rng=rng)
    if noise_type == 'shaped' and shaper_taps is not None:
        return colored_noise_from_shaper(N, shaper_taps, rng=rng)
    return zscore(rng.standard_normal(N))

# --------------------------- MUAP templates ------------------------------

def generate_muap_shape_dog(dur_s, fs, width_scale=1.0, tri_ratio=0.35, mu_factor=0.0):
    L = max(16, int(round(dur_s*fs)))
    t = np.linspace(-0.5, 0.5, L)
    s1 = 0.11*width_scale*(1 - 0.25*mu_factor)
    s2 = 0.20*width_scale*(1 - 0.15*mu_factor)
    g1 = np.exp(-0.5*(t/s1)**2); g2 = np.exp(-0.5*(t/s2)**2)
    dog = g1 - 0.65*g2
    if tri_ratio > 0:
        s3 = 0.28 * width_scale
        g3 = np.exp(-0.5*(t/s3)**2)
        dog = dog + tri_ratio*(g3 - g3.mean())
    dog -= dog.mean()
    dog /= (np.max(np.abs(dog)) + 1e-12)
    return dog.astype(float)


_generate_muap_shape_dog = generate_muap_shape_dog  # v1 alias (identical code)


def generate_muap_shape_original(d, dt, tau=0.18):
    n = max(2, int(round((d/2)/dt)))
    t = np.linspace(0, d/2, n)
    shape1 = 5*np.sin(np.pi*t/(d/2)) * np.exp((1/tau)*((t/(d/2))-1))
    shape2 = -np.flip(shape1)[1:]
    return np.concatenate([shape1, shape2])


def force_modulation_sine(t, f, phase=0):
    return (np.sin(2*np.pi*f*t + phase) + 1.0)/2.0

# ------------------------ Leaky integrate-and-fire ------------------------

def lif_population(Im, group, Vth_base, V_reset, V_e, Rm, tau_m, dt,
                   state=None, adaptation=True,
                   min_isi_ms=18.0, dVth=0.003, tau_adapt_ms=250.0):
    """Vectorised LIF motor-neuron pool with after-spike threshold adaptation.

    Parameters
    ----------
    Im : array (n_groups, L)
        Input current of each group (e.g. one row per limb).
    group : int array (n_units,)
        Row of `Im` that drives each unit.
    Vth_base : array (n_units,)
        Static threshold of each unit (V).
    state : dict or None
        {'V', 'Vth', 'since'} carried from a previous call (continuous
        integration across chunks). None = fresh start (V = V_reset, no
        adaptation, refractory clock expired).

    Update rule per sample t (identical to v1's lif_window_with_adaptation):
        Vth <- Vth_base + (Vth - Vth_base) * exp(-dt/tau_adapt)
        if t - t_last < min_isi : V[t+1] = V_reset                 (refractory)
        elif V[t] >= Vth        : spike at t, V[t+1] = V_reset, Vth += dVth
        else                    : V[t+1] = V[t] + dt*(-(V[t]-V_e) + Im[t]*Rm)/tau_m
    With adaptation=False the refractory period and the threshold step are
    disabled (v1's non-adaptive branch).

    Returns
    -------
    spikes : bool array (n_units, L)
    state  : dict, to pass to the next call for seamless continuation.
    """
    Im = np.atleast_2d(np.asarray(Im, float))
    group = np.asarray(group, int)
    Vb = np.asarray(Vth_base, float)
    n, L = Vb.size, Im.shape[1]
    if state is None:
        V = np.full(n, float(V_reset)); Vth = Vb.copy(); since = np.full(n, 10**9, dtype=np.int64)
    else:
        V = np.asarray(state['V'], float).copy(); Vth = np.asarray(state['Vth'], float).copy()
        since = np.asarray(state['since'], np.int64).copy()
    if adaptation:
        min_isi = int(round(min_isi_ms/1000.0/dt))
        decay = np.exp(-dt/max(tau_adapt_ms/1000.0, 1e-6))
        step = dVth
    else:
        min_isi, decay, step = 0, 0.0, 0.0
        Vth = Vb.copy()
    last = -since                              # index of the last spike relative to t = 0
    k = dt/tau_m
    ImT = np.ascontiguousarray(Im.T)           # (L, n_groups): fast row access
    spikes = np.zeros((n, L), dtype=bool)
    for t in range(L):
        Vth = Vb + (Vth - Vb)*decay
        refr = (t - last) < min_isi
        fire = (~refr) & (V >= Vth)
        Vn = V + k*(-(V - V_e) + ImT[t][group]*Rm)
        Vn[refr | fire] = V_reset
        if fire.any():
            spikes[fire, t] = True
            last[fire] = t
            Vth[fire] += step
        V = Vn
    return spikes, {'V': V, 'Vth': Vth, 'since': L - last}


def lif_window_with_adaptation(Im, Vth_base, V_reset, V_e, Rm, tau_m, dt, v_mem_start,
                               min_isi_ms=18.0, dVth=0.003, tau_adapt_ms=250.0,
                               state=None, return_state=False):
    """Single-unit LIF with threshold adaptation (backward-compatible wrapper).

    v1 restarted the adaptive threshold and refractory clock on every call; the
    v1 generator called it once per 100-ms window, which produced the window
    artifact. Pass the `state` returned with return_state=True to continue a
    unit seamlessly across calls. Returns (spikes, V_end) or
    (spikes, V_end, state).
    """
    if state is None:
        state = {'V': np.array([v_mem_start], float), 'Vth': np.array([Vth_base], float),
                 'since': np.array([10**9])}
    sp, st = lif_population(np.asarray(Im, float)[None, :], np.array([0]), np.array([Vth_base], float),
                            V_reset, V_e, Rm, tau_m, dt, state=state, adaptation=True,
                            min_isi_ms=min_isi_ms, dVth=dVth, tau_adapt_ms=tau_adapt_ms)
    out = (sp[0].astype(float), float(st['V'][0]))
    return out + (st,) if return_state else out

# ------------------------------ Fibre dispersion -------------------------

def simulate_fiber_emg_vectorized(unit_signal, n_fibers, delay_std_samp, rng, amp_mean=1.0, amp_std=0.1):
    L = len(unit_signal)
    if n_fibers <= 0:
        return np.zeros(L)
    delays = np.rint(rng.normal(0, delay_std_samp, n_fibers)).astype(int)
    amps = np.abs(rng.normal(amp_mean, amp_std, n_fibers))
    out = np.zeros(L)
    uniq, inv = np.unique(delays, return_inverse=True)
    sums = np.zeros_like(uniq, dtype=float)
    np.add.at(sums, inv, amps)
    for d, a in zip(uniq, sums):
        if d > 0:   out[d:] += a*unit_signal[:L-d]
        elif d < 0: out[:L+d] += a*unit_signal[-d:]
        else:       out += a*unit_signal
    return out

# --------------------- Minimal physiological simulator -------------------

def generate_modulated_EMG_physiological_upgraded(
    motorunits_max, n_fibers, FreqInputTotal, SNR, SNRE, SNRF,
    V_reset, V_e, dt, T_end, const_current, common_mod_subcortical,
    freq_left, freq_right, phase_left, phase_right, Intent_variability, window_size,
    cross_talk_R2L, cross_talk_L2R, common_mod_cortical,
    Vth_min, Vth_max, I_scale, Rm, tau_m, seed=1234, cortical_high_hz=30,
    # MUAP
    use_dog_muaps=True, muap_dur_s=0.015, muap_jitter_s=0.004,
    mu_gain_low=0.6, mu_gain_high=1.4,
    # LIF adaptation
    use_lif_adaptation=True, min_isi_ms=18.0, dVth=0.003, tau_adapt_ms=250.0,
    # Noise
    noise_type='pink', shaper_taps_mu=None, shaper_taps_emg=None,
    # v2 options
    intent_drift='trial', return_spikes=False, progress=False
):
    """
    Simulate a bilateral EMG pair with cortical/subcortical common drive, optional
    cross-limb mixing, LIF motor units, MUAP summation, and realistic noise.

    Returns
    -------
    EMG_L, EMG_R, fs : np.ndarray, np.ndarray, float
        Left and right **monopolar** EMG time series (after trimming the first
        min(1 s, 10 %) & adding EMG-level noise), and the source sampling rate
        fs (= 1/dt). Downstream examples use a fractional-delay bipolar
        projection for realism.
    info : dict   (only if return_spikes=True, as a 4th output)
        'spikes_L', 'spikes_R' (bool, n_MU x n_samples, trimmed like the EMG),
        'muap_dur_s' (per MU), 'gains' (per MU), 'trim' (samples removed).

    v2 changes (see CHANGELOG.md)
    ----------------------------
    The whole trial is integrated continuously: motor-neuron state is never
    reset, inputs/noise are generated for the full trial, MUAPs are convolved
    over the full spike train, each MU has one seeded MUAP duration, and every
    random draw is derived from `seed`. For the same seed the shared cortical,
    subcortical and envelope inputs are identical to v1.

    Core timing
    -----------
    dt : float
        Simulation time step (s). Use 0.0002 for 5 kHz internal rate.
    T_end : float
        Total simulated duration in seconds. (e.g., 18.0)
    window_size : float
        Kept for API compatibility. v1 integrated the LIF pool in windows of
        this length and (erroneously) reset the neurons each window; v2
        integrates continuously, so results do not depend on it.

    Drive & task envelope
    ---------------------
    const_current : float
        Amplitude of the unique (limb-specific) low-pass (FreqInputTotal) drive.
    common_mod_cortical : float
        Amplitude of 13–cortical_high_hz Hz common drive (β/γ-like).
    common_mod_subcortical : float
        Amplitude of 5–13 Hz common drive (α-like).
    cortical_high_hz : float
        Upper cutoff for the “cortical” band (e.g., 30 or 60).
    freq_left, freq_right : float
        Target “force” modulation frequencies (Hz), typically 0.5 vs 1.0.
        Use freq=0 with phase=np.pi/2 for a constant (isometric) envelope.
    phase_left, phase_right : float (rad)
        Phases to create 1:2 Lissajous (e.g., 0 and π/2).
    Intent_variability : float
        Slow drift of the drive envelope, in fraction per second.
    intent_drift : {'trial', 'none', 'window_legacy'}
        'trial' (default): continuous linear drift 1 + Intent_variability*t over
        the trial. 'none': no drift. 'window_legacy': v1 behaviour (ramp
        restarting every window = small sawtooth at 1/window_size Hz).

    Cross-limb mixing (“crosstalk” at the source level)
    ---------------------------------------------------
    cross_talk_R2L, cross_talk_L2R : float in [0,1]
        Proportion of the *other* side’s drive added into each limb (0.20 = 20%).

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
        Enables threshold after-spike adaptation and the refractory period.
    min_isi_ms : float
        Hard refractory (ms).
    dVth, tau_adapt_ms : float
        Adaptation step (V) and decay (ms).

    MUAP synthesis & noise
    ----------------------
    use_dog_muaps : bool
        If True, Difference-of-Gaussians MUAPs; if False, the original biphasic template.
    muap_dur_s, muap_jitter_s : float
        Mean MUAP duration (s) and per-MU uniform jitter (s), drawn once per MU.
    mu_gain_low, mu_gain_high : float
        Linear amplitude ramp across the MU pool (low→high threshold).
    noise_type : {'pink','shaped','white'}
        MU-level and EMG-level noise colouring. For 'shaped', provide FIR taps via
        shaper_taps_mu / shaper_taps_emg (see `colored_noise_from_shaper`).
    SNR, SNRF, SNRE : float
        SNRs for drive noise, MUAP-level noise, and EMG-level noise, respectively.

    Reproducibility
    ---------------
    seed : int
        Master seed; identical seeds give identical outputs.

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

    Notes
    -----
    Output is **monopolar** EMG. For sensors, use
    `simulate_bipolar_emg_spatial(x, fs, tau_ms=1.2, lp_hz=180)`.
    """
    T = np.arange(0, T_end, dt)
    fs = 1/dt
    N = len(T)
    nMU = int(motorunits_max)
    rng = np.random.default_rng(seed)

    def band_limited(length, low, high, amp):
        x = rng.standard_normal(length)
        x = butter_filter_low(x, high, fs); x = butter_filter_high(x, low, fs)
        return amp*zscore(x)

    # shared inputs (same draw order as v1 -> identical for the same seed)
    cort = band_limited(N, 13, cortical_high_hz, common_mod_cortical)
    sub = band_limited(N, 5, 13, common_mod_subcortical)
    envn = band_limited(N, 0.2, 2.0, 0.15)
    force_L = force_modulation_sine(T, freq_left, phase_left)
    force_R = force_modulation_sine(T, freq_right, phase_right)

    # independent, reproducible streams for everything else
    ss = np.random.SeedSequence(seed)
    ss_drive, ss_muap, ss_munoise, ss_fiber, ss_emgnoise = ss.spawn(5)
    rng_drive = np.random.default_rng(ss_drive)

    # slow drift of intent
    if intent_drift == 'trial':
        drift = Intent_variability*(T - T[0])
    elif intent_drift == 'window_legacy':
        wsamp = int(round(window_size/dt))
        drift = Intent_variability*(np.arange(N) % wsamp)*dt
    elif intent_drift == 'none':
        drift = np.zeros(N)
    else:
        raise ValueError("intent_drift must be 'trial', 'none' or 'window_legacy'")
    fL = force_L*(1.0 + drift + envn)
    fR = force_R*(1.0 + drift + envn)

    # unique drive + drive noise, whole trial
    uL = zscore(butter_filter_low(rng_drive.standard_normal(N), FreqInputTotal, fs))
    uR = zscore(butter_filter_low(rng_drive.standard_normal(N), FreqInputTotal, fs))
    sigL = const_current*uL + cort + sub
    sigR = const_current*uR + cort + sub
    nL = zscore(rng_drive.standard_normal(N))*(np.std(sigL)/np.sqrt(SNR))
    nR = zscore(rng_drive.standard_normal(N))*(np.std(sigR)/np.sqrt(SNR))
    ImL0 = I_scale*(sigL + nL)*fL
    ImR0 = I_scale*(sigR + nR)*fR
    # interhemispheric mixing
    ImL = (1 - cross_talk_R2L)*ImL0 + cross_talk_R2L*ImR0
    ImR = (1 - cross_talk_L2R)*ImR0 + cross_talk_L2R*ImL0
    del ImL0, ImR0, uL, uR, nL, nR, sigL, sigR

    # LIF pool, both limbs, continuous integration
    Th = np.linspace(Vth_min, Vth_max, nMU)
    spikes, _ = lif_population(np.vstack([ImL, ImR]), np.r_[np.zeros(nMU, int), np.ones(nMU, int)],
                               np.r_[Th, Th], V_reset, V_e, Rm, tau_m, dt,
                               adaptation=use_lif_adaptation, min_isi_ms=min_isi_ms,
                               dVth=dVth, tau_adapt_ms=tau_adapt_ms)
    spL, spR = spikes[:nMU], spikes[nMU:]

    # MUAPs: one seeded duration per MU; full-length convolution; per-MU noise and fibres
    rng_muap = np.random.default_rng(ss_muap)
    durs = muap_dur_s + rng_muap.uniform(-muap_jitter_s, muap_jitter_s, nMU)
    gains = np.linspace(mu_gain_low, mu_gain_high, nMU)
    rng_mun = np.random.default_rng(ss_munoise)
    fiber_ss = ss_fiber.spawn(nMU)
    EMG_L = np.zeros(N); EMG_R = np.zeros(N)
    it = tqdm(range(nMU), desc="MUAPs", leave=False) if progress else range(nMU)
    for i in it:
        mu_factor = i/(nMU-1 + 1e-12)
        shape = (generate_muap_shape_dog(durs[i], fs, mu_factor=mu_factor)
                 if use_dog_muaps else generate_muap_shape_original(durs[i], dt))
        rng_f = np.random.default_rng(fiber_ss[i])
        for sp, EMG in ((spL[i], EMG_L), (spR[i], EMG_R)):
            conv = np.convolve(sp.astype(float), shape, mode='same')*gains[i]
            sd = np.std(conv)
            if sd > 0:
                conv += _noise(N, fs, noise_type, rng_mun, shaper_taps_mu)*(sd/np.sqrt(SNRF))
            EMG += simulate_fiber_emg_vectorized(conv, n_fibers, delay_std_samp=1, rng=rng_f)

    # trim and add EMG-level noise
    trim = min(5000, N//10)
    EMG_L = EMG_L[trim:]; EMG_R = EMG_R[trim:]
    rng_e = np.random.default_rng(ss_emgnoise)

    def add_emg_noise(x):
        nz = _noise(x.size, fs, noise_type, rng_e, shaper_taps_emg)
        return x + nz*(np.std(x)/np.sqrt(SNRE)) if np.std(x) > 0 else x

    EMG_L = add_emg_noise(EMG_L); EMG_R = add_emg_noise(EMG_R)
    if return_spikes:
        info = dict(spikes_L=spL[:, trim:], spikes_R=spR[:, trim:], muap_dur_s=durs,
                    gains=gains, trim=trim)
        return EMG_L, EMG_R, fs, info
    return EMG_L, EMG_R, fs

# ------------------------- Wavelet coherence core ------------------------

def smoothwavelet(wave, dt, period, dj, scale):
    """Torrence & Webster / Grinsted smoothing used for wavelet coherence:
    Gaussian along TIME (width proportional to each scale), then a boxcar of
    0.6 octave along SCALE (Morlet).

    v1 bugs fixed here: (i) the boxcar was applied along time
    (kernel[np.newaxis, :]) instead of scale; (ii) np.real() discarded the
    imaginary part of complex inputs (the cross-spectrum).
    """
    wave = np.asarray(wave)
    if wave.ndim != 2:
        raise ValueError("wave must be a 2-D (scale, time) array")
    n = wave.shape[1]
    npad = 2 ** int(np.ceil(np.log2(n)))
    k = np.arange(1, npad//2 + 1) * (2*np.pi/npad)
    k = np.concatenate(([0.], k, -k[-2::-1]))
    k2 = k**2
    snorm = np.asarray(scale, float)/float(dt)
    F = np.exp(-0.5*(snorm[:, None]**2)*k2[None, :])
    twave = np.fft.ifft(F*np.fft.fft(wave, npad, axis=1), axis=1)[:, :n]
    if np.isrealobj(wave):
        twave = twave.real
    dj0 = 0.6
    dj0steps = dj0/(dj*2)
    kernel = np.concatenate(([dj0steps % 1], np.ones(int(2*round(dj0steps)-1)), [dj0steps % 1]))
    kernel /= kernel.sum()
    return convolve2d(twave, kernel[:, np.newaxis], mode="same")


def _wct_from_cwt(X, Y, scales, dt, dj):
    """Squared wavelet coherence from two CWTs on the same grid."""
    sinv = 1.0/scales
    period = None
    sX = smoothwavelet(sinv[:, None]*(np.abs(X)**2), dt, period, dj, scales)
    sY = smoothwavelet(sinv[:, None]*(np.abs(Y)**2), dt, period, dj, scales)
    sWxy = smoothwavelet(sinv[:, None]*(X*np.conj(Y)), dt, period, dj, scales)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rsq = np.abs(sWxy)**2/(sX*sY)
    Rsq = np.clip(np.real(Rsq), 0.0, 1.0)
    Rsq[~np.isfinite(Rsq)] = np.nan
    return Rsq


def compute_wavelet_coherence(sig1, sig2, fs, fmax=128.0, dj=1/8, s0=None, J=None, w0=6):
    """Squared Morlet wavelet coherence (Grinsted et al. 2004) of two signals.

    Returns freqs, period, coi, Rsq (n_freqs x n_times, masked to <= fmax),
    dt, dj, s0, J, mother. Defaults (s0 = dt, J = 8/dj) match v1, so the
    frequency grid is unchanged. Use band_medians(Rsq, freqs, period=period,
    coi=coi) to exclude the cone of influence.
    """
    sig1 = np.asarray(sig1, float); sig2 = np.asarray(sig2, float)
    if sig1.ndim != 1 or sig1.shape != sig2.shape:
        raise ValueError("sig1 and sig2 must be equal-length 1-D arrays")
    dt = 1.0/fs
    if s0 is None: s0 = 1*dt
    J = int(round(8/dj)) if J is None else int(J)
    mother = wavelet.Morlet(w0)

    X, scales, freqs, coix, *_ = wavelet.cwt(sig1, dt, dj, s0, J, mother)
    Y, scales_y, freqs_y, coiy, *_ = wavelet.cwt(sig2, dt, dj, s0, J, mother)
    if not np.allclose(scales, scales_y):
        raise RuntimeError("CWT grids differ between the two signals")
    period = 1/freqs
    Rsq = _wct_from_cwt(X, Y, scales, dt, dj)
    coi = np.minimum(coix, coiy)
    m = freqs <= fmax
    return freqs[m], period[m], coi, Rsq[m, :], dt, dj, s0, J, mother

# ------------------------ Monte-Carlo null (with PB) ---------------------

def mc_null_threshold(sig_len, dt, dj, s0, J, mother,
                      alpha=0.05, num_simulations=300,
                      bp_low=5.0, bp_high=499.0,
                      fmax=128.0, rng=None, coi_mask=False):
    """
    Generate independent band-passed Gaussian processes of length `sig_len`
    and compute the (1-alpha) percentile of the time-median wavelet coherence
    at each frequency. Returns (freqs_masked, upper_masked).

    coi_mask=True takes the time-median only inside the cone of influence
    (match this to how the data are summarised).
    """
    rng = np.random.default_rng() if rng is None else rng
    b, a = butter(4, [bp_low/(0.5/dt), bp_high/(0.5/dt)], btype="band")
    tmp = rng.standard_normal(sig_len)
    _, scales, freqs_full, coi, *_ = wavelet.cwt(tmp, dt, dj, s0, J, mother)
    F = freqs_full.size
    fmask = freqs_full <= fmax
    inside = (1/freqs_full)[:, None] <= coi[None, :] if coi_mask else None
    median_coh = np.empty((num_simulations, F), dtype=float)
    for i in tqdm(range(num_simulations), desc="MC null", leave=False):
        n1 = zscore(filtfilt(b, a, rng.standard_normal(sig_len)))
        n2 = zscore(filtfilt(b, a, rng.standard_normal(sig_len)))
        X, *_ = wavelet.cwt(n1, dt, dj, s0, J, mother)
        Y, *_ = wavelet.cwt(n2, dt, dj, s0, J, mother)
        Rsq = _wct_from_cwt(X, Y, scales, dt, dj)
        if inside is not None:
            Rsq = np.where(inside, Rsq, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_coh[i, :] = np.nanmedian(Rsq, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        upper_full = np.nanpercentile(median_coh, 100*(1-alpha), axis=0)
    return freqs_full[fmask], upper_full[fmask]
