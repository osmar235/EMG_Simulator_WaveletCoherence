# functions.py
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from typing import Tuple
from tqdm.auto import tqdm

from scipy.signal import (
    butter, filtfilt, resample_poly, welch, firls, convolve as sp_convolve
)
from scipy.signal import convolve2d
import pycwt as wavelet

# ------------------------------ Basics ------------------------------

def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)

def band_medians(Rsq, freqs, bands=((5,13),(13,30),(30,60),(60,100))):
    out = {}
    for (f0,f1) in bands:
        idx = np.where((freqs>=f0) & (freqs<f1))[0]
        out[f"{f0}-{f1}"] = float(np.median(Rsq[idx,:])) if len(idx)>0 else np.nan
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
    xpad = np.pad(x, (pad, pad), mode=pad_mode) if pad>0 else x
    ypad = np.convolve(xpad, h, mode="same")
    return ypad[pad:-pad] if pad>0 else ypad

def simulate_bipolar_emg_spatial(x, fs, tau_ms=1.2, lp_hz=180, wA=1.0, wB=1.0, fir_len=21):
    D = tau_ms*1e-3*fs
    chA = x
    chB = _frac_delay(x, D, N=fir_len, pad_mode="reflect")
    y = wA*chA - wB*chB
    if lp_hz is not None:
        b, a = butter(4, lp_hz/(fs/2), btype="low")
        y = filtfilt(b, a, y, padtype="odd", padlen=3*(max(len(a),len(b))-1))
    return y

# ----------------------------- Noise helpers -----------------------------

def pinkish_noise_fft(N, fs, alpha=1.0, f_lo=10, f_hi=400, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    w = rng.standard_normal(N)
    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    shape = np.ones_like(freqs)
    mask = (freqs>=f_lo) & (freqs<=f_hi)
    shape[~mask] = 0.0
    shape[mask] = np.maximum(freqs[mask], 1e-3)**(-alpha/2.0)
    y = np.fft.irfft(W*shape, n=N)
    return zscore(y)

# --------------------- Minimal physiological simulator -------------------

def _generate_muap_shape_dog(dur_s, fs, width_scale=1.0, tri_ratio=0.35, mu_factor=0.0):
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

def force_modulation_sine(t, f, phase=0):
    return (np.sin(2*np.pi*f*t + phase) + 1.0)/2.0

def lif_window_with_adaptation(Im, Vth_base, V_reset, V_e, Rm, tau_m, dt, v_mem_start,
                               min_isi_ms=18.0, dVth=0.003, tau_adapt_ms=250.0):
    Lw = Im.size
    spikes = np.zeros(Lw, dtype=float)
    V = np.empty(Lw); V[0] = v_mem_start
    Vth_dyn = Vth_base
    last_spike_t = -1e9
    min_isi = int(round(min_isi_ms/1000.0/dt))
    tau_adapt = max(tau_adapt_ms/1000.0, 1e-6)
    for t in range(Lw-1):
        Vth_dyn = Vth_base + (Vth_dyn - Vth_base)*np.exp(-dt/tau_adapt)
        if (t - last_spike_t) < min_isi:
            V[t+1] = V_reset
        elif V[t] >= Vth_dyn:
            spikes[t] = 1.0
            last_spike_t = t
            V[t+1] = V_reset
            Vth_dyn += dVth
        else:
            V[t+1] = V[t] + dt*((-(V[t]-V_e) + Im[t]*Rm)/tau_m)
    return spikes, V[-1]

def colored_noise_from_shaper(N, shaper_taps, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    w = rng.standard_normal(N + len(shaper_taps) - 1)
    y = np.convolve(w, shaper_taps, mode='valid')[:N]
    return zscore(y)

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

def generate_muap_shape_original(d, dt, tau=0.18):
    n = max(2, int(round((d/2)/dt)))
    t = np.linspace(0, d/2, n)
    shape1 = 5*np.sin(np.pi*t/(d/2)) * np.exp((1/tau)*((t/(d/2))-1))
    shape2 = -np.flip(shape1)[1:]
    return np.concatenate([shape1, shape2])

def simulate_fiber_emg_vectorized(unit_signal, n_fibers, delay_std_samp, rng, amp_mean=1.0, amp_std=0.1):
    L = len(unit_signal)
    if n_fibers <= 0: return np.zeros(L)
    delays = np.rint(rng.normal(0, delay_std_samp, n_fibers)).astype(int)
    amps   = np.abs(rng.normal(amp_mean, amp_std, n_fibers))
    out = np.zeros(L)
    uniq, inv = np.unique(delays, return_inverse=True)
    sums = np.zeros_like(uniq, dtype=float)
    np.add.at(sums, inv, amps)
    for d, a in zip(uniq, sums):
        if d > 0:   out[d:]     += a*unit_signal[:L-d]
        elif d < 0: out[:L+d]   += a*unit_signal[-d:]
        else:       out         += a*unit_signal
    return out

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
    noise_type='pink', shaper_taps_mu=None, shaper_taps_emg=None
):
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
    T = np.arange(0, T_end, dt)
    fs=1/dt
    rng = np.random.default_rng(seed)
    nMU = motorunits_max
    N = len(T)
    wsamp = int(round(window_size/dt))
    nW = int(np.ceil(N/wsamp))

    Th = np.linspace(Vth_min, Vth_max, nMU)
    mu = np.zeros((nMU, N))
    Vm_mem = np.full(nMU, V_reset)

    def band_limited(length, low, high, amp):
        x = rng.standard_normal(length)
        x = butter_filter_low(x, high, fs); x = butter_filter_high(x, low, fs)
        return amp*zscore(x)

    # shared inputs
    cort = band_limited(N, 13, cortical_high_hz, common_mod_cortical)
    sub  = band_limited(N,  5, 13, common_mod_subcortical)
    envn = band_limited(N, 0.2, 2.0, 0.15)
    force_L = force_modulation_sine(T, freq_left,  phase_left)
    force_R = force_modulation_sine(T, freq_right, phase_right)

    # integrate windowed
    EMG_L = np.zeros(N); EMG_R = np.zeros(N)
    for w in range(nW):
        s = w*wsamp; e = min((w+1)*wsamp, N)
        Lw = e - s; Tw = T[s:e]
        fL = force_L[s:e]*(1.0 + Intent_variability*(Tw-Tw[0]) + envn[s:e])
        fR = force_R[s:e]*(1.0 + Intent_variability*(Tw-Tw[0]) + envn[s:e])

        uL = zscore(butter_filter_low(rng.standard_normal(Lw), FreqInputTotal, fs))
        uR = zscore(butter_filter_low(rng.standard_normal(Lw), FreqInputTotal, fs))
        sigL = const_current*uL + cort[s:e] + sub[s:e]
        sigR = const_current*uR + cort[s:e] + sub[s:e]
        nL = zscore(rng.standard_normal(Lw)) * (np.std(sigL)/np.sqrt(SNR))
        nR = zscore(rng.standard_normal(Lw)) * (np.std(sigR)/np.sqrt(SNR))
        ImL = I_scale*(sigL + nL)*fL
        ImR = I_scale*(sigR + nR)*fR
        # interhemispheric mixing
        L0, R0 = ImL.copy(), ImR.copy()
        ImL = (1 - cross_talk_R2L)*L0 + cross_talk_R2L*R0
        ImR = (1 - cross_talk_L2R)*R0 + cross_talk_L2R*L0

        swL = np.zeros((nMU, Lw)); swR = np.zeros((nMU, Lw))
        for i in range(nMU):
            if use_lif_adaptation:
                sp, Vm_mem[i] = lif_window_with_adaptation(ImL, Th[i], V_reset, V_e, Rm, tau_m, dt, Vm_mem[i],
                                                           min_isi_ms, dVth, tau_adapt_ms)
            else:
                V = np.empty(Lw); V[0] = Vm_mem[i]
                for t in range(Lw-1):
                    if V[t] >= Th[i]: V[t+1] = V_reset
                    else:             V[t+1] = V[t] + dt*((-(V[t]-V_e) + ImL[t]*Rm)/tau_m)
                sp = (V>=Th[i]).astype(float); Vm_mem[i] = V[-1]
            swL[i] = sp

        Vm_mem_R = np.full(nMU, V_reset)
        for i in range(nMU):
            if use_lif_adaptation:
                sp, Vm_mem_R[i] = lif_window_with_adaptation(ImR, Th[i], V_reset, V_e, Rm, tau_m, dt, V_reset,
                                                             min_isi_ms, dVth, tau_adapt_ms)
            else:
                V = np.empty(Lw); V[0] = V_reset
                for t in range(Lw-1):
                    if V[t] >= Th[i]: V[t+1] = V_reset
                    else:             V[t+1] = V[t] + dt*((-(V[t]-V_e) + ImR[t]*Rm)/tau_m)
                sp = (V>=Th[i]).astype(float)
            swR[i] = sp

        # accumulate spike trains
        mu[:, s:e] = swL  # store left for MDR if desired (right not stored to save mem)

        # convolve & sum (full-window, then overlap-safe place)
        gains = np.linspace(0.6, 1.4, nMU)
        MU_L = np.zeros(Lw); MU_R = np.zeros(Lw)
        for i in range(nMU):
            d = muap_dur_s + np.random.uniform(-muap_jitter_s, muap_jitter_s)
            mu_factor = i/(nMU-1 + 1e-12)
            shape = (generate_muap_shape_dog(d, 1/dt, mu_factor=mu_factor)
                     if use_dog_muaps else
                     generate_muap_shape_original(d, dt))
            convL = np.convolve(swL[i], shape, mode='same') * gains[i]
            convR = np.convolve(swR[i], shape, mode='same') * gains[i]
            # MU-level noise
            if noise_type == 'pink':
                nzL = pinkish_noise_fft(Lw, 1/dt, alpha=1.2, f_lo=10, f_hi=450)
                nzR = pinkish_noise_fft(Lw, 1/dt, alpha=1.2, f_lo=10, f_hi=450)
            elif noise_type == 'shaped' and (shaper_taps_mu is not None):
                nzL = colored_noise_from_shaper(Lw, shaper_taps_mu)
                nzR = colored_noise_from_shaper(Lw, shaper_taps_mu)
            else:
                nzL = zscore(np.random.standard_normal(Lw))
                nzR = zscore(np.random.standard_normal(Lw))
            if np.std(convL)>0: convL += nzL*(np.std(convL)/np.sqrt(SNRF))
            if np.std(convR)>0: convR += nzR*(np.std(convR)/np.sqrt(SNRF))
            # fiber dispersion (fast, vectorized)
            rng = np.random.default_rng(seed=1234+i+w*17)
            MU_L += simulate_fiber_emg_vectorized(convL, n_fibers, delay_std_samp=1, rng=rng)
            MU_R += simulate_fiber_emg_vectorized(convR, n_fibers, delay_std_samp=1, rng=rng)
        EMG_L[s:e] += MU_L; EMG_R[s:e] += MU_R

    # trim and add EMG-level noise
    trim = min(5000, N//10)
    EMG_L = EMG_L[trim:]; EMG_R = EMG_R[trim:]
    def add_emg_noise(x):
        if noise_type == 'pink':
            nz = pinkish_noise_fft(x.size, 1/dt, alpha=1.2, f_lo=10, f_hi=450)
        elif noise_type == 'shaped' and (shaper_taps_emg is not None):
            nz = colored_noise_from_shaper(x.size, shaper_taps_emg)
        else:
            nz = zscore(np.random.standard_normal(x.size))
        return x + nz*(np.std(x)/np.sqrt(SNRE)) if np.std(x)>0 else x

    EMG_L = add_emg_noise(EMG_L); EMG_R = add_emg_noise(EMG_R)
    return EMG_L, EMG_R, 1/dt

# ------------------------- Wavelet coherence core ------------------------

def smoothwavelet(wave, dt, period, dj, scale):
    """Gaussian along frequency, boxcar along time (pycwt-style)."""
    n = wave.shape[1]
    twave = np.zeros_like(wave)
    npad = 2 ** int(np.ceil(np.log2(n)))
    k = np.arange(1, npad//2 + 1) * (2*np.pi/npad)
    k = np.concatenate(([0.], k, -k[-2::-1]))
    k2 = k**2
    snorm = scale/dt
    for ii in range(wave.shape[0]):
        F = np.exp(-0.5*(snorm[ii]**2)*k2)
        smooth = np.fft.ifft(F*np.fft.fft(wave[ii,:], npad))
        twave[ii,:] = np.real(smooth[:n])
    dj0 = 0.6
    dj0steps = dj0/(dj*2)
    kernel = np.concatenate(([dj0steps%1], np.ones(int(2*round(dj0steps)-1)), [dj0steps%1]))
    kernel /= (2*round(dj0steps)-1 + 2*(dj0steps%1))
    return convolve2d(twave, kernel[np.newaxis,:], mode="same")

def compute_wavelet_coherence(sig1, sig2, fs, fmax=128.0, dj=1/8, s0=None, J=None, w0=6):
    dt = 1.0/fs
    if s0 is None: s0 = 1*dt
    if J  is None: J  = 8/dj
    mother = wavelet.Morlet(w0)

    X, scales, freqs, coix, *_ = wavelet.cwt(sig1, dt, dj, s0, J, mother)
    Y, scales, freqs, coiy, *_ = wavelet.cwt(sig2, dt, dj, s0, J, mother)
    period = 1/freqs
    sinv   = 1.0/scales

    sX   = smoothwavelet(sinv[:,None]*(np.abs(X)**2), dt, 1/freqs, dj, scales)
    sY   = smoothwavelet(sinv[:,None]*(np.abs(Y)**2), dt, 1/freqs, dj, scales)
    sWxy = smoothwavelet(sinv[:,None]*(X*np.conj(Y)), dt, period, dj, scales)
    Rsq  = np.abs(sWxy)**2/(sX*sY)  # (F, T)

    # COI (same length as time dimension)
    coi = np.minimum(coix, coiy)

    # mask to fmax for output
    m = freqs <= fmax
    return freqs[m], period[m], coi, Rsq[m,:], dt, dj, s0, J, mother

# ------------------------ Monte-Carlo null (with PB) ---------------------

def mc_null_threshold(sig_len, dt, dj, s0, J, mother,
                      alpha=0.05, num_simulations=300,
                      bp_low=5.0, bp_high=499.0,
                      fmax=128.0, rng=None):
    """
    Generate independent band-passed Gaussian processes of length `sig_len`
    and compute the (1-alpha) percentile of time-median wavelet coherence.
    Returns (freqs_masked, upper_masked).
    """
    rng = np.random.default_rng() if rng is None else rng

    # frequency grid (fixed across sims)
    tmp = rng.standard_normal(sig_len)
    X0, scales, freqs_full, *_ = wavelet.cwt(tmp, dt, dj, s0, J, mother)
    F = freqs_full.size
    fmask = (freqs_full <= fmax)

    # preallocate
    median_coh = np.empty((num_simulations, F), dtype=float)

    # band-pass for null signals
    b, a = butter(4, [bp_low/(0.5/dt), bp_high/(0.5/dt)], btype="band")

    for i in tqdm(range(num_simulations), desc="MC null", leave=False):
        n1 = filtfilt(b, a, rng.standard_normal(sig_len))
        n2 = filtfilt(b, a, rng.standard_normal(sig_len))
        n1 = (n1 - n1.mean())/(n1.std()+1e-12)
        n2 = (n2 - n2.mean())/(n2.std()+1e-12)

        X, scales, freqs, *_ = wavelet.cwt(n1, dt, dj, s0, J, mother)
        Y, scales, freqs, *_ = wavelet.cwt(n2, dt, dj, s0, J, mother)
        # shape checks (must match the fixed grid)
        if freqs.size != F:
            # extremely unlikely with fixed dj/s0/J; guard anyway
            # pad/truncate to match F
            take = min(F, freqs.size)
            if take < F:
                pad = np.full(F-take, np.nan)
            # compute Rsq on available part, then pad
        period = 1.0/freqs
        sinv   = 1.0/scales

        sX   = smoothwavelet(sinv[:,None]*(np.abs(X)**2), dt, 1/freqs, dj, scales)
        sY   = smoothwavelet(sinv[:,None]*(np.abs(Y)**2), dt, 1/freqs, dj, scales)
        sWxy = smoothwavelet(sinv[:,None]*(X*np.conj(Y)), dt, period, dj, scales)
        Rsq  = np.abs(sWxy)**2/(sX*sY)  # (F, T)

        v = np.median(Rsq, axis=1)      # (F,)
        v = np.asarray(v, float)        # coerce dtype
        # replace any NaNs/Infs before write
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        # final shape assert
        if v.shape[0] != F:
            # safe reconcile (pad or crop)
            if v.shape[0] < F:
                v = np.pad(v, (0, F - v.shape[0]), mode="edge")
            else:
                v = v[:F]

        median_coh[i, :] = v

    upper_full = np.percentile(median_coh, 100*(1-alpha), axis=0)
    return freqs_full[fmask], upper_full[fmask]
