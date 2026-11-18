# functions.py - EMG Simulator - Wavelet Coherence
#
# Version 3.3.0. Scientific/reproducibility revision built from v2.0.0.
# v1 and v2 are preserved under legacy/ for exact comparison.
#
# Main v3 changes:
#   1. Common cortical/subcortical drive can be RMS-normalized so source
#      composition changes without changing total common-drive magnitude.
#   2. SNR values are interpreted explicitly in dB by default.
#   3. T_end is now the returned usable duration; burn-in is simulated separately.
#   4. Intent variability defaults to bounded low-frequency stochastic variation
#      rather than an unbounded percent-per-second ramp.
#   5. The exact Matviyenko MATLAB window is retained for sensitivity analyses,
#      but V3.3 primary coherence analysis uses no full-record taper and excludes
#      the cone of influence (COI).
#   6. Phase randomization is retained only as a legacy diagnostic. Primary null
#      inference uses same-condition cross-trial pairing/derangements so the
#      burst/task structure of the simulated EMG is preserved.
#   7. Any analysis taper is applied only to a copy immediately before CWT and
#      never alters exported/plotted time-domain bipolar EMG.
#   8. pycwt is optional at import time; simulation-only workflows still run.

__version__ = "3.3.0"

import warnings
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401  (kept for backward compatibility)
from fractions import Fraction
from typing import Tuple  # noqa: F401
from tqdm.auto import tqdm

from scipy.signal import (  # noqa: F401
    butter, filtfilt, resample_poly, welch, firls, convolve as sp_convolve, detrend as scipy_detrend
)
from scipy.signal import convolve2d
from scipy.signal import windows as signal_windows

try:
    import pycwt as wavelet
except ImportError:  # simulation-only use must not fail when pycwt is absent
    wavelet = None

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



# ------------------------ Matviyenko spectral taper ------------------------
# Exact coefficient port of the user-supplied MATLAB function
#     b = matviyenko(K, n)
# by Francois Meyer (2003), based on Gregory Matviyenko (1996).
# The MATLAB routine returns a symmetric window of length 2*n.  K controls
# steepness / sidelobe suppression and is defined for integer K=2,...,13.
_MATVIYENKO_G = {
    2: [1.1002143947640],
    3: [1.1723768006269012949, 0.1855148479250006034],
    4: [1.2031447668472587192, 0.2487749850071917170, 0.0475141123801596348],
    5: [1.2196727213232474166, 0.2853161868567129887, 0.0789155484618257136, 0.0135550276613148530],
    6: [1.2299263341780548351, 0.3091936162560012950, 0.1022547499107077172, 0.0270080848154667718, 0.0040643028367457815],
    7: [1.2368960390161255733, 0.3260454684070418003, 0.1202144284772571658, 0.0392824672307590613, 0.0094646390135188831, 0.0012540398189263445],
    8: [1.2419380540495429195, 0.3385842628287665721, 0.1344330305316184567, 0.0501467090467806012, 0.0153116614831942916, 0.0033448841387432324, 0.0003942039965801714],
    9: [1.2457536263723204704, 0.3482817014907560773, 0.1459560151659402844, 0.0596831230559218041, 0.0211609941444967639, 0.0059658781972807312, 0.0011854380315612793, 0.0001255469017738763],
    10: [1.2487411472878298253, 0.3560069611425606312, 0.1554771240803420597, 0.0680536775001451859, 0.0267991398186155608, 0.0088894805175620360, 0.0023126782210685839, 0.0004203150366958263, 0.0000403732905776057],
    11: [1.2511435101979626018, 0.3623069592485838316, 0.1634729471600794903, 0.0754255125823668245, 0.0321305559573675648, 0.0119592651970663211, 0.0036995960643158676, 0.0008907386396059369, 0.0001489420206098608, 0.0000130803773243655],
    12: [1.2531171572760240244, 0.3675432476836647534, 0.1702809341651916143, 0.0819484046917260704, 0.0371207535134114532, 0.0150712511279723392, 0.0052752483692935777, 0.0015235383929696997, 0.0003408100998412218, 0.0000527237866578197, 0.0000042630197561061],
    13: [1.2547673684254528441, 0.3719644997534442686, 0.1761463604899294465, 0.0877499693755897336, 0.0417667654449872536, 0.0181579310814718085, 0.0069802273692443469, 0.0022969056584170223, 0.0006209177060710873, 0.0001295783756168840, 0.0000186407482263636, 0.0000013960635001692],
}


def matviyenko(K, n):
    """Exact NumPy port of the supplied MATLAB ``matviyenko(K,n)``.

    Parameters
    ----------
    K : int
        Window steepness/order, 2 <= K <= 13.
    n : int
        Half-window length.  The returned window contains exactly ``2*n``
        samples, matching the MATLAB routine.
    """
    if isinstance(K, bool) or int(K) != K or not (2 <= int(K) <= 13):
        raise ValueError('K must be an integer from 2 through 13')
    if isinstance(n, bool) or int(n) != n or int(n) < 1:
        raise ValueError('n must be a positive integer half-window length')
    K, n = int(K), int(n)
    coeff = np.asarray(_MATVIYENKO_G[K], dtype=float)
    x = (np.arange(1, n + 1, dtype=float) / n) - 0.5
    harmonic = np.arange(K - 1, dtype=float) + 0.5
    S = np.sin(np.pi * x[:, None] * harmonic[None, :])
    r = 0.5 * (1.0 + S @ coeff)
    b = np.empty(2 * n, dtype=float)
    b[:n] = r
    b[n:] = r[::-1]
    return b


def matviyenko_window(length, K=8):
    """Return a Matviyenko window with exactly ``length`` samples.

    The original MATLAB definition is intrinsically even-length (2*n), so this
    wrapper intentionally rejects odd lengths instead of silently altering the
    historical formula.  Paper simulations at 18 s x 1000 Hz contain 18,000
    samples and therefore satisfy this requirement exactly.

    ``K=8`` is a Version-3 reproducible preset, *not* a claim that K=8 was the
    historical value used in the published analysis; the supplied MATLAB file
    defines K as a free parameter and does not record the old call site.
    """
    if isinstance(length, bool) or int(length) != length or int(length) < 2:
        raise ValueError('length must be an integer >= 2')
    length = int(length)
    if length % 2:
        raise ValueError('Exact Matviyenko port requires an even signal length (2*n)')
    return matviyenko(K, length // 2)


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

# ---------------------------- v3 helpers ---------------------------------

def _require_pycwt():
    if wavelet is None:
        raise ImportError(
            "Wavelet analysis requires the optional dependency 'pycwt'. "
            "Install dependencies with: pip install -r requirements.txt"
        )


def _snr_noise_sd(signal_sd, snr, units='db'):
    """Return noise SD required for the requested SNR.

    For dB, SNR_dB = 20*log10(signal_sd/noise_sd), equivalent to the
    conventional 10*log10 power ratio.  'linear_power' reproduces v1/v2.
    """
    signal_sd = float(signal_sd)
    snr = float(snr)
    if signal_sd <= 0:
        return 0.0
    if units == 'db':
        return signal_sd / (10.0 ** (snr / 20.0))
    if units == 'linear_power':
        if snr <= 0:
            raise ValueError('linear-power SNR must be > 0')
        return signal_sd / np.sqrt(snr)
    raise ValueError("snr_units must be 'db' or 'linear_power'")


def measured_snr_db(signal, noise):
    """Measured RMS/power SNR in dB (utility for validation)."""
    s = np.std(np.asarray(signal, float))
    n = np.std(np.asarray(noise, float))
    return np.inf if n == 0 else 20.0*np.log10(s/n)


def compose_common_drive(cortical, subcortical, cortical_weight, subcortical_weight,
                         target_rms=6.0, normalization='rms',
                         fraction_interpretation='amplitude'):
    """Compose cortical/subcortical common drive without a strength confound.

    Parameters
    ----------
    cortical, subcortical : arrays
        Preferably zero-mean/unit-SD band-limited processes.
    cortical_weight, subcortical_weight : float
        Used to infer the mixture fraction.  E.g. 6/0 -> 100/0,
        4.875/1.625 -> 75/25, 3.25/3.25 -> 50/50.
    target_rms : float
        SD/RMS of the *combined* common drive after mixing when
        normalization='rms'.  6.0 corresponds to 37.5% of an amplitude sum
        with unique-drive amplitude 10: 6/(10+6)=0.375.
    normalization : {'rms','legacy_coefficients'}
        'rms' is the v3 default.  'legacy_coefficients' reproduces v2's direct
        coefficient multiplication, where total common-drive RMS varies with
        the mixture.
    fraction_interpretation : {'amplitude','power'}
        With 'amplitude', p:(1-p) is the pre-normalization amplitude ratio.
        With 'power', sqrt(p):sqrt(1-p) is used so p is the intended power
        fraction before final RMS normalization.
    """
    c = zscore(cortical)
    s = zscore(subcortical)
    cw = abs(float(cortical_weight)); sw = abs(float(subcortical_weight))
    total = cw + sw
    if total <= 0:
        return np.zeros_like(c), dict(cortical_fraction=np.nan, rms=0.0,
                                      coefficients=(0.0, 0.0))
    p = cw / total
    if normalization == 'legacy_coefficients':
        out = float(cortical_weight)*c + float(subcortical_weight)*s
        return out, dict(cortical_fraction=p, rms=float(np.std(out)),
                         coefficients=(float(cortical_weight), float(subcortical_weight)))
    if normalization != 'rms':
        raise ValueError("common_drive_normalization must be 'rms' or 'legacy_coefficients'")
    if fraction_interpretation == 'amplitude':
        a, b = p, 1.0-p
    elif fraction_interpretation == 'power':
        a, b = np.sqrt(p), np.sqrt(1.0-p)
    else:
        raise ValueError("common_fraction_interpretation must be 'amplitude' or 'power'")
    raw = a*c + b*s
    raw_sd = np.std(raw)
    out = np.zeros_like(raw) if raw_sd <= 1e-15 else raw * (float(target_rms)/raw_sd)
    scale = 0.0 if raw_sd <= 1e-15 else float(target_rms)/raw_sd
    return out, dict(cortical_fraction=p, rms=float(np.std(out)),
                     coefficients=(a*scale, b*scale))


def _bounded_intent_variation(n, fs, rms_fraction, rng, cutoff_hz=0.25, clip_sd=3.0):
    """Smooth low-frequency intent variation with requested fractional RMS.

    A short random Fourier series is used instead of filtering near-DC white
    noise, avoiding numerical/edge instabilities for 5-kHz data.  Five
    components span roughly one cycle/trial to `cutoff_hz`; the result is
    linearly detrended, zero-centered, and RMS-normalized.  A final safety clip
    at `clip_sd` times the target RMS is applied only if needed.
    """
    n = int(n); rms_fraction = float(rms_fraction); fs = float(fs)
    if rms_fraction <= 0 or n <= 2:
        return np.zeros(n, float)
    duration = n/fs
    f_hi = max(float(cutoff_hz), 1.0/duration)
    f_lo = max(1.0/duration, min(0.05, f_hi/5.0))
    freqs = np.linspace(f_lo, f_hi, 5)
    amps = rng.standard_normal(freqs.size)
    phases = rng.uniform(0.0, 2*np.pi, freqs.size)
    t = np.arange(n)/fs
    y = np.zeros(n, float)
    for a0, f0, p0 in zip(amps, freqs, phases):
        y += a0*np.sin(2*np.pi*f0*t + p0)
    y = scipy_detrend(y, type='linear')
    y -= y.mean()
    sd = y.std()
    if sd <= 1e-15:
        return np.zeros(n, float)
    y *= rms_fraction/sd
    limit = abs(float(clip_sd))*rms_fraction
    if limit > 0 and np.max(np.abs(y)) > limit:
        y = np.clip(y, -limit, limit)
        y -= y.mean()
        # Recover as much target RMS as possible without violating the bound.
        sd = y.std()
        if sd > 0:
            scale = min(rms_fraction/sd, limit/(np.max(np.abs(y)) + 1e-15))
            y *= scale
    return y


def preprocess_emg_for_coherence(x, fs, low=5.0, high=499.0,
                                 detrend=True, taper=None, taper_kwargs=None,
                                 zscore_output=True):
    """Prepare non-rectified EMG for coherence analysis.

    Detrending is explicit and enabled by default. ``taper`` may be None,
    ``'matviyenko'``, ``'tukey'``, a callable returning a length-N window, or
    a length-N array.  For ``'matviyenko'``, pass ``taper_kwargs={'K': ...}``.
    The exact supplied MATLAB coefficients are used. In V3.3 the primary
    analysis uses no full-record taper; Matviyenko K is a sensitivity parameter
    because the historical call site specifying K was not supplied.
    """
    y = np.asarray(x, float)
    y = bandpass_emg(y, fs, low=low, high=min(high, fs/2.0-1e-6))
    if detrend:
        y = scipy_detrend(y, type='linear')
    if taper is not None:
        kw = {} if taper_kwargs is None else dict(taper_kwargs)
        if isinstance(taper, str):
            taper_name = taper.lower()
            if taper_name == 'matviyenko':
                K = int(kw.pop('K', 8))
                if kw:
                    raise TypeError(f"Unexpected Matviyenko taper kwargs: {sorted(kw)}")
                w = matviyenko_window(y.size, K=K)
            elif taper_name == 'tukey':
                w = signal_windows.tukey(y.size, alpha=float(kw.pop('alpha', 0.1)))
                if kw:
                    raise TypeError(f"Unexpected Tukey taper kwargs: {sorted(kw)}")
            else:
                raise ValueError("Unknown taper. Use None, 'matviyenko', 'tukey', callable, or an array.")
        elif callable(taper):
            w = np.asarray(taper(y.size, **kw), float)
        else:
            w = np.asarray(taper, float)
        if w.shape != y.shape:
            raise ValueError('taper must have the same length as the signal')
        y = y*w
    return zscore(y) if zscore_output else y


def apply_analysis_taper(x, taper=None, taper_kwargs=None, zscore_output=True):
    """Apply an optional analysis-only taper to an already preprocessed signal.

    ``x`` is expected to have already been filtered/detrended for coherence.
    V3.3 primary analysis passes ``taper=None``. Matviyenko/Tukey are retained
    for sensitivity analyses. This helper never mutates the input array.
    """
    y = np.asarray(x, float).copy()
    if taper is not None:
        kw = {} if taper_kwargs is None else dict(taper_kwargs)
        if isinstance(taper, str):
            taper_name = taper.lower()
            if taper_name == 'matviyenko':
                K = int(kw.pop('K', 8))
                if kw:
                    raise TypeError(f"Unexpected Matviyenko taper kwargs: {sorted(kw)}")
                w = matviyenko_window(y.size, K=K)
            elif taper_name == 'tukey':
                w = signal_windows.tukey(y.size, alpha=float(kw.pop('alpha', 0.1)))
                if kw:
                    raise TypeError(f"Unexpected Tukey taper kwargs: {sorted(kw)}")
            else:
                raise ValueError("Unknown taper. Use None, 'matviyenko', 'tukey', callable, or an array.")
        elif callable(taper):
            w = np.asarray(taper(y.size, **kw), float)
        else:
            w = np.asarray(taper, float)
        if w.shape != y.shape:
            raise ValueError('taper must have the same length as the signal')
        y *= w
    return zscore(y) if zscore_output else y


def random_derangement(n, rng=None):
    """Return a random permutation of ``range(n)`` with no fixed points.

    Used by V3.3 same-condition cross-trial null inference. For n < 2 no
    derangement exists and a ValueError is raised.
    """
    n = int(n)
    if n < 2:
        raise ValueError('a derangement requires at least two trials')
    rng = np.random.default_rng() if rng is None else rng
    base = np.arange(n)
    # Rejection is efficient for the small trial counts used here (n=15).
    for _ in range(10000):
        p = rng.permutation(n)
        if np.all(p != base):
            return p
    # Guaranteed fallback: a non-zero cyclic shift is always a derangement.
    shift = int(rng.integers(1, n))
    return np.roll(base, shift)


def phase_randomize(x, rng=None):
    """Plain Fourier phase-randomized surrogate preserving the periodogram.

    This is *not* IAAFT. In V3.3 it is retained only for legacy/diagnostic
    comparisons; it is not the primary significance/null procedure because it
    destroys the task-locked burst/nonstationary structure of the EMG.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, float)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    phase = np.angle(X)
    if X.size > 2:
        hi = X.size-1 if x.size % 2 == 0 else X.size
        phase[1:hi] = rng.uniform(0.0, 2*np.pi, hi-1)
    Y = mag*np.exp(1j*phase)
    # preserve real DC and Nyquist bins
    Y[0] = X[0].real + 0j
    if x.size % 2 == 0 and Y.size > 1:
        Y[-1] = X[-1].real + 0j
    return np.fft.irfft(Y, n=x.size)

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
    # v3 options
    snr_units='db', burn_in_s=1.0,
    common_drive_normalization='rms', common_drive_target_rms=6.0,
    common_fraction_interpretation='amplitude',
    intent_drift='bounded', intent_cutoff_hz=0.25,
    return_spikes=False, progress=False
):
    """Simulate bilateral monopolar EMG with a continuously integrated LIF pool.

    Version 3 defaults intentionally differ from v2 in four scientific places:
    (1) the combined common drive is normalized to a fixed RMS across cortical /
    subcortical mixtures; (2) SNR values are dB; (3) T_end is the *returned*
    usable duration, with a separate burn-in; and (4) Intent_variability is a
    bounded low-frequency fractional RMS rather than an unbounded %/s ramp.

    To approximate v2 behavior use::

        snr_units='linear_power', burn_in_s=0.0,
        common_drive_normalization='legacy_coefficients',
        intent_drift='linear_per_second'

    The old v2 implementation itself is preserved in legacy/v2/functions_v2.py.

    Parameters retained from v2 keep their meaning unless overridden above.
    `common_mod_cortical` and `common_mod_subcortical` now primarily determine
    the mixture ratio when common_drive_normalization='rms'.  For example,
    6/0 -> 100/0, 4.875/1.625 -> 75/25, and 3.25/3.25 -> 50/50.  The final
    combined common-drive SD is `common_drive_target_rms` for every mixture.

    `fraction_interpretation='amplitude'` matches the manuscript's coefficient
    language.  Set it to 'power' if 75/25 should mean 75/25 variance contribution.

    Returns EMG_L, EMG_R, fs.  With return_spikes=True a fourth `info` dict is
    returned with spike trains, exact common-drive metadata, duration/burn-in,
    and the effective SNR interpretation.
    """
    if dt <= 0 or T_end <= 0:
        raise ValueError('dt and T_end must be positive')
    if not (0 <= cross_talk_R2L <= 1 and 0 <= cross_talk_L2R <= 1):
        raise ValueError('cross-talk coefficients must lie in [0,1]')
    if cortical_high_hz <= 13:
        raise ValueError('cortical_high_hz must be > 13 Hz')
    fs = 1.0/dt
    if cortical_high_hz >= fs/2:
        raise ValueError('cortical_high_hz must be below Nyquist')

    n_out = int(round(float(T_end)/dt))
    n_burn = int(round(max(float(burn_in_s), 0.0)/dt))
    N = n_burn + n_out
    # t=0 corresponds exactly to the first returned sample after burn-in.
    T = (np.arange(N) - n_burn)*dt
    nMU = int(motorunits_max)
    if nMU < 1:
        raise ValueError('motorunits_max must be >= 1')

    # Independent deterministic streams by simulation stage.
    ss = np.random.SeedSequence(seed)
    (ss_shared, ss_drive, ss_intent, ss_muap, ss_munoise,
     ss_fiber, ss_emgnoise) = ss.spawn(7)
    rng_shared = np.random.default_rng(ss_shared)
    rng_drive = np.random.default_rng(ss_drive)
    rng_intent = np.random.default_rng(ss_intent)

    def band_limited_unit(length, low, high, rng):
        x = rng.standard_normal(length)
        x = butter_filter_low(x, high, fs)
        x = butter_filter_high(x, low, fs)
        return zscore(x)

    cort_z = band_limited_unit(N, 13, cortical_high_hz, rng_shared)
    sub_z = band_limited_unit(N, 5, 13, rng_shared)
    common, common_meta = compose_common_drive(
        cort_z, sub_z, common_mod_cortical, common_mod_subcortical,
        target_rms=common_drive_target_rms,
        normalization=common_drive_normalization,
        fraction_interpretation=common_fraction_interpretation)

    # Keep the v2 shared 0.2-2 Hz envelope modulation as part of the published
    # architecture; v3 changes only the separate Intent_variability term.
    envn = 0.15*band_limited_unit(N, 0.2, 2.0, rng_shared)
    force_L = force_modulation_sine(T, freq_left, phase_left)
    force_R = force_modulation_sine(T, freq_right, phase_right)

    if intent_drift == 'bounded':
        drift_L = _bounded_intent_variation(N, fs, Intent_variability, rng_intent,
                                            cutoff_hz=intent_cutoff_hz)
        drift_R = _bounded_intent_variation(N, fs, Intent_variability, rng_intent,
                                            cutoff_hz=intent_cutoff_hz)
    elif intent_drift == 'linear_per_second':
        drift_L = drift_R = Intent_variability*T
    elif intent_drift == 'window_legacy':
        wsamp = max(1, int(round(window_size/dt)))
        d = Intent_variability*(np.arange(N) % wsamp)*dt
        drift_L = drift_R = d
    elif intent_drift == 'none':
        drift_L = drift_R = np.zeros(N)
    else:
        raise ValueError("intent_drift must be 'bounded', 'linear_per_second', "
                         "'window_legacy', or 'none'")

    fL = force_L*(1.0 + envn + drift_L)
    fR = force_R*(1.0 + envn + drift_R)

    # Unique limb-specific broadband drive + fixed-magnitude common drive.
    uL = zscore(butter_filter_low(rng_drive.standard_normal(N), FreqInputTotal, fs))
    uR = zscore(butter_filter_low(rng_drive.standard_normal(N), FreqInputTotal, fs))
    sigL = const_current*uL + common
    sigR = const_current*uR + common
    nL0 = zscore(rng_drive.standard_normal(N))
    nR0 = zscore(rng_drive.standard_normal(N))
    nL = nL0*_snr_noise_sd(np.std(sigL), SNR, snr_units)
    nR = nR0*_snr_noise_sd(np.std(sigR), SNR, snr_units)
    ImL0 = I_scale*(sigL + nL)*fL
    ImR0 = I_scale*(sigR + nR)*fR

    # Source-level cross-limb mixing. Convex mixing keeps overall scaling stable.
    ImL = (1.0-cross_talk_R2L)*ImL0 + cross_talk_R2L*ImR0
    ImR = (1.0-cross_talk_L2R)*ImR0 + cross_talk_L2R*ImL0

    # LIF pool, both limbs, one continuous integration over burn-in + data.
    Th = np.linspace(Vth_min, Vth_max, nMU)
    spikes, _ = lif_population(
        np.vstack([ImL, ImR]),
        np.r_[np.zeros(nMU, int), np.ones(nMU, int)],
        np.r_[Th, Th], V_reset, V_e, Rm, tau_m, dt,
        adaptation=use_lif_adaptation, min_isi_ms=min_isi_ms,
        dVth=dVth, tau_adapt_ms=tau_adapt_ms)
    spL, spR = spikes[:nMU], spikes[nMU:]

    # MUAP synthesis: one duration/MU; full-trial convolution; seeded noise/fibres.
    rng_muap = np.random.default_rng(ss_muap)
    durs = muap_dur_s + rng_muap.uniform(-muap_jitter_s, muap_jitter_s, nMU)
    gains = np.linspace(mu_gain_low, mu_gain_high, nMU)
    rng_mun = np.random.default_rng(ss_munoise)
    fiber_ss = ss_fiber.spawn(2*nMU)
    EMG_L = np.zeros(N); EMG_R = np.zeros(N)
    it = tqdm(range(nMU), desc='MUAPs', leave=False) if progress else range(nMU)
    for i in it:
        mu_factor = i/(nMU-1 + 1e-12)
        shape = (generate_muap_shape_dog(durs[i], fs, mu_factor=mu_factor)
                 if use_dog_muaps else generate_muap_shape_original(durs[i], dt))
        for side, sp, EMG in ((0, spL[i], EMG_L), (1, spR[i], EMG_R)):
            conv = np.convolve(sp.astype(float), shape, mode='same')*gains[i]
            sd = np.std(conv)
            if sd > 0:
                nz = _noise(N, fs, noise_type, rng_mun, shaper_taps_mu)
                conv += nz*_snr_noise_sd(sd, SNRF, snr_units)
            rng_f = np.random.default_rng(fiber_ss[2*i + side])
            EMG += simulate_fiber_emg_vectorized(
                conv, n_fibers, delay_std_samp=1, rng=rng_f)

    # Remove burn-in *after* all dynamical/EMG processing, then add output noise.
    EMG_L = EMG_L[n_burn:n_burn+n_out]
    EMG_R = EMG_R[n_burn:n_burn+n_out]
    rng_e = np.random.default_rng(ss_emgnoise)

    def add_emg_noise(x):
        sd = np.std(x)
        if sd <= 0:
            return x
        nz = _noise(x.size, fs, noise_type, rng_e, shaper_taps_emg)
        return x + nz*_snr_noise_sd(sd, SNRE, snr_units)

    EMG_L = add_emg_noise(EMG_L)
    EMG_R = add_emg_noise(EMG_R)
    if EMG_L.size != n_out or EMG_R.size != n_out:
        raise RuntimeError('internal duration error')

    if return_spikes:
        info = dict(
            spikes_L=spL[:, n_burn:n_burn+n_out],
            spikes_R=spR[:, n_burn:n_burn+n_out],
            muap_dur_s=durs, gains=gains,
            burn_in_samples=n_burn, burn_in_s=n_burn*dt,
            duration_samples=n_out, duration_s=n_out*dt,
            common_drive_rms=common_meta['rms'],
            common_cortical_fraction=common_meta['cortical_fraction'],
            common_effective_coefficients=common_meta['coefficients'],
            common_drive_normalization=common_drive_normalization,
            common_fraction_interpretation=common_fraction_interpretation,
            snr_units=snr_units, intent_drift=intent_drift,
            drive_noise_snr_db_L=measured_snr_db(sigL, nL),
            drive_noise_snr_db_R=measured_snr_db(sigR, nR),
        )
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


def compute_cwt_for_coherence(sig, fs, fmax=128.0, dj=1/8, s0=None, J=None, w0=6):
    """Compute and cache the Morlet CWT ingredients used for coherence.

    This public helper lets the V3.3 cross-trial null reuse each trial's CWT
    rather than recomputing the transform for every off-diagonal L/R pairing.
    The returned dictionary is compatible with ``coherence_from_cached_cwts``.
    """
    sig = np.asarray(sig, float)
    if sig.ndim != 1:
        raise ValueError('sig must be a 1-D array')
    dt = 1.0/float(fs)
    if s0 is None:
        s0 = dt
    J = int(round(8/dj)) if J is None else int(J)
    _require_pycwt()
    mother = wavelet.Morlet(w0)
    W, scales, freqs_full, coi, *_ = wavelet.cwt(sig, dt, dj, s0, J, mother)
    fmask = freqs_full <= fmax
    return dict(
        W=W, scales=np.asarray(scales, float), freqs_full=np.asarray(freqs_full, float),
        freqs=np.asarray(freqs_full[fmask], float), period=np.asarray(1.0/freqs_full[fmask], float),
        coi=np.asarray(coi, float), fmask=np.asarray(fmask, bool), dt=float(dt), dj=float(dj),
        s0=float(s0), J=int(J), w0=float(w0), mother=mother, fs=float(fs), fmax=float(fmax),
    )


def coherence_from_cached_cwts(left, right):
    """Squared wavelet coherence from two cached CWT dictionaries.

    The two transforms must use the same time/frequency grid. Returns the same
    core arrays used by ``band_medians``: ``freqs, period, coi, Rsq``.
    """
    for key in ('dt','dj','s0','J','w0','fs'):
        if not np.isclose(left[key], right[key]):
            raise ValueError(f'CWT grids differ ({key})')
    if not np.allclose(left['scales'], right['scales']):
        raise ValueError('CWT scales differ')
    if not np.array_equal(left['fmask'], right['fmask']):
        raise ValueError('CWT frequency masks differ')
    Rsq_full = _wct_from_cwt(left['W'], right['W'], left['scales'], left['dt'], left['dj'])
    m = left['fmask']
    coi = np.minimum(left['coi'], right['coi'])
    return left['freqs'], left['period'], coi, Rsq_full[m, :]


def compute_wavelet_coherence(sig1, sig2, fs, fmax=128.0, dj=1/8, s0=None, J=None, w0=6):
    """Squared Morlet wavelet coherence (Grinsted et al. 2004) of two signals.

    Returns freqs, period, coi, Rsq (n_freqs x n_times, masked to <= fmax),
    dt, dj, s0, J, mother. Defaults (s0 = dt, J = 8/dj) match v1, so the
    frequency grid is unchanged. Use band_medians(Rsq, freqs, period=period,
    coi=coi) to exclude the cone of influence.
    """
    sig1 = np.asarray(sig1, float); sig2 = np.asarray(sig2, float)
    if sig1.ndim != 1 or sig1.shape != sig2.shape:
        raise ValueError('sig1 and sig2 must be equal-length 1-D arrays')
    L = compute_cwt_for_coherence(sig1, fs, fmax=fmax, dj=dj, s0=s0, J=J, w0=w0)
    R = compute_cwt_for_coherence(sig2, fs, fmax=fmax, dj=dj, s0=s0, J=J, w0=w0)
    freqs, period, coi, Rsq = coherence_from_cached_cwts(L, R)
    return freqs, period, coi, Rsq, L['dt'], L['dj'], L['s0'], L['J'], L['mother']

# ------------------------ Monte-Carlo null (with PB) ---------------------

def mc_null_threshold(sig_len, dt, dj, s0, J, mother,
                      alpha=0.05, num_simulations=300,
                      bp_low=5.0, bp_high=499.0,
                      fmax=128.0, rng=None, coi_mask=False):
    """Legacy v2 Gaussian-null frequency-median threshold.

    Kept only for backward compatibility. V3.3 primary inference uses
    same-condition cross-trial pairing/derangements (see trial_shuffle_null_v3.py).
    """
    _require_pycwt()
    rng = np.random.default_rng() if rng is None else rng
    b, a = butter(4, [bp_low/(0.5/dt), bp_high/(0.5/dt)], btype='band')
    tmp = rng.standard_normal(sig_len)
    _, scales, freqs_full, coi, *_ = wavelet.cwt(tmp, dt, dj, s0, J, mother)
    F = freqs_full.size
    fmask = freqs_full <= fmax
    inside = (1/freqs_full)[:, None] <= coi[None, :] if coi_mask else None
    median_coh = np.empty((num_simulations, F), dtype=float)
    for i in tqdm(range(num_simulations), desc='MC null', leave=False):
        n1 = zscore(filtfilt(b, a, rng.standard_normal(sig_len)))
        n2 = zscore(filtfilt(b, a, rng.standard_normal(sig_len)))
        X, *_ = wavelet.cwt(n1, dt, dj, s0, J, mother)
        Y, *_ = wavelet.cwt(n2, dt, dj, s0, J, mother)
        Rsq = _wct_from_cwt(X, Y, scales, dt, dj)
        if inside is not None:
            Rsq = np.where(inside, Rsq, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            median_coh[i, :] = np.nanmedian(Rsq, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        upper_full = np.nanpercentile(median_coh, 100*(1-alpha), axis=0)
    return freqs_full[fmask], upper_full[fmask]


def surrogate_wavelet_threshold(sig1, sig2, fs, alpha=0.05, num_simulations=300,
                                mode='global_max', fmax=128.0, dj=1/8,
                                s0=None, J=None, w0=6, rng=None, progress=True):
    """Legacy phase-randomized wavelet-coherence diagnostic.

    This preserves each signal periodogram but destroys the task-locked burst
    structure. It is therefore not used for V3.3 primary null inference. Use
    same-condition cross-trial pairing/derangements instead.

    Parameters
    ----------
    mode : {'global_max','pointwise_frequency','frequency_median'}
        global_max: one threshold from the max coherence anywhere inside the
            COI for each surrogate (FWER-style control over the displayed map).
        pointwise_frequency: frequency-specific pointwise thresholds formed
            from all inside-COI surrogate time samples (uncorrected across the
            full map; useful descriptively, not a global significance claim).
        frequency_median: frequency-specific threshold for the *time-median*
            statistic; use with frequency/band summaries, not map pixels.

    Returns
    -------
    freqs, threshold, meta
        threshold is scalar for global_max, otherwise one value/frequency.
    """
    _require_pycwt()
    x = np.asarray(sig1, float); y = np.asarray(sig2, float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError('sig1 and sig2 must be equal-length 1-D arrays')
    if mode not in {'global_max','pointwise_frequency','frequency_median'}:
        raise ValueError('invalid surrogate threshold mode')
    rng = np.random.default_rng() if rng is None else rng
    dt = 1.0/float(fs)
    if s0 is None:
        s0 = dt
    J = int(round(8/dj)) if J is None else int(J)
    mother = wavelet.Morlet(w0)
    X0, scales, freqs_full, coi, *_ = wavelet.cwt(x, dt, dj, s0, J, mother)
    fmask = freqs_full <= fmax
    freqs = freqs_full[fmask]
    periods = (1.0/freqs_full)[fmask]
    inside = periods[:, None] <= coi[None, :]

    if mode == 'global_max':
        stats = np.empty(num_simulations, float)
    else:
        stats = np.empty((num_simulations, freqs.size), float)
        pointwise_pool = [[] for _ in range(freqs.size)] if mode == 'pointwise_frequency' else None

    iterator = tqdm(range(num_simulations), desc='Legacy phase-surrogate null', leave=False) if progress else range(num_simulations)
    for i in iterator:
        xs = phase_randomize(x, rng)
        ys = phase_randomize(y, rng)
        X, *_ = wavelet.cwt(xs, dt, dj, s0, J, mother)
        Y, *_ = wavelet.cwt(ys, dt, dj, s0, J, mother)
        R = _wct_from_cwt(X, Y, scales, dt, dj)[fmask]
        masked = np.where(inside, R, np.nan)
        if mode == 'global_max':
            stats[i] = np.nanmax(masked)
        elif mode == 'frequency_median':
            stats[i, :] = np.nanmedian(masked, axis=1)
        else:
            # Pooling time samples gives a per-frequency pointwise null.
            for fi in range(freqs.size):
                vals = masked[fi]
                pointwise_pool[fi].append(vals[np.isfinite(vals)])

    q = 100.0*(1.0-alpha)
    if mode == 'global_max':
        threshold = float(np.nanpercentile(stats, q))
    elif mode == 'frequency_median':
        threshold = np.nanpercentile(stats, q, axis=0)
    else:
        threshold = np.array([
            np.nanpercentile(np.concatenate(v), q) if any(a.size for a in v) else np.nan
            for v in pointwise_pool
        ], float)
    meta = dict(alpha=float(alpha), num_simulations=int(num_simulations), mode=mode,
                surrogate='legacy_independent_phase_randomization', coi_mask=True,
                fmax=float(fmax), dj=float(dj), s0=float(s0), J=int(J), w0=float(w0))
    return freqs, threshold, meta

