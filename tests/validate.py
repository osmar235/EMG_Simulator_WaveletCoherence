"""Quick self-test of the installation (about 10-20 s).

    python tests/validate.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions as F

results = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    results.append(bool(ok))


print(f"Self-test, functions.py v{F.__version__}\n")
rng = np.random.default_rng(123)

# Common drive: fixed total RMS for every mixture
c, s = rng.standard_normal(50_000), rng.standard_normal(50_000)
for cw, sw in ((6.0, 0.0), (4.875, 1.625), (3.25, 3.25)):
    y, meta = F.compose_common_drive(c, s, cw, sw, target_rms=6.0)
    check(f"common drive {cw}/{sw} has RMS 6", abs(np.std(y) - 6.0) < 1e-9)

# SNR in dB
sig = F.zscore(rng.standard_normal(50_000))
nz = F.zscore(rng.standard_normal(sig.size))*F._snr_noise_sd(1.0, 20.0)
check("SNR 20 dB", abs(F.measured_snr_db(sig, nz) - 20.0) < 1e-6)

# LIF: integration in chunks with carried state equals continuous integration
dt, L = 2e-4, 10_000
I = 1.2e-9*10*F.zscore(F.butter_filter_low(rng.standard_normal(L), 100, 5000))
Th = np.linspace(-0.055, -0.040, 5)
full, _ = F.lif_population(I[None], np.zeros(5, int), Th, -0.08, -0.075, 10e6, 10e-3, dt)
st, parts = None, []
for w in range(0, L, 500):
    p, st = F.lif_population(I[None, w:w+500], np.zeros(5, int), Th, -0.08, -0.075, 10e6, 10e-3, dt, state=st)
    parts.append(p)
check("LIF chunked == continuous", np.array_equal(full, np.hstack(parts)))

# Simulator: duration, reproducibility
kw = dict(T_end=1.0, burn_in_s=0.2, motorunits_max=20, n_fibers=40)
a = F.generate_modulated_EMG_physiological_upgraded(seed=5, **kw)
b = F.generate_modulated_EMG_physiological_upgraded(seed=5, **kw)
check("returned duration", a[0].size == int(round(1.0/0.0002)))
check("same seed -> identical output", np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

# Wavelet smoothing against brute force (Gaussian in time, 0.6-octave boxcar in scale)
S, n, dtw, dj = 20, 600, 1e-3, 1/8
scales = 2*dtw*2**(np.arange(S)*dj)
Wt = rng.standard_normal((S, n)) + 1j*rng.standard_normal((S, n))
out = F.smoothwavelet(Wt, dtw, None, dj, scales)
npad, ref = 1024, np.zeros_like(Wt)
for i in range(S):
    tt = np.arange(-npad//2, npad//2); g = np.exp(-0.5*(tt/(scales[i]/dtw))**2); g /= g.sum()
    xx = np.zeros(npad, complex); xx[:n] = Wt[i]
    ref[i] = np.fft.ifft(np.fft.fft(xx)*np.fft.fft(np.fft.ifftshift(g)))[:n]
k = np.array([0.4, 1, 1, 1, 0.4]); k /= k.sum()
ref = np.apply_along_axis(lambda col: np.convolve(col, k, mode="same"), 0, ref)
check("wavelet smoothing", np.max(np.abs(out - ref)) < 1e-6)

# Matviyenko window reference values (K=8, n=8)
ref8 = np.array([0.00254930175329626, 0.03892177913110023, 0.19531240185940340, 0.5,
                 0.8046875981405965, 0.9610782208688997, 0.9974506982467037, 0.9999969440797141])
w = F.matviyenko(8, 8)
check("Matviyenko window", np.allclose(w[:8], ref8, atol=5e-15) and np.array_equal(w, w[::-1]))

if F.wavelet is None:
    print("\n[INFO] pycwt not installed: coherence tests skipped (pip install -r requirements.txt)")
else:
    from scipy.signal import hilbert
    fs, N = 1000, 6000
    r2 = np.random.default_rng(3)
    com = F.butter_filter_high(F.butter_filter_low(r2.standard_normal(N), 30, fs), 20, fs); com /= com.std()
    x = com + 0.7*r2.standard_normal(N)
    y_in = com + 0.7*r2.standard_normal(N)
    y_90 = np.imag(hilbert(com)) + 0.7*r2.standard_normal(N)
    y_ind = r2.standard_normal(N)
    vals = {}
    for lab, y in (("in-phase", y_in), ("90-deg lag", y_90), ("independent", y_ind)):
        f, p, coi, R, *_ = F.compute_wavelet_coherence(x, y, fs)
        vals[lab] = F.band_medians(R, f, ((20, 30),), period=p, coi=coi)["20-30"]
    check("shared 20-30 Hz component detected (in phase)", vals["in-phase"] > 0.8, f"{vals['in-phase']:.3f}")
    check("shared 20-30 Hz component detected (90 deg lag)", vals["90-deg lag"] > 0.8, f"{vals['90-deg lag']:.3f}")
    check("independent signals lower", vals["independent"] < 0.6, f"{vals['independent']:.3f}")
    ca, cb = F.compute_cwt_for_coherence(x, fs), F.compute_cwt_for_coherence(y_in, fs)
    R1 = F.coherence_from_cached_cwts(ca, cb)[3]
    R0 = F.compute_wavelet_coherence(x, y_in, fs)[3]
    check("cached CWT pairing == direct", np.allclose(R0, R1, equal_nan=True))

print(f"\nRESULT: {sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
