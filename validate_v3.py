"""Validation checks for EMG Simulator v3.3.0.

V3.3 restores the neuron-continuity/window-artifact regression checks from v2,
adds the V3 scientific corrections, verifies Matviyenko remains analysis-only,
and validates helpers used by the same-condition cross-trial null.
"""
from pathlib import Path
import numpy as np
import functions as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
log = []


def say(msg=""):
    print(msg); log.append(str(msg))


def check(name, condition, detail=""):
    status = "PASS" if bool(condition) else "FAIL"
    say(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(condition)


def v1_lif_reference(Im, Vth_base, V_reset, V_e, Rm, tau_m, dt, v_mem_start,
                     min_isi_ms=18.0, dVth=0.003, tau_adapt_ms=250.0):
    """Local copy of the v1 single-call LIF update rule used for regression.

    We keep this tiny reference here so validation does not require importing
    legacy/functions_v1.py (which hard-imports pycwt even for LIF-only tests).
    """
    Im=np.asarray(Im,float); L=Im.size
    V=np.empty(L+1,float); V[0]=v_mem_start
    spikes=np.zeros(L,float); Vth=float(Vth_base); last=-10**9
    min_isi=int(round(min_isi_ms/1000.0/dt)); decay=np.exp(-dt/(tau_adapt_ms/1000.0))
    for t in range(L):
        Vth=Vth_base+(Vth-Vth_base)*decay
        if (t-last)<min_isi:
            V[t+1]=V_reset
        elif V[t]>=Vth:
            spikes[t]=1.0; last=t; V[t+1]=V_reset; Vth+=dVth
        else:
            V[t+1]=V[t]+(dt/tau_m)*(-(V[t]-V_e)+Im[t]*Rm)
    return spikes,V[-1]


say(f"EMG Simulator validation - functions.py v{F.__version__}")
say("="*78)
results = []
rng = np.random.default_rng(123)

# 1. Common-drive normalization.
say("\n1. Fixed-RMS common drive")
c = rng.standard_normal(100_000); s = rng.standard_normal(100_000)
mixtures = [(6.0, 0.0, 1.00), (4.875, 1.625, 0.75), (3.25, 3.25, 0.50)]
for cw, sw, expected_p in mixtures:
    y, meta = F.compose_common_drive(c, s, cw, sw, target_rms=6.0,
                                     normalization='rms', fraction_interpretation='amplitude')
    ok = abs(np.std(y)-6.0) < 1e-10 and abs(meta['cortical_fraction']-expected_p) < 1e-12
    results.append(check(f"mixture {expected_p:.0%}/{1-expected_p:.0%}", ok,
                         f"RMS={np.std(y):.12f}, amplitude fraction={meta['cortical_fraction']:.3f}"))
# 75/25 amplitude -> 90/10 variance for independent equal-variance sources.
varfrac = 0.75**2/(0.75**2 + 0.25**2)
results.append(check("75/25 amplitude implies 90/10 variance under independence", abs(varfrac-0.9)<1e-12,
                     f"cortical variance fraction={varfrac:.3f}"))
say("   Legacy coefficient RMS values (diagnostic only):")
for cw, sw, expected_p in mixtures:
    y, _ = F.compose_common_drive(c, s, cw, sw, normalization='legacy_coefficients')
    say(f"      {expected_p:.0%}/{1-expected_p:.0%}: {np.std(y):.4f}")

# 2. SNR units.
say("\n2. Explicit dB SNR")
sig = F.zscore(rng.standard_normal(100_000))
for target in (8.0, 20.0, 40.0):
    nz = F.zscore(rng.standard_normal(sig.size))*F._snr_noise_sd(np.std(sig), target, 'db')
    measured = F.measured_snr_db(sig, nz)
    results.append(check(f"requested {target:.0f} dB", abs(measured-target) < 1e-8,
                         f"measured={measured:.8f} dB"))

# 3. Restore v2 LIF regression checks.
say("\n3. LIF continuity / 100-ms reset regression")
r0 = np.random.default_rng(0); dt = 2e-4; Ltest = 20_000
I = 1.2e-9*10*F.zscore(F.butter_filter_low(r0.standard_normal(Ltest), 100, 5000))
Th = np.linspace(-0.055, -0.040, 5)
same = []
for th in Th:
    s1, _ = v1_lif_reference(I, th, -0.08, -0.075, 10e6, 10e-3, dt, -0.08)
    s2, _ = F.lif_population(I[None], np.array([0]), np.array([th]), -0.08, -0.075, 10e6, 10e-3, dt)
    same.append(np.array_equal(s1[:-1], s2[0, :-1].astype(float)))
results.append(check("single continuous call matches v1 local LIF rule", all(same), "5 thresholds"))
full, _ = F.lif_population(I[None], np.zeros(5, int), Th, -0.08, -0.075, 10e6, 10e-3, dt)
st, parts = None, []
for w in range(0, Ltest, 500):
    pp, st = F.lif_population(I[None, w:w+500], np.zeros(5, int), Th,
                              -0.08, -0.075, 10e6, 10e-3, dt, state=st)
    parts.append(pp)
results.append(check("100-ms chunks with carried state equal continuous integration",
                     np.array_equal(full, np.hstack(parts))))

# Direct no-reset test: deliberately resetting state every 100 ms must differ
# from the continuous result, while carried-state chunking above is identical.
reset_parts = []
for w in range(0, Ltest, 500):
    pp, _ = F.lif_population(I[None, w:w+500], np.zeros(5, int), Th,
                              -0.08, -0.075, 10e6, 10e-3, dt, state=None)
    reset_parts.append(pp)
reset_sp = np.hstack(reset_parts)
results.append(check("reset-every-100ms artifact path is not the V3 result",
                     not np.array_equal(full, reset_sp)))

# 4. Exact duration + reproducibility compact generator.
say("\n4. Exact usable duration and reproducibility")
P = dict(
    motorunits_max=8, n_fibers=20, FreqInputTotal=100, SNR=40, SNRE=40, SNRF=8,
    V_reset=-0.080, V_e=-0.075, dt=0.0002, T_end=2.0, const_current=10.0,
    common_mod_subcortical=1.625, freq_left=0.5, freq_right=1.0,
    phase_left=0.0, phase_right=np.pi/2, Intent_variability=0.01, window_size=0.1,
    cross_talk_R2L=0.20, cross_talk_L2R=0.01, common_mod_cortical=4.875,
    Vth_min=-0.055, Vth_max=-0.040, I_scale=1.5e-9, Rm=10e6, tau_m=10e-3,
    seed=43, cortical_high_hz=60, muap_dur_s=0.018, muap_jitter_s=0.004,
    use_dog_muaps=True, use_lif_adaptation=True, noise_type='pink', burn_in_s=0.5,
    snr_units='db', common_drive_normalization='rms', common_drive_target_rms=6.0,
    intent_drift='bounded', return_spikes=True,
)
L1, R1, fs, info1 = F.generate_modulated_EMG_physiological_upgraded(**P)
L2, R2, fs2, info2 = F.generate_modulated_EMG_physiological_upgraded(**P)
expected_n = int(round(P['T_end']/P['dt']))
results.append(check("returned duration", len(L1) == expected_n == len(R1),
                     f"n={len(L1)}, expected={expected_n}, seconds={len(L1)/fs:.6f}"))
results.append(check("bitwise reproducibility", np.array_equal(L1,L2) and np.array_equal(R1,R2)))
results.append(check("common RMS reported by generator", abs(info1['common_drive_rms']-6.0)<1e-10,
                     f"RMS={info1['common_drive_rms']:.12f}"))
results.append(check("generator dB SNR", abs(info1['drive_noise_snr_db_L']-40.0)<1e-8,
                     f"left={info1['drive_noise_snr_db_L']:.8f} dB"))

# Population firing should not be locked to the 100-ms integration-window grid.
def window_lock(sp, fs, period_s=0.1):
    pop = sp.sum(0).astype(float); w = int(round(period_s*fs))
    nW = pop.size//w
    if nW < 3: return np.nan
    # ten equal phase bins per fold
    pop = pop[:nW*w].reshape(nW, 10, w//10).sum((0,2))
    return float(pop[0]/(pop.mean()+1e-12))
lock100 = window_lock(info1['spikes_L'], fs, 0.1)
near = np.array([window_lock(info1['spikes_L'], fs, p) for p in (0.08,0.09,0.11,0.12)])
# This is a modest smoke diagnostic, not a hypothesis test. The robust state
# equality above is the exact regression guarantee.
results.append(check("no conspicuous 100-ms population burst",
                     np.isfinite(lock100) and lock100 < 1.8,
                     f"100-ms first-bin/mean={lock100:.3f}; nearby={np.round(near,3)}"))

# 5. Bounded intent variability.
say("\n5. Bounded intent variability")
d = F._bounded_intent_variation(100_000, 5000, 0.01, np.random.default_rng(9), cutoff_hz=0.25)
results.append(check("1% RMS interpretation", abs(np.std(d)-0.01)<1e-8,
                     f"RMS={np.std(d):.8f}, range=[{d.min():.4f},{d.max():.4f}]"))
results.append(check("no secular ramp", abs(np.polyfit(np.arange(d.size),d,1)[0]*d.size)<0.01,
                     "end-to-end fitted drift < 1 percentage point"))

# 6. Legacy Fourier phase diagnostic (not V3.3 primary inference).
say("\n6. Legacy Fourier phase-randomized diagnostic")
x = rng.standard_normal(4096); xs = F.phase_randomize(x, np.random.default_rng(2))
p0 = np.abs(np.fft.rfft(x))**2; p1 = np.abs(np.fft.rfft(xs))**2
rel = np.max(np.abs(p1-p0)/(np.maximum(p0,1e-12)))
results.append(check("plain phase randomization preserves periodogram", rel<1e-10,
                     f"max relative error={rel:.2e}"))

# 7. Cross-trial derangement helper.
say("\n7. Same-condition cross-trial null helper")
dr = F.random_derangement(15, np.random.default_rng(4))
results.append(check("derangement has no fixed trial labels", np.all(dr != np.arange(15))))
results.append(check("derangement is a true permutation", np.array_equal(np.sort(dr), np.arange(15))))

# 8. Correct complex wavelet smoothing.
say("\n8. Complex wavelet smoothing")
S,n,dtw,dj = 20,600,1e-3,1/8; scales = 2*dtw*2**(np.arange(S)*dj)
Wt = rng.standard_normal((S,n)) + 1j*rng.standard_normal((S,n))
out = F.smoothwavelet(Wt,dtw,None,dj,scales)
npad=1024; ref=np.zeros_like(Wt)
for i in range(S):
    tt=np.arange(-npad//2,npad//2); g=np.exp(-0.5*(tt/(scales[i]/dtw))**2); g/=g.sum()
    xx=np.zeros(npad,complex); xx[:n]=Wt[i]
    ref[i]=np.fft.ifft(np.fft.fft(xx)*np.fft.fft(np.fft.ifftshift(g)))[:n]
k=np.array([0.4,1,1,1,0.4]); k/=k.sum()
ref=np.apply_along_axis(lambda col: np.convolve(col,k,mode='same'),0,ref)
err=np.max(np.abs(out-ref))
results.append(check("time + scale smoothing", err<1e-6, f"brute-force max error={err:.2e}"))
results.append(check("imaginary cross-spectrum retained", np.max(np.abs(np.imag(out)))>1e-6))

# 9. Exact Matviyenko port (sensitivity analysis only).
say("\n9. Matviyenko taper port (sensitivity only)")
expected_k8_n8=np.array([
0.00254930175329626,0.03892177913110023,0.19531240185940340,0.5,
0.8046875981405965,0.9610782208688997,0.9974506982467037,0.9999969440797141,
0.9999969440797141,0.9974506982467037,0.9610782208688997,0.8046875981405965,
0.5,0.19531240185940340,0.03892177913110023,0.00254930175329626])
w=F.matviyenko(8,8)
results.append(check("MATLAB-equation reference values", np.allclose(w,expected_k8_n8,rtol=0,atol=5e-15),
                     f"max abs error={np.max(np.abs(w-expected_k8_n8)):.2e}"))
results.append(check("symmetric 2*n output", w.size==16 and np.array_equal(w,w[::-1])))
results.append(check("valid coefficient orders K=2..13", all(F.matviyenko(kv,7).size==14 for kv in range(2,14))))
try:
    F.matviyenko_window(5001,K=8); odd_rejected=False
except ValueError:
    odd_rejected=True
results.append(check("odd lengths rejected rather than silently altered", odd_rejected))

# 10. Coherence preprocessing / analysis-only taper separation.
say("\n10. Coherence preprocessing")
t=np.arange(5000)/1000.0
raw=0.4*t+np.sin(2*np.pi*20*t)+0.1*rng.standard_normal(t.size)
raw_copy=raw.copy()
base=F.preprocess_emg_for_coherence(raw,1000,detrend=True,taper=None)
tapered=F.apply_analysis_taper(base,taper='matviyenko',taper_kwargs={'K':8})
tapered2=F.apply_analysis_taper(base,taper='matviyenko',taper_kwargs={'K':8})
results.append(check("primary untapered preprocessing is z-scored", abs(base.mean())<1e-10 and abs(base.std()-1)<1e-10))
results.append(check("deterministic Matviyenko sensitivity preprocessing", np.array_equal(tapered,tapered2)))
results.append(check("analysis taper leaves source EMG unchanged", np.array_equal(raw,raw_copy)))
results.append(check("untapered primary differs from tapered sensitivity input", not np.array_equal(base,tapered)))

# 11. Optional pycwt runtime/cache equivalence.
say("\n11. Optional pycwt runtime")
if F.wavelet is None:
    say("[INFO] pycwt not installed; generator/scientific tests passed and CWT cache equivalence skipped.")
else:
    say(f"[INFO] pycwt available: {getattr(F.wavelet,'__version__','version unknown')}")
    a=F.zscore(rng.standard_normal(3000)); b=F.zscore(rng.standard_normal(3000))
    fq,pd_,coi,R0,*_=F.compute_wavelet_coherence(a,b,1000,fmax=100,dj=1/8,w0=6)
    ca=F.compute_cwt_for_coherence(a,1000,fmax=100,dj=1/8,w0=6)
    cb=F.compute_cwt_for_coherence(b,1000,fmax=100,dj=1/8,w0=6)
    fq2,pd2,coi2,R1=F.coherence_from_cached_cwts(ca,cb)
    results.append(check("cached CWT pairing equals direct coherence",
                         np.allclose(fq,fq2) and np.allclose(pd_,pd2) and np.allclose(coi,coi2) and np.allclose(R0,R1,equal_nan=True)))

passed=sum(results); total=len(results)
say("\n"+"="*78); say(f"RESULT: {passed}/{total} checks passed")
report=OUT/'v3_validation_report.txt'; report.write_text("\n".join(log)+"\n")
say(f"Saved: {report}")
if passed != total:
    raise SystemExit(1)
