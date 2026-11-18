# validate_fixes.py — checks for the v2.0.0 bug-fix release of the EMG simulator.
#
# Compares the corrected functions.py with the published code (legacy/functions_v1.py):
#   1. LIF motor neurons: v2 reproduces v1 inside a single call, and chunked
#      integration with carried state equals continuous integration.
#   2. Wavelet coherence: smoothing matches a brute-force implementation of
#      Grinsted et al. (2004) and recovers known in-phase and 90-degree-lagged coupling.
#   3. Generator: reproducibility from `seed` and the window-locked firing artifact.
#   4. Published configuration (1:2 bimanual, R->L 20 %, 75/25 cortical/subcortical,
#      13-60 Hz): left-right wavelet coherence per band for
#        A = v1 generator + v1 coherence   (published pipeline)
#        B = v1 generator + v2 coherence   (effect of the coherence fix alone)
#        C = v2 generator + v2 coherence   (fully corrected)
#      plus a trial-shuffled surrogate (left EMG of one seed vs right EMG of another seed:
#      no shared drive, only the shared task timing = the floor that common drive must exceed).
#      Note: v1 is not reproducible run-to-run (its MU/EMG noise used unseeded generators).
#
# USAGE (from the repository folder):
#   python validate_fixes.py            # 6 seeds, ~3-4 min
#   python validate_fixes.py 12         # more seeds
# Writes results/validation_report.txt and results/fig_validation.png

import os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "legacy"))
import functions as F           # v2 (fixed)
import functions_v1 as V1       # v1 (published)

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
LOG = []
def say(s=""):
    print(s); LOG.append(s)

PAPER = dict(motorunits_max=72, n_fibers=346, FreqInputTotal=100, SNR=40, SNRE=40, SNRF=8, T_end=18,
             V_reset=-0.080, V_e=-0.075, Vth_min=-0.055, Vth_max=-0.040, Rm=10e6, tau_m=10e-3, dt=0.0002,
             const_current=10.0, freq_left=0.5, freq_right=1.0, phase_left=0.0, phase_right=np.pi/2,
             window_size=0.1, Intent_variability=0.01, cross_talk_R2L=0.20, cross_talk_L2R=0.01,
             common_mod_cortical=4.875, common_mod_subcortical=1.625, I_scale=1.5e-9, cortical_high_hz=60,
             muap_dur_s=0.018, muap_jitter_s=0.004, use_dog_muaps=True, use_lif_adaptation=True,
             noise_type="pink")
BANDS = ((5, 13), (13, 30), (30, 60), (60, 100))
BNAME = ["alpha 5-13", "beta 13-30", "l-gamma 30-60", "h-gamma 60-100"]
FS_W = 1000.0

def ok(flag):
    return "PASS" if flag else "FAIL"

# ---------------------------------------------------------------- 1. LIF
say(f"EMG simulator validation — functions.py v{F.__version__} vs legacy v1")
say("=" * 78)
say("\n1. LIF motor neurons")
rng = np.random.default_rng(0); dt = 2e-4; Ltest = 20000
I = 1.2e-9*10*F.zscore(F.butter_filter_low(rng.standard_normal(Ltest), 100, 5000))
Th = np.linspace(-0.055, -0.040, 5)
same = []
for th in Th:
    s1, _ = V1.lif_window_with_adaptation(I, th, -0.08, -0.075, 10e6, 10e-3, dt, -0.08)
    s2, _ = F.lif_population(I[None], np.array([0]), np.array([th]), -0.08, -0.075, 10e6, 10e-3, dt)
    same.append(np.array_equal(s1[:-1], s2[0, :-1].astype(float)))
say(f"   single call identical to v1 (5 thresholds): {ok(all(same))}")
full, _ = F.lif_population(I[None], np.zeros(5, int), Th, -0.08, -0.075, 10e6, 10e-3, dt)
st, parts = None, []
for w in range(0, Ltest, 500):
    p, st = F.lif_population(I[None, w:w+500], np.zeros(5, int), Th, -0.08, -0.075, 10e6, 10e-3, dt, state=st)
    parts.append(p)
say(f"   100-ms chunks with carried state == continuous: {ok(np.array_equal(full, np.hstack(parts)))}")

# ---------------------------------------------------------------- 2. coherence
say("\n2. Wavelet coherence")
S, n, dtw, dj = 20, 600, 1e-3, 1/8
scales = 2*dtw*2**(np.arange(S)*dj)
Wt = rng.standard_normal((S, n)) + 1j*rng.standard_normal((S, n))
out = F.smoothwavelet(Wt, dtw, None, dj, scales)
npad = 1024; ref = np.zeros_like(Wt)
for i in range(S):
    tt = np.arange(-npad//2, npad//2); g = np.exp(-0.5*(tt/(scales[i]/dtw))**2); g /= g.sum()
    x = np.zeros(npad, complex); x[:n] = Wt[i]
    ref[i] = np.fft.ifft(np.fft.fft(x)*np.fft.fft(np.fft.ifftshift(g)))[:n]
k = np.array([0.4, 1, 1, 1, 0.4]); k /= k.sum()
ref = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, ref)
err = np.abs(out - ref).max()
say(f"   smoothing = Gaussian(time) + 0.6-octave boxcar(scale), brute force max error {err:.1e}: {ok(err < 1e-6)}")

fs = 1000; nn = 8000; r2 = np.random.default_rng(3)
c = F.butter_filter_high(F.butter_filter_low(r2.standard_normal(nn), 30, fs), 20, fs); c /= c.std()
a = c + 0.7*r2.standard_normal(nn)
b_in = c + 0.7*r2.standard_normal(nn)
b_90 = np.imag(hilbert(c)) + 0.7*r2.standard_normal(nn)
known = {}
for lab, b in (("in-phase", b_in), ("90-deg lag", b_90)):
    f2, p2, coi2, R2, *_ = F.compute_wavelet_coherence(a, b, fs)
    f1, p1, coi1, R1, *_ = V1.compute_wavelet_coherence(a, b, fs)
    v2b = F.band_medians(R2, f2, ((20, 30),), period=p2, coi=coi2)["20-30"]
    v1b = F.band_medians(R1, f1, ((20, 30),), period=p1, coi=coi1)["20-30"]
    known[lab] = (v1b, v2b)
    say(f"   shared 20-30 Hz component, {lab:10s}: coherence v1 {v1b:.3f} | v2 {v2b:.3f}")
say(f"   v2 detects lagged coupling (>0.8): {ok(known['90-deg lag'][1] > 0.8)};  "
    f"v1 misses it: {known['90-deg lag'][0]:.3f}")

# ---------------------------------------------------------------- helpers
def run_v1(seed):
    calls = []
    orig = V1.lif_window_with_adaptation
    def spy(*a, **k):
        sp, v = orig(*a, **k); calls.append(sp); return sp, v
    V1.lif_window_with_adaptation = spy
    try:
        np.random.seed(seed)
        L, R, fs = V1.generate_modulated_EMG_physiological_upgraded(seed=seed, **PAPER)
    finally:
        V1.lif_window_with_adaptation = orig
    nMU = PAPER["motorunits_max"]; nW = len(calls)//(2*nMU)
    spL = np.hstack([np.vstack(calls[w*2*nMU:w*2*nMU+nMU]) for w in range(nW)])
    spR = np.hstack([np.vstack(calls[w*2*nMU+nMU:(w+1)*2*nMU]) for w in range(nW)])
    trim = min(5000, spL.shape[1]//10)
    return L, R, fs, spL[:, trim:] > 0, spR[:, trim:] > 0

def run_v2(seed, **kw):
    L, R, fs, info = F.generate_modulated_EMG_physiological_upgraded(seed=seed, return_spikes=True, **PAPER, **kw)
    return L, R, fs, info["spikes_L"], info["spikes_R"]

def window_lock(sp, fs, trim, period_s=0.1):
    """Population firing in the first 10 % of each window / mean (1.0 = none).
    Folding starts at the true window grid (the trim offset is removed)."""
    pop = sp.sum(0).astype(float)
    w = int(round(period_s*fs)); off = (-trim) % w
    pop = pop[off:]; nW = pop.size//w
    fold = pop[:nW*w].reshape(nW, 10, w//10).sum((0, 2))
    return fold[0]/fold.mean(), fold/fold.mean()

def lock_null(sp, fs, trim):
    """Same index for fold periods of 80-120 ms (excluding 95-105 ms) = chance level."""
    vals = [window_lock(sp, fs, trim, P/fs)[0] for P in range(400, 601, 10) if abs(P-500) > 25]
    return np.mean(vals), np.std(vals, ddof=1)

def isi_cv(sp, fs):
    cv = []
    for row in sp:
        t = np.flatnonzero(row)/fs
        if t.size > 5:
            d = np.diff(t); cv.append(d.std()/d.mean())
    return np.median(cv) if cv else np.nan

def prep(x, fs):
    b = F.simulate_bipolar_emg_spatial(x, fs, tau_ms=1.2, lp_hz=180.0)
    return F.zscore(F.bandpass_emg(F.resample_to_fs(b, fs, FS_W), FS_W))

def coh(mod, x, y):
    f, p, coi, R, *_ = mod.compute_wavelet_coherence(x, y, fs=FS_W, fmax=128.0, dj=1/8)
    bm = F.band_medians(R, f, BANDS, period=p, coi=coi)
    inside = p[:, None] <= coi[None, :]
    spec = np.array([np.nanmean(np.where(inside[i], R[i], np.nan)) for i in range(f.size)])
    return np.array([bm[f"{a}-{b}"] for a, b in BANDS]), f, spec

# ---------------------------------------------------------------- 3. generator
say("\n3. Generator (published configuration)")
L_a, R_a, fs_sim, _, _ = run_v2(7); L_b, R_b, _, _, _ = run_v2(7)
say(f"   same seed -> identical output: {ok(np.array_equal(L_a, L_b) and np.array_equal(R_a, R_b))}")
t0 = time.time(); run_v2(8); t_v2 = time.time() - t0
t0 = time.time(); run_v1(8); t_v1 = time.time() - t0
say(f"   run time per 18-s trial: v1 {t_v1:.1f} s | v2 {t_v2:.1f} s")

# ---------------------------------------------------------------- 4. paper configuration
say(f"\n4. Published configuration, {N_SEEDS} seeds  (window lock: 1.00 = no artifact; "
    f"z = distance from chance folds)")
res = {k: [] for k in ("A0", "A", "B", "C", "Cw", "floor1", "floor2")}
lock = {"v1": [], "v2": []}; cvs = {"v1": [], "v2": []}
specs = {"A": [], "B": [], "C": [], "floor2": []}; folds = {"v1": [], "v2": []}
trim = min(5000, int(PAPER["T_end"]/PAPER["dt"])//10)
prevR = {}
for s in range(N_SEEDS):
    seed = 100 + s
    L1, R1, fs_sim, sL1, sR1 = run_v1(seed)
    L2, R2, _, sL2, sR2 = run_v2(seed)
    L3, R3, _, _, _ = run_v2(seed, intent_drift="window_legacy")
    for tag, sp in (("v1", sL1), ("v2", sL2)):
        li, fo = window_lock(sp, fs_sim, trim); m0, sd0 = lock_null(sp, fs_sim, trim)
        lock[tag].append((li, (li - m0)/sd0)); folds[tag].append(fo)
        cvs[tag].append(isi_cv(sp, fs_sim))
    x1, y1 = prep(L1, fs_sim), prep(R1, fs_sim)
    x2, y2 = prep(L2, fs_sim), prep(R2, fs_sim)
    f0_, p0_, c0_, R0_, *_ = V1.compute_wavelet_coherence(x1, y1, fs=FS_W, fmax=128.0, dj=1/8)
    b0_ = V1.band_medians(R0_, f0_); res["A0"].append(np.array([b0_[f"{a}-{b}"] for a, b in BANDS]))
    for key, mod, x, y in (("A", V1, x1, y1), ("B", F, x1, y1), ("C", F, x2, y2)):
        bm, f, sp = coh(mod, x, y); res[key].append(bm); specs.setdefault(key, []).append(sp)
    res["Cw"].append(coh(F, prep(L3, fs_sim), prep(R3, fs_sim))[0])
    if s > 0:   # trial-shuffled surrogate: left of this seed vs right of the previous seed
        res["floor1"].append(coh(V1, x1, prevR["v1"])[0])
        bm, f, sp = coh(F, x2, prevR["v2"]); res["floor2"].append(bm); specs["floor2"].append(sp)
    prevR = {"v1": y1, "v2": y2}
    say(f"   seed {seed}: lock v1 {lock['v1'][-1][0]:.2f} (z {lock['v1'][-1][1]:+.1f}) | "
        f"v2 {lock['v2'][-1][0]:.2f} (z {lock['v2'][-1][1]:+.1f})")

l1 = np.array(lock["v1"]); l2 = np.array(lock["v2"])
say(f"\n   Window-locked firing (left limb, mean): v1 {l1[:,0].mean():.2f}x (z {l1[:,1].mean():+.1f}) | "
    f"v2 {l2[:,0].mean():.2f}x (z {l2[:,1].mean():+.1f})")
say(f"   v2 has no window artifact (|mean z| < 2): {ok(abs(l2[:,1].mean()) < 2)}")
say(f"   Median ISI CV (left): v1 {np.nanmean(cvs['v1']):.2f} | v2 {np.nanmean(cvs['v2']):.2f}  "
    f"(both high because every MU of a limb receives the same input; see CHANGELOG)")

def row(lab, arr):
    a = np.array(arr); m = a.mean(0); sd = a.std(0, ddof=1) if len(a) > 1 else np.zeros_like(m)
    return f"   {lab:44s}" + "".join(f"{m[i]:.3f}±{sd[i]:.3f}  " for i in range(len(BANDS)))
say("\n   Left-right wavelet coherence, band medians inside the COI (mean ± SD across seeds)")
say("   " + " "*44 + "".join(f"{b:15s}" for b in BNAME))
say(row("A0 published pipeline (v1, no COI mask)", res["A0"]))
say(row("A  v1 generator + v1 coherence (published)", res["A"]))
say(row("B  v1 generator + v2 coherence", res["B"]))
say(row("C  v2 generator + v2 coherence (corrected)", res["C"]))
say(row("C' as C, legacy per-window intent ramp", res["Cw"]))
if res["floor1"]:
    say(row("shuffled surrogate, v1 coherence", res["floor1"]))
    say(row("shuffled surrogate, v2 coherence", res["floor2"]))
A, B, C = (np.array(res[k]) for k in "ABC")
say("\n   Change in band ordering / effect sizes:")
say(f"   ranking A: {[BNAME[i] for i in np.argsort(-A.mean(0))]}")
say(f"   ranking C: {[BNAME[i] for i in np.argsort(-C.mean(0))]}")
if res["floor2"]:
    fl1 = np.array(res["floor1"]).mean(0); fl2 = np.array(res["floor2"]).mean(0)
    say("   coherence above the shuffled surrogate (A -> C): " +
        ", ".join(f"{BNAME[i]} {A.mean(0)[i]-fl1[i]:+.3f} -> {C.mean(0)[i]-fl2[i]:+.3f}" for i in range(4)))

with open(os.path.join(OUT, "validation_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
x = (np.arange(10) + 0.5)*10
ax[0].plot(x, np.mean(folds["v1"], 0), "o-", c="C3", label="v1 (published)")
ax[0].plot(x, np.mean(folds["v2"], 0), "o-", c="C0", label="v2 (fixed)")
ax[0].axhline(1, c="k", lw=0.8, ls=":")
ax[0].set(xlabel="Time within 100-ms integration window (ms)", ylabel="Population firing / mean",
          title="(a) Window-locked firing (left limb)"); ax[0].legend(frameon=False)
fq = f
for key, col, lab in (("A", "C3", "A: v1 gen + v1 coherence"), ("B", "C1", "B: v1 gen + v2 coherence"),
                      ("C", "C0", "C: v2 gen + v2 coherence"), ("floor2", "0.5", "trial-shuffled surrogate (v2)")):
    if specs.get(key):
        ax[1].plot(fq, np.nanmean(specs[key], 0), c=col, lw=2 if key != "floor2" else 1.2,
                   ls="-" if key != "floor2" else "--", label=lab)
for f0, f1 in BANDS: ax[1].axvline(f0, c="k", lw=0.4, ls=":")
ax[1].set(xlim=(5, 100), ylim=(0, 1), xlabel="Frequency (Hz)", ylabel="Coherence (time-avg, inside COI)",
          title="(b) Left-right coherence, published config"); ax[1].legend(frameon=False, fontsize=8)
w = 0.35; xi = np.arange(2)
ax[2].bar(xi - w/2, [known[k][0] for k in known], w, color="C3", label="v1")
ax[2].bar(xi + w/2, [known[k][1] for k in known], w, color="C0", label="v2")
ax[2].set_xticks(xi); ax[2].set_xticklabels(list(known)); ax[2].set_ylim(0, 1)
ax[2].set(ylabel="Coherence 20-30 Hz", title="(c) Known shared 20-30 Hz component")
ax[2].legend(frameon=False)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_validation.png"), dpi=200)
say(f"\nSaved: {os.path.join(OUT, 'validation_report.txt')}\n       {os.path.join(OUT, 'fig_validation.png')}")
