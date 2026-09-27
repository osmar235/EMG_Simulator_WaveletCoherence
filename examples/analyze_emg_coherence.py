"""Wavelet coherence between two EMG channels in your own recordings.

Each input file is one trial: a CSV (or TXT) with one column per channel. The
script filters the raw (non-rectified) EMG, resamples it to 1 kHz, computes the
squared Morlet wavelet coherence between the two channels, and summarises it
as the median inside the cone of influence in each frequency band.

With two or more trial files (same condition), it also tests whether coherence
exceeds a matched cross-trial null (left of trial i paired with right of trial
j != i), which preserves each signal's spectrum and amplitude modulation.

Examples (from the repository folder):
    # one trial, sampling rate given, 60-Hz mains
    python examples/analyze_emg_coherence.py my_trial.csv --fs 2000 --left FDI_L --right FDI_R --notch 60

    # several trials of one condition (wildcards are expanded), with the matched null
    python examples/analyze_emg_coherence.py "data/cond1_*.csv" --fs 2000 --notch 50

    # the file written by examples/simulate_demo.py (has a time column, so --fs is optional)
    python examples/analyze_emg_coherence.py results/demo/demo_emg_1k.csv

Channel selection: --left/--right take column names or 0-based column indices.
By default the first two non-time columns are used. If a column named
time/time_s/t exists and --fs is not given, fs is inferred from it.

Outputs (in --outdir, default results/coherence):
    band_medians_per_trial.csv   one row per trial x band
    cross_trial_null.csv         (>= 2 trials) observed, null median, null 95 %, excess, p
    coherence_map_<trial>.png    time-frequency map with cone of influence (skip with --no-maps)
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions as F

TIME_NAMES = {"time", "time_s", "t", "tempo", "seconds"}


def parse_bands(text):
    out = []
    for part in text.split(","):
        lo, hi = part.split("-")
        out.append((float(lo), float(hi)))
    return tuple(out)


def pick(df, spec, default_idx, data_cols):
    if spec is None:
        return data_cols[default_idx]
    if spec in df.columns:
        return spec
    try:
        return df.columns[int(spec)]
    except (ValueError, IndexError):
        raise SystemExit(f"Column {spec!r} not found. Available: {list(df.columns)}")


def read_trial(path, a):
    sep = None if a.sep == "auto" else a.sep
    df = pd.read_csv(path, sep=sep, engine="python")
    time_cols = [c for c in df.columns if str(c).strip().lower() in TIME_NAMES]
    data_cols = [c for c in df.columns if c not in time_cols]
    if len(data_cols) < 2 and (a.left is None or a.right is None):
        raise SystemExit(f"{path}: need two EMG columns, found {list(df.columns)}")
    lc, rc = pick(df, a.left, 0, data_cols), pick(df, a.right, 1, data_cols)
    fs = a.fs
    if fs is None:
        if not time_cols:
            raise SystemExit(f"{path}: no time column; please give --fs")
        dt = np.median(np.diff(df[time_cols[0]].to_numpy(float)))
        fs = 1.0/dt
    return df[lc].to_numpy(float), df[rc].to_numpy(float), float(fs), lc, rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="one file per trial (wildcards allowed)")
    ap.add_argument("--fs", type=float, default=None, help="sampling rate of the files (Hz)")
    ap.add_argument("--left", default=None, help="left/first channel: column name or index")
    ap.add_argument("--right", default=None, help="right/second channel: column name or index")
    ap.add_argument("--sep", default="auto", help="column separator (default: auto-detect)")
    ap.add_argument("--notch", type=float, default=None, help="power-line frequency to remove (50 or 60)")
    ap.add_argument("--fs-analysis", type=float, default=1000.0, help="analysis sampling rate (default 1000)")
    ap.add_argument("--low", type=float, default=5.0, help="band-pass low edge (default 5 Hz)")
    ap.add_argument("--high", type=float, default=499.0, help="band-pass high edge (default 499 Hz, capped below Nyquist)")
    ap.add_argument("--bands", type=parse_bands, default=F.DEFAULT_BANDS,
                    help='frequency bands, e.g. "5-13,13-30,30-60,60-100" (default)')
    ap.add_argument("--fmax", type=float, default=128.0, help="highest analysed frequency (default 128 Hz)")
    ap.add_argument("--permutations", type=int, default=999, help="derangements for the cross-trial null")
    ap.add_argument("--seed", type=int, default=1, help="random seed for the null")
    ap.add_argument("--no-maps", action="store_true", help="do not save coherence-map figures")
    ap.add_argument("--outdir", type=Path, default=Path("results/coherence"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    paths = []
    for pattern in a.files:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])
    names, X, Y = [], [], []
    for p in paths:
        l, r, fs, lc, rc = read_trial(p, a)
        # raw -> (notch) -> anti-alias resample -> band-pass, detrend, z-score
        if a.notch is not None:
            l = F.notch_filter(l, fs, a.notch); r = F.notch_filter(r, fs, a.notch)
        l = F.resample_to_fs(l, fs, a.fs_analysis); r = F.resample_to_fs(r, fs, a.fs_analysis)
        X.append(F.preprocess_emg_for_coherence(l, a.fs_analysis, low=a.low, high=a.high))
        Y.append(F.preprocess_emg_for_coherence(r, a.fs_analysis, low=a.low, high=a.high))
        names.append(Path(p).stem)
        print(f"{Path(p).name}: channels {lc!r} vs {rc!r}, fs={fs:g} Hz, {l.size/a.fs_analysis:.1f} s")

    fs = a.fs_analysis
    rows = []
    for name, x, y in zip(names, X, Y):
        freqs, period, coi, Rsq, *_ = F.compute_wavelet_coherence(x, y, fs, fmax=a.fmax)
        for band, v in F.band_medians(Rsq, freqs, a.bands, period=period, coi=coi).items():
            rows.append({"trial": name, "band_hz": band, "coherence_median": v})
        if not a.no_maps:
            F.plot_coherence_map(Rsq, freqs, period, coi, fs, a.outdir/f"coherence_map_{name}.png",
                               title=f"Wavelet coherence - {name}")
    per_trial = pd.DataFrame(rows)
    per_trial.to_csv(a.outdir/"band_medians_per_trial.csv", index=False)
    print("\nMedian coherence inside the COI (median across trials):")
    print(per_trial.groupby("band_hz", sort=False)["coherence_median"].median().round(3).to_string())

    if len(X) >= 2:
        n = min(v.size for v in X + Y)
        if any(v.size != n for v in X + Y):
            print(f"\nTrials differ in length; the null uses the first {n/fs:.2f} s of every trial.")
        res = F.cross_trial_null([v[:n] for v in X], [v[:n] for v in Y], fs, bands=a.bands,
                                 n_perm=a.permutations, rng=np.random.default_rng(a.seed), fmax=a.fmax)
        out = pd.DataFrame({"band_hz": res["bands"], "observed_median": res["observed"],
                            "null_median": res["null_median"], "null_95pct": res["null_95"],
                            "excess_over_null": res["excess"], "p_one_sided": res["p"]})
        out["n_trials"] = res["n_trials"]; out["n_permutations"] = res["n_perm"]
        out.to_csv(a.outdir/"cross_trial_null.csv", index=False)
        print(f"\nMatched cross-trial null ({res['n_trials']} trials, {res['n_perm']} derangements; "
              f"smallest possible p = {1/(res['n_perm']+1):.3g}):")
        print(out[["band_hz", "observed_median", "null_median", "excess_over_null", "p_one_sided"]]
              .round(4).to_string(index=False))
    else:
        print("\nSingle trial: no significance test (give >= 2 trials of the same condition for the null).")
    print("\nSaved outputs to", a.outdir.resolve())


if __name__ == "__main__":
    main()
