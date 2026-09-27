"""Matched cross-trial null for every condition of the factorial study.

For each condition, the left EMG of trial i is paired with the right EMG of
trial j != i (same condition). This keeps each signal's spectrum, amplitude
modulation and task timing but removes within-trial coupling. The statistic is
the median (across trials) of the band-median coherence; its null distribution
comes from random derangements of the trial labels. P-values are one-sided and
Benjamini-Hochberg FDR-corrected across all conditions x bands.

Usage (after run_factorial.py):
    python paper_analysis/cross_trial_null_factorial.py results/factorial
    python paper_analysis/cross_trial_null_factorial.py results/factorial --permutations 199   # faster

Output: <folder>/cross_trial_null_summary.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions as F


def bh_fdr(p):
    p = np.asarray(p, float); n = p.size
    order = np.argsort(p)
    q = p[order]*n/np.arange(1, n+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="output folder of run_factorial.py")
    ap.add_argument("--permutations", type=int, default=999)
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args()
    files = sorted((a.folder/"trial_signals").glob("*.npz"))
    if not files:
        raise SystemExit(f"No trial_signals/*.npz in {a.folder}. Run run_factorial.py without --no-save-signals.")
    rng = np.random.default_rng(a.seed)
    rows = []
    for k, path in enumerate(files, 1):
        z = np.load(path)
        res = F.cross_trial_null(list(z["left"].astype(float)), list(z["right"].astype(float)),
                                 float(z["fs_hz"]), n_perm=a.permutations, rng=rng, progress=False)
        for b, band in enumerate(res["bands"]):
            rows.append(dict(cortical_high_hz=int(z["cortical_high_hz"]), mixture=str(z["mixture"]),
                             crosstalk_R2L=float(z["crosstalk_R2L"]), band=band, n_trials=res["n_trials"],
                             observed_median=res["observed"][b], null_median=res["null_median"][b],
                             null_95pct=res["null_95"][b], excess_over_null=res["excess"][b],
                             p_one_sided=res["p"][b]))
        print(f"[{k}/{len(files)}] {path.stem}: excess over null " +
              ", ".join(f"{bd} {e:+.3f}" for bd, e in zip(res["bands"], res["excess"])))
    out = pd.DataFrame(rows)
    out["q_fdr"] = bh_fdr(out["p_one_sided"].to_numpy())
    out.to_csv(a.folder/"cross_trial_null_summary.csv", index=False)
    print("Saved", (a.folder/"cross_trial_null_summary.csv").resolve())


if __name__ == "__main__":
    main()
