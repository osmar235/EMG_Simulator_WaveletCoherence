"""Same-condition cross-trial null inference for EMG Simulator V3.3.

Why this null:
Phase randomization preserves the periodogram but destroys the strongly
nonstationary, task-locked burst structure of the simulated EMG. V3.3 instead
pairs left EMG from one trial with right EMG from a *different* trial in the
same factorial condition. This preserves marginal spectra, burst timing/task
envelopes, duration and preprocessing while breaking within-trial bilateral
common input/crosstalk.

Workflow:
1. run_factorial_v3.py (default saves null_signals/*.npz);
2. this script computes each trial CWT once per condition;
3. all off-diagonal L_i/R_j band coherences are cached;
4. random derangements of trial labels produce a null distribution for the
   condition-level median coherence statistic;
5. empirical one-sided p-values and BH-FDR q-values are exported.

The main factorial inference remains the mixed ART model. This null analysis is
particularly useful for interpreting absolute/band-specific coherence floors.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import functions as F

BANDS = ((5, 13), (13, 30), (30, 60), (60, 100))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('factorial_dir', type=Path,
                    help='Directory created by run_factorial_v3.py')
    ap.add_argument('--permutations', type=int, default=999,
                    help='Random derangements per condition after the off-diagonal pair matrix is computed.')
    ap.add_argument('--seed', type=int, default=20260926)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--taper', choices=['none','matviyenko','tukey'], default='none',
                    help='Must match the factorial analysis. V3.3 primary = none.')
    ap.add_argument('--matviyenko-k', type=int, default=8)
    ap.add_argument('--max-cells', type=int, default=None,
                    help='Optional smoke-test limit on number of condition files.')
    return ap.parse_args()


def _bh_fdr(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * vals.size / np.arange(1, vals.size+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q); tmp[order] = q
    out[ok] = tmp
    return out


def _load_factorial_csv(folder):
    candidates = [folder/'factorial_v3_3_band_medians.csv', folder/'factorial_v3_band_medians.csv']
    for p in candidates:
        if p.exists():
            return p, pd.read_csv(p)
    raise FileNotFoundError('No factorial band-median CSV found in the supplied directory.')


def _condition_from_npz(z):
    def scalar(name):
        a = z[name]
        return a.reshape(-1)[0].item() if a.dtype.kind != 'U' else str(a.reshape(-1)[0])
    return dict(
        cortical_high_hz=int(scalar('cortical_high_hz')),
        mixture=str(scalar('mixture')),
        crosstalk_R2L=float(scalar('crosstalk_R2L')),
        crosstalk_L2R=float(scalar('crosstalk_L2R')),
    )


def main():
    args = parse_args()
    folder = args.factorial_dir
    null_dir = folder/'null_signals'
    if not null_dir.exists():
        raise FileNotFoundError(f'Missing {null_dir}. Re-run factorial V3.3 without --no-save-null-signals.')
    csv_path, obs = _load_factorial_csv(folder)
    if 'analysis_taper' in obs.columns:
        seen = sorted(set(obs['analysis_taper'].astype(str)))
        if seen != [args.taper]:
            raise ValueError(f'--taper {args.taper!r} does not match factorial CSV analysis_taper={seen}')
    if args.taper == 'matviyenko' and 'matviyenko_K' in obs.columns:
        kvals = sorted(set(int(v) for v in obs['matviyenko_K'].dropna().unique()))
        if kvals and kvals != [args.matviyenko_k]:
            raise ValueError(f'--matviyenko-k {args.matviyenko_k} does not match factorial CSV K={kvals}')
    taper = None if args.taper == 'none' else args.taper
    taper_kwargs = {'K': args.matviyenko_k} if taper == 'matviyenko' else None
    rng = np.random.default_rng(args.seed)

    files = sorted(null_dir.glob('*.npz'))
    if args.max_cells is not None:
        files = files[:args.max_cells]
    if not files:
        raise FileNotFoundError(f'No .npz cell files found in {null_dir}')

    summary_rows = []
    pair_rows = []
    print(f'Using factorial data: {csv_path}')
    print(f'Cross-trial null: {len(files)} conditions, {args.permutations} derangements/condition, taper={args.taper}')

    for cell_index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as z:
            left = np.asarray(z['left'], float)
            right = np.asarray(z['right'], float)
            fs = float(np.asarray(z['fs_hz']).reshape(-1)[0])
            cond = _condition_from_npz(z)
        n = left.shape[0]
        if right.shape != left.shape or n < 2:
            raise ValueError(f'Invalid trial arrays in {path.name}')
        print(f'[{cell_index}/{len(files)}] bw={cond["cortical_high_hz"]} mix={cond["mixture"]} ct={cond["crosstalk_R2L"]:.2f} n={n}')

        # Apply the same analysis-only taper as the observed analysis, if any.
        Lsig = [F.apply_analysis_taper(x, taper=taper, taper_kwargs=taper_kwargs) for x in left]
        Rsig = [F.apply_analysis_taper(y, taper=taper, taper_kwargs=taper_kwargs) for y in right]

        # Cache all CWTs once. Pairing then only repeats the coherence smoothing.
        Lc = [F.compute_cwt_for_coherence(x, fs, fmax=128.0, dj=1/8, w0=6) for x in tqdm(Lsig, desc='  left CWT', leave=False)]
        Rc = [F.compute_cwt_for_coherence(y, fs, fmax=128.0, dj=1/8, w0=6) for y in tqdm(Rsig, desc='  right CWT', leave=False)]

        pair = np.full((n, n, len(BANDS)), np.nan, float)
        for i in tqdm(range(n), desc='  off-diagonal pairs', leave=False):
            for j in range(n):
                if i == j:
                    continue
                freqs, period, coi, Rsq = F.coherence_from_cached_cwts(Lc[i], Rc[j])
                meds = F.band_medians(Rsq, freqs, bands=BANDS, period=period, coi=coi)
                vals = np.array(list(meds.values()), float)
                pair[i, j, :] = vals
                for bi, ((f0,f1), v) in enumerate(zip(BANDS, vals)):
                    pair_rows.append({**cond, 'left_trial': i, 'right_trial': j,
                                      'band': f'{f0}-{f1}', 'coherence_median': v})

        # Observed condition-level statistic from within-trial results.
        mask = (
            (obs['cortical_high_hz'].astype(int) == cond['cortical_high_hz']) &
            (obs['mixture'].astype(str) == cond['mixture']) &
            np.isclose(obs['crosstalk_R2L'].astype(float), cond['crosstalk_R2L'])
        )
        od = obs.loc[mask]
        if od.empty:
            raise RuntimeError(f'No observed rows match {cond}')
        observed = np.array([
            od.loc[od['band'].astype(str)==f'{f0}-{f1}', 'coherence_median'].median()
            for f0,f1 in BANDS
        ], float)

        # Each permutation is a derangement: every L_i is paired to R_j, j != i.
        null_stats = np.empty((args.permutations, len(BANDS)), float)
        base = np.arange(n)
        for b in range(args.permutations):
            perm = F.random_derangement(n, rng)
            null_stats[b] = np.nanmedian(pair[base, perm, :], axis=0)

        q95 = np.nanpercentile(null_stats, 100*(1-args.alpha), axis=0)
        null_med = np.nanmedian(null_stats, axis=0)
        p_emp = (1 + np.sum(null_stats >= observed[None, :], axis=0)) / (args.permutations + 1)
        for bi, (f0,f1) in enumerate(BANDS):
            summary_rows.append({
                **cond, 'band': f'{f0}-{f1}',
                'n_trials': n, 'n_offdiag_pairs': n*(n-1), 'n_derangements': args.permutations,
                'analysis_taper': args.taper,
                'matviyenko_K': args.matviyenko_k if args.taper == 'matviyenko' else np.nan,
                'observed_condition_median': observed[bi],
                'null_derangement_median': null_med[bi],
                'null_95pct': q95[bi],
                'observed_minus_null_median': observed[bi]-null_med[bi],
                'empirical_p_one_sided': p_emp[bi],
            })

    summary = pd.DataFrame(summary_rows)
    summary['bh_fdr_q_across_reported_tests'] = _bh_fdr(summary['empirical_p_one_sided'].to_numpy())
    pairs = pd.DataFrame(pair_rows)
    out_summary = folder/'trial_shuffle_null_summary.csv'
    out_pairs = folder/'trial_shuffle_offdiagonal_pairs.csv'
    summary.to_csv(out_summary, index=False)
    pairs.to_csv(out_pairs, index=False)
    meta = {
        'code_version': F.__version__, 'method': 'same-condition cross-trial derangement',
        'permutations': args.permutations, 'seed': args.seed, 'alpha': args.alpha,
        'analysis_taper': args.taper,
        'matviyenko_K': args.matviyenko_k if args.taper == 'matviyenko' else None,
        'interpretation': 'Preserves task/burst structure and marginal trial properties while breaking within-trial bilateral pairing. Main factorial inference remains mixed ART.',
    }
    (folder/'trial_shuffle_null_config.json').write_text(json.dumps(meta, indent=2))
    print('Saved:', out_summary.resolve())
    print('Saved:', out_pairs.resolve())


if __name__ == '__main__':
    main()
