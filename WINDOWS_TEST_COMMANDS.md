# Windows CMD commands - EMG Simulator V3.3

Assumed extracted folder:

`C:\Users\osmar\projects\Version3.3_EMGSim`

Virtual environment:

`C:\Users\osmar\projects\.venv`

## 1. Enter folder and check environment

```bat
cd /d C:\Users\osmar\projects\Version3.3_EMGSim
C:\Users\osmar\projects\.venv\Scripts\python.exe --version
C:\Users\osmar\projects\.venv\Scripts\python.exe -c "import numpy, scipy, pandas, matplotlib, pycwt; print('Python dependencies OK')"
```

## 2. Validate V3.3

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe validate_v3.py
```

Expected core result: `RESULT: 30/30 checks passed`.

## 3. Single primary demo - no taper

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_single_demo.py --taper none --surrogates 0 --outdir results\demo_v3_3_primary
```

## 4. Dry run

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --dry-run
```

Expected: 720 bilateral trials, 1,440 EMG signals, 2,880 band rows.

## 5. One-trial-per-cell smoke run

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 1 --taper none --outdir results\factorial_v3_3_smoke
```

## 6. Full corrected V3.3 run

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 15 --base-seed 20260926 --taper none --outdir results\factorial_v3_3_primary
```

## 7. Same-condition cross-trial null - smoke test

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe trial_shuffle_null_v3.py results\factorial_v3_3_primary --permutations 99 --max-cells 1 --taper none
```

## 8. Same-condition cross-trial null - full

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe trial_shuffle_null_v3.py results\factorial_v3_3_primary --permutations 999 --seed 20260926 --taper none
```

## 9. Primary mixed ART

```bat
Rscript analysis_art_v3.R results\factorial_v3_3_primary\factorial_v3_3_band_medians.csv
```

## 10. Optional Matviyenko sensitivity

```bat
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 3 --base-seed 20260926 --taper matviyenko --matviyenko-k 2 --outdir results\factorial_sensitivity_K2
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 3 --base-seed 20260926 --taper matviyenko --matviyenko-k 8 --outdir results\factorial_sensitivity_K8
C:\Users\osmar\projects\.venv\Scripts\python.exe run_factorial_v3.py --trials 3 --base-seed 20260926 --taper matviyenko --matviyenko-k 13 --outdir results\factorial_sensitivity_K13
```

V3.3 primary analysis remains **no taper**. Matviyenko is sensitivity only.
