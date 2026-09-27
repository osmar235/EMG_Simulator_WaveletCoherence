"""One-click demo: simulate a bilateral EMG pair (article's representative
configuration) and compute its wavelet coherence. Outputs go to results/demo/.
Options: python run_single_demo.py --help   (see examples/simulate_demo.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "examples"))
from simulate_demo import main

if __name__ == "__main__":
    main()
