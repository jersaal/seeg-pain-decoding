"""Run the complete analysis pipeline in order.

Long steps (02, 06, 07) skip participants whose outputs already exist, so an
interrupted run can simply be restarted. Extra arguments (for example
``--config other.yaml``) are passed on to every script.

    python scripts/run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    "01_cohort_summary.py",
    "02_multivariate_decoding.py",
    "03_multivariate_summary.py",
    "04_feature_redundancy.py",
    "05_temporal_autocorrelation.py",
    "06_continuous_regression.py",
    "07_univariate_decoding.py",
    "08_univariate_statistics.py",
    "09_network_enrichment.py",
    "10_coordinate_jitter.py",
    "11_verify_manuscript_values.py",
]


def main():
    here = Path(__file__).resolve().parent
    for step in STEPS:
        print(f"\n===== {step} =====", flush=True)
        start = time.time()
        subprocess.run([sys.executable, str(here / step), *sys.argv[1:]], check=True)
        print(f"===== {step} finished in {(time.time() - start) / 60:.1f} min =====", flush=True)


if __name__ == "__main__":
    main()
