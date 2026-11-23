# Decoding chronic pain states from distributed intracranial recordings

Analysis code for

> Saal J, Khambhati AN, Chang EF, Shirvalkar P. *Decoding chronic pain states
> from distributed intracranial recordings.*

The code decodes high versus low chronic pain states from stereo-EEG spectral
power in six participants and characterizes where pain-predictive signals are
found across contacts, frequency bands, anatomical regions and canonical
(Yeo 7) networks. It reproduces every statistic reported in the Results,
Methods, figure legends and supplementary tables of the manuscript, starting
from the per-survey spectral power features.

## Contents

| Step | Script | Analysis | Manuscript | Run time* |
|---|---|---|---|---|
| 1 | `01_cohort_summary.py` | Survey counts, pain ratings, high/low thresholds | Supp. Table 1 | seconds |
| 2 | `02_multivariate_decoding.py` | Region-level PCA + random forest, Leave-2-Days-Out cross-validation, permutation test; detrended, cross-context and one-day-buffer controls | Fig. 2, Supp. Fig. 5c, 7 | about 7 h |
| 3 | `03_multivariate_summary.py` | AUC confidence intervals, FDR correction, Youden operating points | Fig. 2, Supp. Table 2, Supp. Fig. 4 | seconds |
| 4 | `04_feature_redundancy.py` | Feature correlations, effective dimensionality | Supp. Fig. 6 | seconds |
| 5 | `05_temporal_autocorrelation.py` | Autocorrelation of pain ratings and spectral features | Supp. Fig. 5a | under a minute |
| 6 | `06_continuous_regression.py` | Random forest regression on continuous ratings | Supp. Table 3 | 1.7 h |
| 7 | `07_univariate_decoding.py` | One logistic regression per contact and band, permutation nulls | Fig. 3a,b | 2 h |
| 8 | `08_univariate_statistics.py` | Contact-level significance, band tests, shared regions, directionality | Fig. 3c–f, Fig. 4, Supp. Fig. 8 | seconds |
| 9 | `09_network_enrichment.py` | Network assignment, enrichment test, atlas and threshold sensitivity | Fig. 5, Supp. Table 4 | seconds |
| 10 | `10_coordinate_jitter.py` | Enrichment after displacing contact coordinates | Results | 3 min |
| 11 | `11_verify_manuscript_values.py` | Compares all reproduced values with the manuscript | — | seconds |

\* On a 10-core Apple M-series laptop with 32 GB of memory. Permutation tests
dominate the run time (about 11 h in total) and scale with the number of cores.

Preprocessing of the raw recordings (re-referencing, filtering, artifact
rejection, multitaper spectral estimation; see Methods) and figure rendering
are not part of this repository. The artifact classifier is available at
<https://github.com/jersaal/ieeg-noise-detection>.

## System requirements

- Python 3.8.20 with the package versions pinned in `requirements.txt`
  (NumPy 1.24.3, pandas 2.0.3, SciPy 1.10.1, scikit-learn 1.3.2,
  statsmodels 0.14.1, nilearn 0.10.4, PyYAML 6.0.1).
- Tested on macOS (Apple Silicon). The code is pure Python and is expected to
  run unchanged on Linux and Windows.
- No GPU or other special hardware. A multi-core CPU is recommended.
- Internet access on the first run of steps 9–10, when nilearn downloads the
  Schaefer atlas (about 10 MB) to `~/nilearn_data`.

## Installation

```bash
git clone https://github.com/jersaal/seeg-pain-decoding.git
cd seeg-pain-decoding
conda env create -f environment.yml     # creates the environment "seeg-pain-decoding"
conda activate seeg-pain-decoding
```

or, in an existing Python 3.8 environment,

```bash
pip install -r requirements.txt
pip install -e .
```

Installation takes a few minutes.

## Data

The de-identified processed data are included in `data/`, as one folder per
participant (`P1` to `P6`, about 8 MB in total), so the repository is
self-contained. `data/README.md` describes the files. To keep the data
elsewhere, set `data_dir` in `config.yaml`.

## Usage

Run everything, in order:

```bash
python scripts/run_all.py
```

or run single steps. Every script accepts `--config` and `--participants`,
and `--help` describes its outputs:

```bash
python scripts/02_multivariate_decoding.py --analysis main --participants P1
python scripts/07_univariate_decoding.py --participants P4
```

The long steps (2, 6 and 7) write one file per participant and skip
participants that are already done, so an interrupted run can be restarted
with the same command (`--overwrite` forces recomputation).

**Quick demonstration.** With `--n-permutations 0`, steps 2 and 7 compute the
cross-validated decoding performance without the permutation tests, in about
a minute for all participants:

```bash
python scripts/07_univariate_decoding.py --n-permutations 0
python scripts/02_multivariate_decoding.py --analysis cross_context --n-permutations 0
```

Expected output of the second command:

```
cross_context P1: mean AUC = 0.702 (41 folds), p = nan [10 s]
cross_context P2: mean AUC = 0.422 (27 folds), p = nan [7 s]
...
```

All analysis parameters (cross-validation, numbers of trees, permutations and
bootstrap resamples, atlas resolution, distance thresholds, random seed) are
set in `config.yaml`.

## Outputs

Results are written to `results/`:

```
results/
├── cohort/participant_summary.csv           Supp. Table 1
├── multivariate/
│   ├── <analysis>/<participant>.npz         fold AUCs, held-out predictions, permutation null
│   ├── summary.csv                          Supp. Table 2; Fig. 2c,f,g
│   ├── operating_points.csv                 Fig. 2e
│   ├── survey_predictions.csv               Fig. 2d, Supp. Fig. 4
│   ├── feature_importance.csv               Supp. Fig. 7
│   └── feature_redundancy.csv               Supp. Fig. 6
├── temporal/                                Supp. Fig. 5a
├── regression/summary.csv                   Supp. Table 3
├── univariate/
│   ├── models/                              per-model AUCs and permutation nulls
│   ├── features.csv, contacts.csv           Fig. 3c,d; Fig. 4
│   ├── participant_summary.csv              Fig. 3b,c
│   ├── band_proportions.csv, band_tests.csv Fig. 3e
│   ├── region_consistency.csv               Fig. 4
│   └── top_features.csv                     Fig. 3f, Supp. Fig. 8
├── network/
│   ├── contact_assignments.csv              Fig. 5a
│   ├── enrichment.csv, network_vs_rest.csv  Fig. 5b,c
│   ├── sensitivity.csv                      Supp. Table 4
│   └── coordinate_jitter_summary.csv        Results
└── manuscript_values.csv                    reproduced versus printed values
```

## Reproducing the manuscript

`python scripts/run_all.py` ends with `11_verify_manuscript_values.py`, which
recomputes each statistic quoted in the manuscript from the files above,
formats it as printed and compares the two:

```
Multivariate decoding (Results; Fig. 2; Supplementary Table 2)
  [ok] median AUC: 0.72
  [ok] AUC range: 0.41-0.91
  ...
69 of 69 values match the manuscript.
```

Notes on exact reproduction:

- All random number generators are seeded from `random_seed` in
  `config.yaml`. Permutation tests give identical results for any number of
  worker processes.
- Use the pinned package versions. Random forest AUCs change slightly under
  newer scikit-learn releases (by up to 0.03 with 1.8.0); the univariate,
  anatomical and network results do not depend on the scikit-learn version
  in our tests.
- Participants are labeled P1–P6 as in the manuscript, in the data folders
  and in all outputs.

## Tests

```bash
pip install pytest
pytest
```

The tests run the main building blocks on synthetic data and take about a
minute.

## Repository structure

```
config.yaml               analysis parameters
requirements.txt          pinned dependencies
environment.yml           conda environment
src/pain_decoding/
    config.py             configuration and command-line conventions
    data.py               loading of features and ratings, high/low pain labels
    cv.py                 Leave-2-Days-Out splits, optional day buffer
    multivariate.py       region-level features, random forest decoding, permutation test
    regression.py         regression on continuous ratings
    univariate.py         single-feature decoding, permutation nulls, significance
    temporal.py           lag-binned autocorrelation
    network.py            atlas, network assignment, enrichment tests
    stats.py              permutation p-values, FDR, bootstrap, parallel map
    io.py                 intermediate result files
scripts/                  one script per analysis step (see Contents)
tests/                    unit tests on synthetic data
data/README.md            description of the data files
```

## License

MIT; see `LICENSE`.
