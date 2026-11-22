"""Regression on continuous pain ratings (Supplementary Table 3).

Same features and Leave-2-Days-Out scheme as the classifier, with a random
forest regressor and all surveys. Performance is the mean Spearman
correlation between predicted and reported ratings within held-out folds.

Outputs in results/regression/:
  summary.csv
  <participant>.npz    fold correlations and permutation null
Participants with an existing output file are skipped unless --overwrite is given.
"""

import time

import numpy as np
import pandas as pd

from pain_decoding import regression
from pain_decoding.config import base_parser, load_config, selected_participants
from pain_decoding.data import load_recording
from pain_decoding.stats import bootstrap_mean_ci, fdr_bh, permutation_pvalue


def run(cfg, participant, n_permutations):
    params = cfg.multivariate
    rec = load_recording(cfg, participant)
    folds = regression.regression_folds(
        rec.during,
        rec.vas,
        rec.day,
        rec.features["group"],
        min_test=cfg.regression["min_test_surveys"],
        min_train=params["min_train_surveys"],
        seed=cfg.seed,
    )
    correlations, predictions = regression.fold_correlations(
        folds, rec.vas, params["n_trees"], cfg.seed
    )
    null = regression.permutation_null(
        folds, rec.vas, n_permutations, params["n_trees"], cfg.seed, cfg.n_workers
    )
    return {
        "fold_correlation": correlations,
        "null_correlation": null,
        "r2": regression.pooled_r2(folds, predictions, rec.vas),
        "n_surveys": rec.n_surveys,
    }


def summarize(cfg, out):
    rows = []
    for participant in cfg.labels:
        path = out / f"{participant}.npz"
        if not path.exists():
            continue
        with np.load(path) as saved:
            correlations, null = saved["fold_correlation"], saved["null_correlation"]
            r2, n_surveys = float(saved["r2"]), int(saved["n_surveys"])
        low, high = bootstrap_mean_ci(
            correlations, cfg.multivariate["n_bootstrap"], np.random.RandomState(cfg.seed)
        )
        rows.append(
            {
                "participant": participant,
                "n_surveys": n_surveys,
                "n_folds": len(correlations),
                "spearman": correlations.mean(),
                "ci_low": low,
                "ci_high": high,
                "n_permutations": len(null),
                "p": permutation_pvalue(correlations.mean(), null) if len(null) else np.nan,
                "r2": r2,
            }
        )
    summary = pd.DataFrame(rows)
    if len(summary) and summary["p"].notna().all():
        summary["p_fdr"] = fdr_bh(summary["p"], cfg.alpha)[1]
    return summary


def main():
    parser = base_parser(__doc__)
    parser.add_argument(
        "--n-permutations", type=int, default=None, help="Override the configured number."
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = cfg.out_dir("regression")
    n_permutations = (
        cfg.multivariate["n_permutations"] if args.n_permutations is None else args.n_permutations
    )

    for participant in selected_participants(cfg, args.participants):
        path = out / f"{participant}.npz"
        if path.exists() and not args.overwrite:
            print(f"{participant}: output exists, skipping", flush=True)
            continue
        start = time.time()
        result = run(cfg, participant, n_permutations)
        np.savez_compressed(path, **result)
        print(
            f"{participant}: mean fold rho = {result['fold_correlation'].mean():.3f} "
            f"[{time.time() - start:.0f} s]",
            flush=True,
        )

    summary = summarize(cfg, out)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
