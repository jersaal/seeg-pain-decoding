"""Multivariate decoding of high versus low pain, with permutation test.

Analyses (--analysis):
  main            train and test on during-survey power (Fig. 2b-e)
  detrended       linear trend over surveys removed from every feature (Fig. 2f)
  cross_context   train on during-survey power, test on pre-survey power (Fig. 2g)
  one_day_buffer  training days adjacent to the held-out days removed
                  (Supplementary Fig. 5)

Output: results/multivariate/<analysis>/<participant>.npz
Participants with an existing output file are skipped unless --overwrite is given.
"""

import time

from pain_decoding import io
from pain_decoding.config import base_parser, load_config, selected_participants
from pain_decoding.data import binarize_pain, load_recording
from pain_decoding.multivariate import classification_folds, decode, permutation_null
from pain_decoding.stats import permutation_pvalue


def run(cfg, analysis, participant, n_permutations):
    params = cfg.multivariate
    rec = load_recording(cfg, participant)
    if analysis == "detrended":
        rec = rec.detrended()
    surveys, y, _, _ = binarize_pain(rec.vas, **cfg.pain_labels)

    X_fit = rec.during[surveys]
    X_eval = rec.pre[surveys] if analysis == "cross_context" else X_fit
    folds = classification_folds(
        X_fit,
        X_eval,
        y,
        rec.day[surveys],
        rec.features["group"],
        buffer_days=params["buffer_days"] if analysis == "one_day_buffer" else 0,
        min_train=params["min_train_surveys"],
        seed=cfg.seed,
    )
    result = decode(
        folds,
        y,
        params["n_trees"],
        cfg.seed,
        n_importance_repeats=params["n_importance_repeats"] if analysis == "main" else 0,
    )
    null = None
    if n_permutations > 0:
        null = permutation_null(
            folds, y, n_permutations, params["n_trees"], cfg.seed, cfg.n_workers
        )
    feature_names = sorted(set(rec.features["group"]))
    io.save_decoding(
        io.multivariate_path(cfg, analysis, participant), result, y, surveys, feature_names, null
    )
    return result, null


def main():
    parser = base_parser(__doc__)
    parser.add_argument(
        "--analysis",
        nargs="+",
        default=list(io.MULTIVARIATE_ANALYSES),
        choices=io.MULTIVARIATE_ANALYSES,
    )
    parser.add_argument(
        "--n-permutations", type=int, default=None, help="Override the configured number."
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n_permutations = (
        cfg.multivariate["n_permutations"] if args.n_permutations is None else args.n_permutations
    )

    for analysis in args.analysis:
        for participant in selected_participants(cfg, args.participants):
            if io.multivariate_path(cfg, analysis, participant).exists() and not args.overwrite:
                print(f"{analysis} {participant}: output exists, skipping", flush=True)
                continue
            start = time.time()
            result, null = run(cfg, analysis, participant, n_permutations)
            p = permutation_pvalue(result.mean_auc, null) if null is not None else float("nan")
            print(
                f"{analysis} {participant}: mean AUC = {result.mean_auc:.3f} "
                f"({len(result.fold_auc)} folds), p = {p:.4f} "
                f"[{time.time() - start:.0f} s]",
                flush=True,
            )


if __name__ == "__main__":
    main()
