"""Univariate decoding: one logistic regression per contact and frequency
band, with a label-permutation null for every model (Fig. 3).

Outputs in results/univariate/models/:
  <participant>_models.csv      mean AUC and direction per model
  <participant>_fold_auc.npz    AUC per model and fold
  <participant>_null_auc.npz    null mean AUC per permutation and model
Participants with existing output files are skipped unless --overwrite is given.
"""

import time

import numpy as np

from pain_decoding import io, univariate
from pain_decoding.config import base_parser, load_config, selected_participants
from pain_decoding.data import binarize_pain, load_recording


def main():
    parser = base_parser(__doc__)
    parser.add_argument(
        "--n-permutations", type=int, default=None, help="Override the configured number."
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n_permutations = (
        cfg.univariate["n_permutations"] if args.n_permutations is None else args.n_permutations
    )

    for participant in selected_participants(cfg, args.participants):
        models_path = io.univariate_models_path(cfg, participant)
        null_path = io.univariate_null_path(cfg, participant)
        rec = load_recording(cfg, participant)
        surveys, y, _, _ = binarize_pain(rec.vas, **cfg.pain_labels)
        X, days = rec.during[surveys], rec.day[surveys]

        if not models_path.exists() or args.overwrite:
            start = time.time()
            aucs = univariate.fold_aucs(univariate.standardized_splits(X, days), y, cfg.seed)
            models = univariate.summarize_models(aucs, X, y, rec.features)
            models.to_csv(models_path, index=False)
            np.savez_compressed(
                models_path.with_name(f"{participant}_fold_auc.npz"),
                feature=rec.features["feature"].to_numpy().astype(str),
                fold_auc=aucs,
            )
            best = models.iloc[0]
            print(
                f"{participant}: {len(models)} models, {aucs.shape[1]} folds, "
                f"median AUC = {models['auc'].median():.3f}, best = {best['feature']} "
                f"({best['auc']:.3f}) [{time.time() - start:.0f} s]",
                flush=True,
            )

        if n_permutations > 0 and (not null_path.exists() or args.overwrite):
            start = time.time()
            null = univariate.permutation_null(
                X, y, days, n_permutations, cfg.seed, cfg.n_workers
            )
            np.savez_compressed(
                null_path, feature=rec.features["feature"].to_numpy().astype(str), null_auc=null
            )
            print(
                f"{participant}: {n_permutations} permutations [{time.time() - start:.0f} s]",
                flush=True,
            )


if __name__ == "__main__":
    main()
