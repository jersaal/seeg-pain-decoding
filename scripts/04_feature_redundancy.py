"""Correlation structure of the region-level features (Supplementary Fig. 6).

Outputs in results/multivariate/:
  feature_redundancy.csv        median |r| and effective dimensionality
  feature_correlations.npz      all pairwise |r| per participant
"""

import numpy as np
import pandas as pd

from pain_decoding.config import base_parser, load_config, selected_participants
from pain_decoding.data import binarize_pain, load_recording
from pain_decoding.multivariate import feature_redundancy


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    out = cfg.out_dir("multivariate")

    rows, correlations = [], {}
    for participant in selected_participants(cfg, args.participants):
        rec = load_recording(cfg, participant)
        surveys, _, _, _ = binarize_pain(rec.vas, **cfg.pain_labels)
        stats = feature_redundancy(
            rec.during[surveys],
            rec.features["group"],
            cfg.multivariate["variance_explained"],
            cfg.seed,
        )
        correlations[participant] = stats["abs_r"]
        rows.append(
            {
                "participant": participant,
                "n_features": stats["n_features"],
                "median_abs_r": stats["median_abs_r"],
                "effective_dim": stats["effective_dim"],
                "effective_dim_pct": 100 * stats["effective_dim"] / stats["n_features"],
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(out / "feature_redundancy.csv", index=False)
    np.savez_compressed(out / "feature_correlations.npz", **correlations)
    print(table.round(3).to_string(index=False))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
