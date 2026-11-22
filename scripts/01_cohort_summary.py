"""Participant characteristics and pain survey statistics (Supplementary Table 1).

Output: results/cohort/participant_summary.csv
"""

import numpy as np
import pandas as pd

from pain_decoding.config import base_parser, load_config, selected_participants
from pain_decoding.data import binarize_pain, load_recording


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)

    rows = []
    for participant in selected_participants(cfg, args.participants):
        rec = load_recording(cfg, participant)
        _, y, low, high = binarize_pain(rec.vas, **cfg.pain_labels)
        q1, median, q3 = np.percentile(rec.vas, [25, 50, 75])
        rows.append(
            {
                "participant": participant,
                "n_surveys": rec.n_surveys,
                "n_days": len(np.unique(rec.day)),
                "vas_median": median,
                "vas_q1": q1,
                "vas_q3": q3,
                "low_threshold": low,
                "high_threshold": high,
                "n_high_pain": int(np.sum(y == 1)),
                "n_low_pain": int(np.sum(y == 0)),
                "median_inter_survey_hours": float(np.median(np.diff(rec.hours))),
                "n_gray_matter_contacts": len(rec.contacts),
                "n_univariate_models": len(rec.features),
                "n_region_band_features": rec.features["group"].nunique(),
            }
        )

    table = pd.DataFrame(rows)
    path = cfg.out_dir("cohort") / "participant_summary.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
