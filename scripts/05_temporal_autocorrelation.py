"""Temporal autocorrelation of pain ratings and spectral features
(Supplementary Fig. 5a).

All surveys are used. Spectral power is log-transformed and every series is
z-scored. A series is considered decorrelated at the first across-day lag at
which the 95% bootstrap interval of its autocorrelation includes zero.

Outputs in results/temporal/:
  autocorrelation.csv   per participant and lag bin
  decorrelation.csv     per participant
"""

import numpy as np
import pandas as pd

from pain_decoding import temporal
from pain_decoding.config import base_parser, load_config, selected_participants
from pain_decoding.data import load_recording


def analyze(cfg, participant):
    params = cfg.temporal
    rec = load_recording(cfg, participant)
    pain = temporal.zscore(rec.vas[:, None])
    neural = temporal.zscore(np.log10(rec.during))
    rng = np.random.RandomState(cfg.seed)

    # Share of pain variance between days, against shuffled day labels.
    day_share = temporal.between_day_variance_share(pain, rec.day)[0]
    day_share_null = np.array(
        [
            temporal.between_day_variance_share(pain, rng.permutation(rec.day))[0]
            for _ in range(params["n_permutations"])
        ]
    )

    n_pairs, pain_acf = temporal.lagged_autocorrelation(
        pain, rec.day, rec.hours, params["min_pairs"]
    )
    _, neural_acf = temporal.lagged_autocorrelation(
        neural, rec.day, rec.hours, params["min_pairs"]
    )
    pain_low, pain_high = temporal.bootstrap_autocorrelation_ci(
        pain, rec.day, rec.hours, params["n_bootstrap"], rng, params["min_pairs"]
    )
    neural_low, _ = temporal.bootstrap_autocorrelation_ci(
        neural, rec.day, rec.hours, params["n_bootstrap"], rng, params["min_pairs"]
    )

    lags = pd.DataFrame(
        {
            "participant": participant,
            "lag": temporal.LAG_LABELS,
            "n_pairs": n_pairs,
            "pain_acf": pain_acf[:, 0],
            "pain_ci_low": pain_low[:, 0],
            "pain_ci_high": pain_high[:, 0],
            "neural_acf_median": np.nanmedian(neural_acf, axis=1),
            "neural_acf_q25": np.nanpercentile(neural_acf, 25, axis=1),
            "neural_acf_q75": np.nanpercentile(neural_acf, 75, axis=1),
            "neural_pct_ci_above_zero": 100 * np.mean(neural_low > 0, axis=1),
        }
    )

    neural_lag = np.array(
        [temporal.decorrelation_lag(neural_low[:, f]) for f in range(neural.shape[1])]
    )
    decorrelation = {
        "participant": participant,
        "pain_decorrelated": temporal.DECORRELATION_LABELS[
            temporal.decorrelation_lag(pain_low[:, 0])
        ],
        "neural_pct_within_1_day": 100 * np.mean(neural_lag <= 0),
        "neural_pct_by_2_days": 100 * np.mean(neural_lag <= 1),
        "neural_pct_by_3_days": 100 * np.mean(neural_lag <= 2),
        "pain_between_day_variance_share": day_share,
        "pain_between_day_variance_share_null95": np.percentile(day_share_null, 95),
    }
    return lags, decorrelation


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    out = cfg.out_dir("temporal")

    lag_tables, rows = [], []
    with np.errstate(invalid="ignore", divide="ignore"):
        for participant in selected_participants(cfg, args.participants):
            lags, decorrelation = analyze(cfg, participant)
            lag_tables.append(lags)
            rows.append(decorrelation)
            print(
                f"{participant}: pain ratings decorrelated {decorrelation['pain_decorrelated']}",
                flush=True,
            )

    pd.concat(lag_tables).to_csv(out / "autocorrelation.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / "decorrelation.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
