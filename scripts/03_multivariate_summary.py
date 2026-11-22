"""Summary statistics of the multivariate decoders.

For every analysis and participant: mean fold AUC, 95% bootstrap confidence
interval over folds, permutation p-value and its Benjamini-Hochberg
correction across participants (Supplementary Table 2). For the main
analysis: per-survey predictions and the Youden operating point (Fig. 2d,e)
and permutation feature importance (Supplementary Fig. 7).

Outputs in results/multivariate/:
  summary.csv, operating_points.csv, survey_predictions.csv,
  feature_importance.csv
"""

import numpy as np
import pandas as pd

from pain_decoding import io
from pain_decoding.config import base_parser, load_config
from pain_decoding.cv import has_both_classes, leave_two_days_out
from pain_decoding.data import binarize_pain, load_recording
from pain_decoding.multivariate import pooled_predictions, youden_operating_point
from pain_decoding.stats import bootstrap_mean_ci, fdr_bh, permutation_pvalue

# The bootstrap generator is shared, in this order, by the analyses of a
# group; the one-day-buffer control uses its own. This reproduces the
# published intervals digit for digit.
BOOTSTRAP_GROUPS = [("main", "detrended", "cross_context"), ("one_day_buffer",)]


def median_training_surveys(cfg, participant, buffer_days):
    """Median training-set size over all usable folds (Supplementary Fig. 5b)."""
    rec = load_recording(cfg, participant)
    surveys, y, _, _ = binarize_pain(rec.vas, **cfg.pain_labels)
    sizes = [
        len(train)
        for train, _ in leave_two_days_out(rec.day[surveys], buffer_days)
        if len(train) >= cfg.multivariate["min_train_surveys"] and has_both_classes(y[train])
    ]
    return float(np.median(sizes))


def summarize(cfg):
    rows = []
    for group in BOOTSTRAP_GROUPS:
        rng = np.random.RandomState(cfg.seed)
        for analysis in group:
            for participant in cfg.labels:
                path = io.multivariate_path(cfg, analysis, participant)
                if not path.exists():
                    continue
                saved = io.load_decoding(path)
                aucs = saved["fold_auc"]
                low, high = bootstrap_mean_ci(aucs, cfg.multivariate["n_bootstrap"], rng)
                null = saved.get("null_auc")
                rows.append(
                    {
                        "analysis": analysis,
                        "participant": participant,
                        "n_surveys": len(saved["y"]),
                        "n_folds": len(aucs),
                        "median_n_train": median_training_surveys(
                            cfg,
                            participant,
                            cfg.multivariate["buffer_days"] if analysis == "one_day_buffer" else 0,
                        ),
                        "mean_auc": aucs.mean(),
                        "ci_low": low,
                        "ci_high": high,
                        "n_permutations": 0 if null is None else len(null),
                        "p": np.nan if null is None else permutation_pvalue(aucs.mean(), null),
                    }
                )
    summary = pd.DataFrame(rows)
    summary["p_fdr"] = np.nan
    for _, index in summary.groupby("analysis").groups.items():
        if summary.loc[index, "p"].notna().all():
            summary.loc[index, "p_fdr"] = fdr_bh(summary.loc[index, "p"], cfg.alpha)[1]
    return summary


def main_analysis_details(cfg):
    """Operating points, per-survey predictions and feature importance."""
    points, predictions, importance = [], [], []
    for participant in cfg.labels:
        path = io.multivariate_path(cfg, "main", participant)
        if not path.exists():
            continue
        saved = io.load_decoding(path)
        pooled = pooled_predictions(saved["result"], saved["y"])
        point = youden_operating_point(
            pooled["label"].to_numpy(), pooled["ranked_score"].to_numpy()
        )
        points.append({"participant": participant, **point})

        vas = load_recording(cfg, participant).vas
        pooled = pooled.reset_index()
        pooled["survey"] = saved["surveys"][pooled["survey"]]
        pooled["vas"] = vas[pooled["survey"]]
        pooled["predicted_high"] = pooled["ranked_score"] >= point["threshold"]
        pooled.insert(0, "participant", participant)
        predictions.append(pooled)

        if "importance" in saved:
            importance.append(
                pd.DataFrame(
                    {
                        "participant": participant,
                        "feature": saved["feature_names"],
                        "importance": saved["importance"].mean(axis=0),
                        "importance_sd": saved["importance"].std(axis=0),
                    }
                ).sort_values("importance", ascending=False)
            )
    return pd.DataFrame(points), predictions, importance


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    out = cfg.out_dir("multivariate")

    summary = summarize(cfg)
    summary.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))

    points, predictions, importance = main_analysis_details(cfg)
    points.to_csv(out / "operating_points.csv", index=False)
    if predictions:
        pd.concat(predictions).to_csv(out / "survey_predictions.csv", index=False)
    if importance:
        pd.concat(importance).to_csv(out / "feature_importance.csv", index=False)
    print("\nYouden operating points (main analysis)")
    print(points.round(3).to_string(index=False))

    main_rows = summary[summary["analysis"] == "main"]
    if len(main_rows):
        auc = main_rows["mean_auc"]
        print(
            f"\nMain analysis: median AUC = {auc.median():.2f}, mean = {auc.mean():.2f}, "
            f"range = {auc.min():.2f}-{auc.max():.2f}; "
            f"{int((main_rows['p_fdr'] < cfg.alpha).sum())} of {len(main_rows)} participants "
            f"above chance (FDR-corrected p < {cfg.alpha})"
        )
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
