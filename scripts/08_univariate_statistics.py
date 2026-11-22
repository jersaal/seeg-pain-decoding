"""Significance and anatomical distribution of the univariate decoders
(Fig. 3c-f, Fig. 4, Supplementary Fig. 8).

Requires the outputs of 07_univariate_decoding.py and, for the group-level
region table, of 03_multivariate_summary.py.

Outputs in results/univariate/:
  features.csv             p-values of every contact-band model, both corrections
  contacts.csv             per-contact significance
  participant_summary.csv  model counts, median AUC, significant contacts
  band_proportions.csv     share of significant features in each band
  band_tests.csv           Friedman and Wilcoxon tests across bands
  region_consistency.csv   regions significant in every covered participant
  top_features.csv         strongest models per participant, with direction
"""

import numpy as np
import pandas as pd

from pain_decoding import io, univariate
from pain_decoding.config import base_parser, load_config
from pain_decoding.stats import fdr_bh

SCHEMES = {
    "contact_fdr": "significant_contact_fdr",
    "participant_fdr": "significant_participant_fdr",
}


def load_features(cfg):
    tables = []
    for participant in cfg.labels:
        models = pd.read_csv(io.univariate_models_path(cfg, participant))
        with np.load(io.univariate_null_path(cfg, participant)) as saved:
            null = pd.DataFrame(saved["null_auc"], columns=saved["feature"])
        table = univariate.feature_significance(models, null, cfg.alpha)
        table.insert(0, "participant", participant)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def participant_summary(cfg, features, contacts):
    top_n = cfg.univariate["top_n_features"]
    rows = []
    for participant, models in features.groupby("participant", sort=False):
        own = contacts[contacts["participant"] == participant]
        significant = models[models["significant_contact_fdr"]]
        row = {
            "participant": participant,
            "n_contacts": len(own),
            "n_models": len(models),
            "median_auc": models["auc"].median(),
            "best_feature": models.iloc[0]["feature"],
            "best_region": models.iloc[0]["region"],
            "best_auc": models.iloc[0]["auc"],
            "n_significant_features": len(significant),
            "dominant_band": significant["band"].value_counts().idxmax()
            if len(significant)
            else "",
            f"top{top_n}_increased_with_pain": int((models.head(top_n)["direction"] == "+").sum()),
        }
        for scheme, column in SCHEMES.items():
            row[f"n_significant_contacts_{scheme}"] = int(own[column].sum())
            row[f"pct_significant_contacts_{scheme}"] = 100 * own[column].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def band_proportions(cfg, features):
    significant = features[features["significant_contact_fdr"]]
    counts = (
        significant.groupby(["participant", "band"]).size().unstack(fill_value=0)
        .reindex(index=cfg.labels, columns=cfg.bands, fill_value=0)
    )
    return counts.div(counts.sum(axis=1), axis=0).reset_index()


def band_test_table(cfg, features):
    rows = []
    for participant, models in features.groupby("participant", sort=False):
        rows.append({"participant": participant, **univariate.band_tests(models, cfg.bands)})
    table = pd.DataFrame(rows)
    table["friedman_p_fdr"] = fdr_bh(table["friedman_p"], cfg.alpha)[1]
    table["wilcoxon_p_fdr"] = fdr_bh(table["wilcoxon_p"], cfg.alpha)[1]
    return table


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    out = cfg.out_dir("univariate")
    pd.set_option("display.width", 250)

    features = load_features(cfg)
    contacts = univariate.contact_significance(features)
    features.to_csv(out / "features.csv", index=False)
    contacts.to_csv(out / "contacts.csv", index=False)

    summary = participant_summary(cfg, features, contacts)
    summary.to_csv(out / "participant_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    band_proportions(cfg, features).to_csv(out / "band_proportions.csv", index=False)
    tests = band_test_table(cfg, features)
    tests.to_csv(out / "band_tests.csv", index=False)
    print("\nBand tests")
    print(tests.to_string(index=False))

    top_n = cfg.univariate["top_n_features"]
    top = features.groupby("participant", sort=False).head(top_n)
    columns = ["participant", "feature", "region", "band", "auc", "auc_sd", "n_folds", "direction"]
    top[columns].to_csv(out / "top_features.csv", index=False)

    decodable = io.decodable_participants(cfg)
    covered = contacts[contacts["participant"].isin(decodable)]
    regions = []
    for scheme, column in SCHEMES.items():
        table = univariate.region_consistency(
            covered, column, cfg.univariate["min_participants_per_region"]
        )
        table.insert(0, "scheme", scheme)
        regions.append(table)
        consistent = table[table["all_covered_significant"]]
        print(
            f"\n{scheme}: {len(consistent)} of {len(table)} regions significant in every "
            "covered participant: "
            + "; ".join(
                f"{r} ({n}/{n})"
                for r, n in zip(consistent["region"], consistent["n_participants_covered"])
            )
        )
    pd.concat(regions).to_csv(out / "region_consistency.csv", index=False)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
