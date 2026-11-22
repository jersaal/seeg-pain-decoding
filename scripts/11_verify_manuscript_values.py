"""Compare the reproduced results with the values printed in the manuscript.

Every statistic quoted in the Results, Methods, figure legends and
supplementary tables is recomputed from the files in ``results/``, formatted
as in the manuscript and compared with the printed value.

Output: results/manuscript_values.csv; the exit status is non-zero if any
value differs or is missing.
"""

import sys

import numpy as np
import pandas as pd

from pain_decoding.config import base_parser, load_config

ROWS = []


def check(section, quantity, printed, reproduced):
    ROWS.append(
        {
            "section": section,
            "quantity": quantity,
            "manuscript": printed,
            "reproduced": reproduced,
            "match": printed == reproduced,
        }
    )


def fmt(values, spec):
    """Format one value or a sequence of per-participant values."""
    if np.ndim(values) == 0:
        return format(values, spec)
    return ", ".join(format(v, spec) for v in values)


def span(values, spec):
    return f"{format(min(values), spec)}-{format(max(values), spec)}"


def with_ci(table, value, spec=".2f"):
    return "; ".join(
        f"{format(r[value], spec)} [{format(r['ci_low'], spec)}, {format(r['ci_high'], spec)}]"
        for _, r in table.iterrows()
    )


# ---------------------------------------------------------------------------

def cohort(res):
    t = pd.read_csv(res / "cohort" / "participant_summary.csv")
    s = "Cohort (Results; Supplementary Table 1)"
    check(s, "surveys, P1-P6", "72, 57, 88, 89, 72, 105", fmt(t.n_surveys, "d"))
    check(s, "recording days", "10, 8, 10, 10, 9, 8", fmt(t.n_days, "d"))
    check(
        s,
        "median VAS [IQR]",
        "55.5 [42.8-60.2]; 50.0 [40.0-52.0]; 50.0 [31.0-60.0]; 56.0 [50.0-60.0]; "
        "15.0 [4.5-43.0]; 82.0 [73.0-90.0]",
        "; ".join(f"{r.vas_median:.1f} [{r.vas_q1:.1f}-{r.vas_q3:.1f}]" for r in t.itertuples()),
    )
    check(
        s,
        "low / high thresholds",
        "47.0 / 59.0; 42.0 / 51.0; 39.0 / 58.0; 52.0 / 59.0; 10.0 / 33.0; 75.0 / 86.0",
        "; ".join(f"{r.low_threshold:.1f} / {r.high_threshold:.1f}" for r in t.itertuples()),
    )
    check(s, "high-pain surveys", "25, 26, 30, 31, 25, 40", fmt(t.n_high_pain, "d"))
    check(s, "low-pain surveys", "24, 22, 30, 30, 25, 38", fmt(t.n_low_pain, "d"))
    check(
        s, "surveys per class, range", "22-40",
        span(list(t.n_high_pain) + list(t.n_low_pain), "d"),
    )
    check(s, "labeled surveys", "49, 48, 60, 61, 50, 78", fmt(t.n_high_pain + t.n_low_pain, "d"))
    check(
        s, "median inter-survey interval (h), range", "0.3-1.2",
        span(t.median_inter_survey_hours, ".1f"),
    )
    check(s, "gray-matter contacts", "60, 68, 61, 57, 64, 90", fmt(t.n_gray_matter_contacts, "d"))
    check(s, "univariate models", "360, 408, 366, 342, 384, 540", fmt(t.n_univariate_models, "d"))
    check(s, "region-level features, range", "120-204", span(t.n_region_band_features, "d"))
    gap = t.high_threshold - t.low_threshold
    close = gap[gap <= 12]
    check(
        s, "participants with thresholds 7-12 VAS points apart", "4 (7-12)",
        f"{len(close)} ({span(close, '.0f')})",
    )


def multivariate(res, alpha):
    t = pd.read_csv(res / "multivariate" / "summary.csv")
    main = t[t.analysis == "main"].reset_index(drop=True)
    decodable = list(main.participant[main.p_fdr < alpha])

    s = "Multivariate decoding (Results; Fig. 2; Supplementary Table 2)"
    check(s, "median AUC", "0.72", fmt(main.mean_auc.median(), ".2f"))
    check(s, "mean AUC", "0.70", fmt(main.mean_auc.mean(), ".2f"))
    check(s, "AUC range", "0.41-0.91", span(main.mean_auc, ".2f"))
    check(s, "participants above chance", "P1, P3, P4, P5, P6", ", ".join(decodable))
    check(s, "retained folds", "41, 27, 30, 44, 22, 28", fmt(main.n_folds, "d"))

    printed = {
        "main": (
            "0.74 [0.67, 0.81]; 0.41 [0.34, 0.48]; 0.77 [0.71, 0.83]; 0.69 [0.63, 0.75]; "
            "0.91 [0.85, 0.97]; 0.70 [0.64, 0.75]",
            "0.013, 0.788, 0.003, 0.023, 0.003, 0.013",
        ),
        "detrended": (
            "0.73 [0.66, 0.81]; 0.59 [0.51, 0.67]; 0.69 [0.60, 0.77]; 0.68 [0.62, 0.74]; "
            "0.49 [0.35, 0.64]; 0.74 [0.67, 0.80]",
            "0.030, 0.247, 0.030, 0.040, 0.516, 0.006",
        ),
        "cross_context": (
            "0.70 [0.64, 0.76]; 0.42 [0.35, 0.49]; 0.72 [0.64, 0.80]; 0.74 [0.68, 0.80]; "
            "0.96 [0.92, 0.99]; 0.62 [0.55, 0.69]",
            "0.025, 0.795, 0.008, 0.003, 0.003, 0.047",
        ),
        "one_day_buffer": (
            "0.75 [0.67, 0.82]; 0.48 [0.40, 0.56]; 0.73 [0.64, 0.80]; 0.66 [0.59, 0.72]; "
            "0.76 [0.66, 0.85]; 0.64 [0.57, 0.71]",
            "0.006, 0.587, 0.006, 0.038, 0.006, 0.025",
        ),
    }
    for analysis, (auc, p) in printed.items():
        rows = t[t.analysis == analysis]
        check(s, f"{analysis}: AUC [95% CI]", auc, with_ci(rows, "mean_auc"))
        check(s, f"{analysis}: FDR-corrected p", p, fmt(rows.p_fdr, ".3f"))

    cross = t[t.analysis == "cross_context"]
    check(s, "cross-context median AUC", "0.71", fmt(cross.mean_auc.median(), ".2f"))
    buffer = t[t.analysis == "one_day_buffer"]
    check(s, "one-day buffer: retained folds", "37, 25, 27, 44, 20, 28", fmt(buffer.n_folds, "d"))
    check(
        s, "one-day buffer: median AUC, decodable participants", "0.73",
        fmt(buffer[buffer.participant.isin(decodable)].mean_auc.median(), ".2f"),
    )

    points = pd.read_csv(res / "multivariate" / "operating_points.csv")
    points = points[points.participant.isin(decodable)]
    low, high = points.sensitivity.idxmin(), points.sensitivity.idxmax()
    check(
        s, "sensitivity range", "0.70 (P3) to 0.92 (P5)",
        f"{points.sensitivity[low]:.2f} ({points.participant[low]}) to "
        f"{points.sensitivity[high]:.2f} ({points.participant[high]})",
    )
    low, high = points.specificity.idxmin(), points.specificity.idxmax()
    check(
        s, "specificity range", "0.57 (P4) to 0.93 (P3)",
        f"{points.specificity[low]:.2f} ({points.participant[low]}) to "
        f"{points.specificity[high]:.2f} ({points.participant[high]})",
    )

    r = pd.read_csv(res / "multivariate" / "feature_redundancy.csv")
    s = "Feature redundancy (Results; Supplementary Fig. 6)"
    low, high = r.median_abs_r.idxmin(), r.median_abs_r.idxmax()
    check(
        s, "median |r| range", "0.20 (P4) to 0.36 (P3)",
        f"{r.median_abs_r[low]:.2f} ({r.participant[low]}) to "
        f"{r.median_abs_r[high]:.2f} ({r.participant[high]})",
    )
    check(s, "effective dimensions, range", "20-35", span(r.effective_dim, "d"))
    check(s, "effective dimensions, % of features", "15-18", span(r.effective_dim_pct, ".0f"))


def temporal(res):
    t = pd.read_csv(res / "temporal" / "decorrelation.csv")
    s = "Temporal autocorrelation (Results)"
    check(
        s, "pain ratings decorrelated",
        "within 1 day, within 1 day, by 2 days, within 1 day, by 3 days, by 2 days",
        ", ".join(t.pain_decorrelated),
    )
    check(s, "features decorrelated by 2 days (%)", "58-98", span(t.neural_pct_by_2_days, ".0f"))
    check(s, "features decorrelated by 3 days (%)", "94-100", span(t.neural_pct_by_3_days, ".0f"))


def regression(res):
    t = pd.read_csv(res / "regression" / "summary.csv")
    s = "Continuous regression (Results; Supplementary Table 3)"
    check(s, "surveys", "72, 57, 88, 89, 72, 105", fmt(t.n_surveys, "d"))
    check(s, "folds", "43, 28, 45, 45, 36, 28", fmt(t.n_folds, "d"))
    check(
        s, "Spearman rho [95% CI]",
        "0.05 [-0.05, 0.15]; -0.22 [-0.31, -0.15]; 0.28 [0.19, 0.36]; 0.28 [0.19, 0.38]; "
        "0.41 [0.29, 0.53]; 0.33 [0.27, 0.39]",
        with_ci(t, "spearman"),
    )
    check(s, "FDR-corrected p", "0.394, 0.943, 0.022, 0.022, 0.003, 0.003", fmt(t.p_fdr, ".3f"))
    check(s, "R2", "-0.35, -0.24, 0.17, 0.11, 0.34, -0.07", fmt(t.r2, ".2f"))


def univariate(res):
    t = pd.read_csv(res / "univariate" / "participant_summary.csv", keep_default_na=False)
    s = "Univariate decoding (Results; Fig. 3)"
    check(s, "median AUC across models, range", "0.48-0.67", span(t.median_auc, ".2f"))
    check(
        s, "significant contacts, FDR across bands of a contact",
        "28/60 (47%), 0/68 (0%), 53/61 (87%), 25/57 (44%), 55/64 (86%), 34/90 (38%)",
        ", ".join(
            f"{r.n_significant_contacts_contact_fdr}/{r.n_contacts} "
            f"({r.pct_significant_contacts_contact_fdr:.0f}%)"
            for r in t.itertuples()
        ),
    )
    check(
        s, "significant contacts (%), FDR across all tests of a participant",
        "37, 0, 89, 12, 88, 12", fmt(t.pct_significant_contacts_participant_fdr, ".0f"),
    )
    check(
        s, "dominant band", "low_gamma, , high_gamma, beta, beta, beta",
        ", ".join(t.dominant_band),
    )
    p5 = t[t.participant == "P5"].iloc[0]
    check(
        s, "best model, P5", "R_Putamen high_gamma 0.99",
        f"{p5.best_region} {p5.best_feature.split('_', 1)[1]} {p5.best_auc:.2f}",
    )
    decodable = t[t.n_significant_features > 0]
    check(
        s, "top-10 features with higher power in high pain (P1, P3, P4, P5, P6)",
        "10, 0, 8, 5, 0", fmt(decodable.top10_increased_with_pain, "d"),
    )

    b = pd.read_csv(res / "univariate" / "band_tests.csv").set_index("participant")
    check(
        s, "Friedman FDR p (P1, P3, P4, P5, P6)", "0.007, 6.8e-05, 2.6e-04, 0.005, 2.1e-14",
        ", ".join(
            format(b.friedman_p_fdr[p], ".3f" if b.friedman_p_fdr[p] >= 0.001 else ".1e")
            for p in ["P1", "P3", "P4", "P5", "P6"]
        ),
    )
    check(
        s, "Wilcoxon FDR p (P1, P3, P4, P5, P6)", "0.010, 0.47, 0.042, 0.010, 1.8e-07",
        ", ".join(
            [
                format(b.wilcoxon_p_fdr["P1"], ".3f"),
                format(b.wilcoxon_p_fdr["P3"], ".2f"),
                format(b.wilcoxon_p_fdr["P4"], ".3f"),
                format(b.wilcoxon_p_fdr["P5"], ".3f"),
                format(b.wilcoxon_p_fdr["P6"], ".1e"),
            ]
        ),
    )

    r = pd.read_csv(res / "univariate" / "region_consistency.csv")
    s = "Regions shared across participants (Results; Fig. 4)"

    def consistent(scheme):
        rows = r[(r.scheme == scheme) & r.all_covered_significant]
        return "; ".join(
            f"{x.region} ({x.n_participants_covered}/{x.n_participants_covered})"
            for x in rows.sort_values("region").itertuples()
        )

    check(
        s, "significant in every covered participant, FDR across bands",
        "L_AINS (4/4); L_DLPFC (4/4); L_PINS (4/4); R_DMPFC (5/5); R_MDThal (3/3); "
        "R_PINS (3/3); R_VPLThal (3/3)",
        consistent("contact_fdr"),
    )
    check(
        s, "significant in every covered participant, FDR across all tests",
        "L_DLPFC (4/4); R_VPLThal (3/3)", consistent("participant_fdr"),
    )
    dmpfc = r[(r.scheme == "participant_fdr") & (r.region == "R_DMPFC")].iloc[0]
    check(
        s, "R DMPFC, FDR across all tests", "4 of 5",
        f"{dmpfc.n_participants_significant} of {dmpfc.n_participants_covered}",
    )


def network(res):
    s = "Network enrichment (Results; Fig. 5; Methods)"
    contacts = pd.read_csv(res / "network" / "contact_assignments.csv")
    assigned = contacts[contacts.network != "Unassigned"]
    check(
        s, "contacts assigned to a network", "106 (32%)",
        f"{len(assigned)} ({100 * len(assigned) / len(contacts):.0f}%)",
    )
    counts = assigned.network.value_counts()
    order = ["Salience", "Frontoparietal", "Default Mode", "Somatomotor",
             "Dorsal Attention", "Limbic", "Visual"]
    check(
        s, "contacts per network (Sal, FP, DMN, SomMot, DorsAttn, Limbic, Visual)",
        "36, 17, 23, 24, 5, 1, 0", fmt([int(counts.get(n, 0)) for n in order], "d"),
    )

    e = pd.read_csv(res / "network" / "enrichment.csv")
    check(s, "network-band combinations tested", "30", str(len(e)))
    check(
        s, "significant combinations (mean ranked AUC, FDR p, n)",
        "Default Mode x beta (0.70, 0.018, 23); Salience x high_gamma (0.71, 0.006, 36)",
        "; ".join(
            f"{x.network} x {x.band} ({x.mean_ranked_auc:.2f}, {x.p_fdr:.3f}, {x.n_contacts})"
            for x in e[e.significant].sort_values("network").itertuples()
        ),
    )
    v = pd.read_csv(res / "network" / "network_vs_rest.csv").set_index(["network", "band"])
    for key, printed in [
        (("Salience", "high_gamma"), "0.67 (36 vs 70, p = 0.003)"),
        (("Default Mode", "beta"), "0.64 (23 vs 83, p = 0.023)"),
    ]:
        x = v.loc[key]
        check(
            s, f"{key[0]} x {key[1]} versus other networks", printed,
            f"{x.effect:.2f} ({int(x.n_inside)} vs {int(x.n_rest)}, p = {x.p:.3f})",
        )
    a = pd.read_csv(res / "network" / "atlas.csv").set_index("n_parcels").loc[400]
    check(
        s, "Schaefer-400 median parcel volume (cm3), equivalent radius (mm)", "2.5, 8.4",
        f"{a.median_volume_cm3:.1f}, {a.equivalent_radius_mm:.1f}",
    )

    s = "Network sensitivity (Supplementary Table 4)"
    t = pd.read_csv(res / "network" / "sensitivity.csv")
    printed = {
        (400, 5): "50 (15%): Default Mode x low_gamma (0.017, 11); Salience x beta (0.041, 16); "
                  "Salience x high_gamma (0.017, 16); Salience x low_gamma (0.022, 16)",
        (400, 7): "106 (32%): Default Mode x beta (0.018, 23); Salience x high_gamma (0.006, 36)",
        (400, 10): "201 (61%): Default Mode x beta (0.001, 52); Limbic x beta (0.001, 8); "
                   "Salience x high_gamma (0.001, 51); Salience x low_gamma (0.003, 51)",
        (200, 7): "51 (15%): Default Mode x beta (0.043, 8); "
                  "Frontoparietal x high_gamma (0.043, 19); "
                  "Frontoparietal x low_gamma (0.006, 19); Limbic x beta (0.043, 5)",
        (1000, 7): "172 (52%): Default Mode x beta (0.001, 42); "
                   "Frontoparietal x low_gamma (0.049, 38); "
                   "Salience x high_gamma (0.001, 44); Salience x low_gamma (0.002, 44)",
    }
    for (n_parcels, mm), text in printed.items():
        rows = t[(t.n_parcels == n_parcels) & (t.max_distance_mm == mm)]
        head = f"{rows.n_contacts_assigned.iloc[0]} ({rows.pct_contacts_assigned.iloc[0]:.0f}%)"
        cells = "; ".join(
            f"{x.network} x {x.band} ({x.p_fdr:.3f}, {x.n_contacts})"
            for x in rows[rows.significant].sort_values(["network", "band"]).itertuples()
        )
        check(s, f"Schaefer {n_parcels}, {mm} mm", text, f"{head}: {cells}")

    s = "Coordinate jitter (Results)"
    j = pd.read_csv(res / "network" / "coordinate_jitter_summary.csv")
    j = j.set_index(["network", "band", "jitter_mm"])
    sal = [j.loc[("Salience", "high_gamma", mm)] for mm in (2, 3)]
    dmn = [j.loc[("Default Mode", "beta", mm)] for mm in (2, 3)]
    check(
        s, "Salience x high_gamma: % significant at 2, 3 mm", "100, 99.7",
        ", ".join(format(x.pct_significant, ".4g") for x in sal),
    )
    check(
        s, "Salience x high_gamma: median FDR p", "0.003, 0.003",
        fmt([x.p_fdr_median for x in sal], ".3f"),
    )
    check(
        s, "Default Mode x beta: % significant at 2, 3 mm", "78, 79",
        fmt([x.pct_significant for x in dmn], ".0f"),
    )
    check(
        s, "Default Mode x beta: median FDR p [IQR]",
        "0.011 [0.003-0.041], 0.012 [0.003-0.042]",
        ", ".join(f"{x.p_fdr_median:.3f} [{x.p_fdr_q25:.3f}-{x.p_fdr_q75:.3f}]" for x in dmn),
    )


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    res = cfg.results_dir

    sections = [
        ("cohort", lambda: cohort(res)),
        ("multivariate", lambda: multivariate(res, cfg.alpha)),
        ("temporal", lambda: temporal(res)),
        ("regression", lambda: regression(res)),
        ("univariate", lambda: univariate(res)),
        ("network", lambda: network(res)),
    ]
    missing = []
    for name, run in sections:
        try:
            run()
        except FileNotFoundError as error:
            missing.append(f"{name}: {error.filename} not found")

    table = pd.DataFrame(ROWS)
    table.to_csv(res / "manuscript_values.csv", index=False)
    for section, rows in table.groupby("section", sort=False):
        print(f"\n{section}")
        for r in rows.itertuples():
            print(f"  [{'ok' if r.match else 'DIFFERS'}] {r.quantity}: {r.reproduced}")
            if not r.match:
                print(f"            manuscript: {r.manuscript}")
    for line in missing:
        print(f"\nNOT CHECKED - {line}")

    n_differ = int((~table["match"]).sum())
    print(f"\n{int(table['match'].sum())} of {len(table)} values match the manuscript.")
    if n_differ or missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
