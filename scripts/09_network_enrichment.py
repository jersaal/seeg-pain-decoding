"""Network-level enrichment of univariate decoding performance (Fig. 5,
Supplementary Table 4).

Contacts of participants with above-chance multivariate decoding are
assigned to Yeo 7 networks (Schaefer atlas) and the mean ranked AUC of each
network-band combination is tested against chance. The test is repeated for
every atlas resolution and distance threshold listed under
``network.sensitivity`` in the configuration.

Requires the outputs of 03_multivariate_summary.py and 07_univariate_decoding.py.

Outputs in results/network/:
  contact_assignments.csv   network of every contact (manuscript configuration)
  enrichment.csv            permutation test (manuscript configuration)
  network_vs_rest.csv       Mann-Whitney test against all other networks
  sensitivity.csv           permutation test for every configuration
  atlas.csv                 parcel volume of each atlas
"""

import numpy as np
import pandas as pd

from pain_decoding import io, network
from pain_decoding.config import base_parser, load_config
from pain_decoding.stats import fdr_bh

CONTACT = ["participant", "contact"]


def assign(models, atlas, max_distance_mm):
    """Add a ``network`` column; returns the per-contact table as well."""
    contacts = models.drop_duplicates(CONTACT)[CONTACT + ["region", "MNI_x", "MNI_y", "MNI_z"]]
    contacts = contacts.reset_index(drop=True)
    contacts["network"] = network.assign_networks(
        contacts[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(), atlas, max_distance_mm
    )
    models = models.drop(columns="network", errors="ignore").merge(
        contacts[CONTACT + ["network"]], on=CONTACT, how="left"
    )
    return models, contacts


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    params = cfg.network
    out = cfg.out_dir("network")
    pd.set_option("display.width", 250)

    participants = io.decodable_participants(cfg)
    models = io.load_ranked_models(cfg, participants)
    print(f"{len(models)} contact-band models from {', '.join(participants)}")

    manuscript = (params["n_parcels"], params["max_distance_mm"])
    configurations = [tuple(c) for c in params["sensitivity"]]
    if manuscript not in configurations:
        configurations.append(manuscript)

    atlases, atlas_rows, sensitivity = {}, [], []
    for n_parcels, max_distance in configurations:
        if n_parcels not in atlases:
            atlases[n_parcels] = network.load_schaefer_atlas(n_parcels)
            atlas_rows.append(
                {
                    "n_parcels": n_parcels,
                    "median_volume_cm3": atlases[n_parcels].median_volume_cm3,
                    "equivalent_radius_mm": atlases[n_parcels].equivalent_radius_mm,
                }
            )
        assigned, contacts = assign(models, atlases[n_parcels], max_distance)
        result = network.enrichment_test(
            assigned,
            cfg.bands,
            params["n_permutations"],
            np.random.RandomState(cfg.seed),
            params["min_contacts"],
            cfg.alpha,
        )
        n_assigned = int((contacts["network"] != network.UNASSIGNED).sum())
        result.insert(0, "n_parcels", n_parcels)
        result.insert(1, "max_distance_mm", max_distance)
        result.insert(2, "n_contacts_assigned", n_assigned)
        result.insert(3, "pct_contacts_assigned", 100 * n_assigned / len(contacts))
        sensitivity.append(result)
        significant = result[result["significant"]].sort_values("p_fdr")
        print(
            f"Schaefer {n_parcels}, {max_distance} mm: {n_assigned}/{len(contacts)} contacts "
            "assigned; significant: "
            + "; ".join(
                f"{r.network} x {r.band} (FDR p = {r.p_fdr:.3f}, n = {r.n_contacts})"
                for r in significant.itertuples()
            ),
            flush=True,
        )

        if (n_parcels, max_distance) == manuscript:
            contacts.to_csv(out / "contact_assignments.csv", index=False)
            result.to_csv(out / "enrichment.csv", index=False)
            versus_rest = network.network_versus_rest(
                assigned, cfg.bands, params["min_contacts"]
            )
            versus_rest["p_fdr"] = fdr_bh(versus_rest["p"], cfg.alpha)[1]
            versus_rest.to_csv(out / "network_vs_rest.csv", index=False)
            coverage = contacts["network"].value_counts()

    pd.concat(sensitivity, ignore_index=True).to_csv(out / "sensitivity.csv", index=False)
    pd.DataFrame(atlas_rows).to_csv(out / "atlas.csv", index=False)
    print("\nContacts per network (manuscript configuration)")
    print(coverage.to_string())
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
