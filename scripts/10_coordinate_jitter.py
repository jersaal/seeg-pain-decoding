"""Sensitivity of the network enrichment to electrode localization.

Every contact's MNI coordinate is displaced by a fixed distance in an
independent, uniformly random direction; contacts are then reassigned to
networks and the enrichment test is repeated. Reported is the share of
repetitions in which each network-band combination of the manuscript
analysis remains significant.

Requires the outputs of 09_network_enrichment.py.

Outputs in results/network/:
  coordinate_jitter.csv            one row per repetition and combination
  coordinate_jitter_summary.csv    per displacement distance and combination
  coordinate_jitter_reference.csv  enrichment test without displacement
"""

import time

import numpy as np
import pandas as pd

from pain_decoding import io, network
from pain_decoding.config import base_parser, load_config

CONTACT = ["participant", "contact"]


def contact_band_grid(models, bands):
    """Ranked AUCs as one row per contact and band, contacts in table order."""
    contacts = models.drop_duplicates(CONTACT).reset_index(drop=True)
    grid = contacts[CONTACT].merge(pd.DataFrame({"band": list(bands)}), how="cross")
    grid = grid.merge(
        models[CONTACT + ["band", "ranked_auc"]], on=CONTACT + ["band"], how="left"
    )
    if grid["ranked_auc"].isna().any():
        raise ValueError("Every contact needs a model in every frequency band.")
    return contacts, grid


def main():
    parser = base_parser(__doc__)
    parser.add_argument(
        "--n-jitter", type=int, default=None, help="Override the configured number."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    params = cfg.network
    out = cfg.out_dir("network")
    n_jitter = params["n_jitter"] if args.n_jitter is None else args.n_jitter

    published = pd.read_csv(out / "enrichment.csv")
    targets = [(r.network, r.band) for r in published[published["significant"]].itertuples()]

    models = io.load_ranked_models(cfg, io.decodable_participants(cfg))
    contacts, grid = contact_band_grid(models, cfg.bands)
    xyz = contacts[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)
    atlas = network.load_schaefer_atlas(params["n_parcels"])
    rng = np.random.default_rng(cfg.seed)

    def test(coordinates):
        assigned = network.assign_networks(coordinates, atlas, params["max_distance_mm"])
        grid["network"] = np.repeat(assigned, len(cfg.bands))
        result = network.enrichment_test(
            grid, cfg.bands, params["n_permutations"], rng, params["min_contacts"], cfg.alpha
        )
        return result, int(np.sum(assigned != network.UNASSIGNED))

    reference, n_assigned = test(xyz)
    reference.to_csv(out / "coordinate_jitter_reference.csv", index=False)
    print(f"{len(contacts)} contacts, {n_assigned} assigned without displacement", flush=True)

    rows = []
    for distance in params["jitter_mm"]:
        start = time.time()
        for repetition in range(n_jitter):
            result, n_assigned = test(network.jitter_coordinates(xyz, distance, rng))
            n_significant = int(result["significant"].sum())
            result = result.set_index(["network", "band"])
            for target in targets:
                tested = target in result.index
                rows.append(
                    {
                        "jitter_mm": distance,
                        "repetition": repetition,
                        "network": target[0],
                        "band": target[1],
                        "n_contacts_assigned": n_assigned,
                        "n_significant_combinations": n_significant,
                        "n_contacts": int(result.loc[target, "n_contacts"]) if tested else 0,
                        "p": result.loc[target, "p"] if tested else np.nan,
                        "p_fdr": result.loc[target, "p_fdr"] if tested else np.nan,
                        "significant": bool(result.loc[target, "significant"]) if tested else False,
                    }
                )
            if (repetition + 1) % 100 == 0:
                print(
                    f"{distance} mm: {repetition + 1}/{n_jitter} [{time.time() - start:.0f} s]",
                    flush=True,
                )
    repetitions = pd.DataFrame(rows)
    repetitions.to_csv(out / "coordinate_jitter.csv", index=False)

    summary = (
        repetitions.groupby(["jitter_mm", "network", "band"], sort=False)
        .agg(
            n_repetitions=("significant", "size"),
            pct_significant=("significant", lambda s: 100 * s.mean()),
            p_fdr_median=("p_fdr", "median"),
            p_fdr_q25=("p_fdr", lambda s: s.quantile(0.25)),
            p_fdr_q75=("p_fdr", lambda s: s.quantile(0.75)),
            n_contacts_median=("n_contacts", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out / "coordinate_jitter_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
