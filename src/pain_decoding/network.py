"""Network-level organization of univariate decoding performance.

Contacts are assigned to the Yeo 7 networks through the nearest parcel
centroid of the Schaefer cortical atlas, and network-by-band enrichment of
within-participant ranked AUCs is tested by permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, rankdata

from .stats import fdr_bh

NETWORKS = [
    "Visual", "Somatomotor", "Dorsal Attention", "Salience",
    "Limbic", "Frontoparietal", "Default Mode",
]
_SCHAEFER_NETWORK_KEYS = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal",
    "Default": "Default Mode",
}
UNASSIGNED = "Unassigned"

Rng = Union[np.random.RandomState, np.random.Generator]


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------

@dataclass
class Atlas:
    """Parcel centroids (MNI, mm), network labels and volumes of a Schaefer atlas."""

    n_parcels: int
    centroids: np.ndarray       # (n_parcels, 3)
    networks: np.ndarray        # (n_parcels,) network name of each parcel
    volumes_mm3: np.ndarray     # (n_parcels,)

    @property
    def median_volume_cm3(self) -> float:
        return float(np.median(self.volumes_mm3)) / 1000.0

    @property
    def equivalent_radius_mm(self) -> float:
        """Radius of a sphere with the median parcel volume."""
        return float((3 * np.median(self.volumes_mm3) / (4 * np.pi)) ** (1 / 3))


def load_schaefer_atlas(n_parcels: int = 400) -> Atlas:
    """Schaefer 2018 atlas (Yeo 7-network labels, 1 mm MNI152) via nilearn.

    The atlas is downloaded to ``~/nilearn_data`` on first use.
    """
    from nilearn import datasets, image

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_parcels, yeo_networks=7, resolution_mm=1
    )
    img = image.load_img(atlas.maps)
    volume = np.asarray(img.get_fdata())
    labels = [lab.decode() if isinstance(lab, bytes) else lab for lab in atlas.labels]
    # Label format: "7Networks_LH_SalVentAttn_Med_1"
    networks = np.array([_SCHAEFER_NETWORK_KEYS[lab.split("_")[2]] for lab in labels])

    voxels = np.argwhere(volume > 0)
    parcel = volume[volume > 0].astype(int)
    counts = np.bincount(parcel, minlength=n_parcels + 1)[1:]
    centroid_voxel = np.column_stack(
        [np.bincount(parcel, weights=voxels[:, k], minlength=n_parcels + 1)[1:] / counts
         for k in range(3)]
    )
    centroids = np.array([(img.affine @ np.append(c, 1.0))[:3] for c in centroid_voxel])
    voxel_volume = float(abs(np.linalg.det(img.affine[:3, :3])))
    return Atlas(n_parcels, centroids, networks, counts * voxel_volume)


def assign_networks(xyz: np.ndarray, atlas: Atlas, max_distance_mm: float) -> np.ndarray:
    """Network of each contact, or ``UNASSIGNED``.

    A contact takes the network of the parcel with the nearest centroid if
    that centroid lies within ``max_distance_mm``.
    """
    distance = cdist(np.asarray(xyz, dtype=float), atlas.centroids)
    nearest = distance.argmin(axis=1)
    network = atlas.networks[nearest].astype(object)
    network[distance[np.arange(len(nearest)), nearest] > max_distance_mm] = UNASSIGNED
    return network.astype(str)


# ---------------------------------------------------------------------------
# Ranked AUCs
# ---------------------------------------------------------------------------

def rank_within_participant(models: pd.DataFrame) -> np.ndarray:
    """AUC ranks across all contact-band models of each participant, scaled to (0, 1].

    Ranking within participants keeps participants with higher overall
    decoding performance from dominating the pooled analysis.
    """
    ranked = np.empty(len(models))
    for _, rows in models.groupby("participant", sort=False).indices.items():
        ranked[rows] = rankdata(models["auc"].to_numpy()[rows]) / len(rows)
    return ranked


# ---------------------------------------------------------------------------
# Enrichment test
# ---------------------------------------------------------------------------

def _permute_within_participants(
    values: np.ndarray, participant: np.ndarray, n_permutations: int, rng: Rng
) -> np.ndarray:
    """``n_permutations`` copies of ``values``, each shuffled within participants.

    Both NumPy generator types are supported so that the published values
    can be regenerated exactly: the enrichment tests of the main and
    sensitivity analyses were run with ``RandomState`` (one shuffle per
    participant and permutation) and those of the coordinate-jitter analysis
    with ``Generator`` (all permutations of a participant drawn at once).
    The two are statistically equivalent.
    """
    permuted = np.empty((n_permutations, len(values)))
    members = [np.flatnonzero(participant == p) for p in pd.unique(participant)]
    if isinstance(rng, np.random.Generator):
        for rows in members:
            permuted[:, rows] = rng.permuted(np.tile(values[rows], (n_permutations, 1)), axis=1)
    else:
        for k in range(n_permutations):
            for rows in members:
                permuted[k, rows] = rng.permutation(values[rows])
    return permuted


def enrichment_test(
    models: pd.DataFrame,
    bands: Sequence[str],
    n_permutations: int,
    rng: Rng,
    min_contacts: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Is the mean ranked AUC of a network-band combination above chance?

    Ranked AUCs are shuffled within each participant across all assigned
    contact-band models, which preserves each participant's distribution of
    decoding performance and breaks its association with network and band.
    One-sided p-values are Benjamini-Hochberg corrected across all
    combinations with at least ``min_contacts`` models.

    ``models`` needs the columns ``participant``, ``network``, ``band`` and
    ``ranked_auc``; unassigned contacts are ignored.
    """
    assigned = models[models["network"] != UNASSIGNED]
    values = assigned["ranked_auc"].to_numpy()
    network = assigned["network"].to_numpy()
    band = assigned["band"].to_numpy()

    cells, membership = [], []
    for net in NETWORKS:
        for b in bands:
            member = (network == net) & (band == b)
            if member.sum() >= min_contacts:
                cells.append((net, b))
                membership.append(member)
    if not cells:
        return pd.DataFrame()
    membership = np.array(membership, dtype=float)      # (n_cells, n_models)
    n = membership.sum(axis=1)

    observed = membership @ values / n
    permuted = _permute_within_participants(
        values, assigned["participant"].to_numpy(), n_permutations, rng
    )
    null = permuted @ membership.T / n                  # (n_permutations, n_cells)
    p = (np.sum(null >= observed[None, :], axis=0) + 1) / (n_permutations + 1)
    reject, p_fdr = fdr_bh(p, alpha)
    return pd.DataFrame(
        {
            "network": [c[0] for c in cells],
            "band": [c[1] for c in cells],
            "n_contacts": n.astype(int),
            "mean_ranked_auc": observed,
            "null_mean": null.mean(axis=0),
            "null_sd": null.std(axis=0),
            "p": p,
            "p_fdr": p_fdr,
            "significant": reject,
        }
    )


def network_versus_rest(
    models: pd.DataFrame, bands: Sequence[str], min_contacts: int = 5
) -> pd.DataFrame:
    """Contacts of a network against contacts of all other networks, per band.

    One-sided Mann-Whitney U test on ranked AUCs. ``effect`` is the
    probability that a contact in the network outranks a contact elsewhere.
    """
    assigned = models[models["network"] != UNASSIGNED]
    rows = []
    for band in bands:
        in_band = assigned[assigned["band"] == band]
        for net in NETWORKS:
            inside = in_band.loc[in_band["network"] == net, "ranked_auc"]
            rest = in_band.loc[in_band["network"] != net, "ranked_auc"]
            if len(inside) < min_contacts:
                continue
            u, p = mannwhitneyu(inside, rest, alternative="greater")
            rows.append(
                {
                    "network": net,
                    "band": band,
                    "n_inside": len(inside),
                    "n_rest": len(rest),
                    "effect": u / (len(inside) * len(rest)),
                    "p": p,
                }
            )
    return pd.DataFrame(rows)


def jitter_coordinates(xyz: np.ndarray, distance_mm: float, rng: np.random.Generator) -> np.ndarray:
    """Displace every coordinate by ``distance_mm`` in an independent random direction."""
    direction = rng.standard_normal(xyz.shape)
    direction *= distance_mm / np.linalg.norm(direction, axis=1, keepdims=True)
    return xyz + direction
