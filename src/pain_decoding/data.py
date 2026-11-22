"""Loading of per-participant spectral power features and pain ratings.

The analysis starts from four tables per participant: the pain surveys, the
during-survey and pre-survey spectral power of every gray-matter contact
(one row per survey), and the contacts (anatomical region and MNI
coordinate). The files are described in ``data/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import detrend

from .config import Config



@dataclass
class Recording:
    """All surveys of one participant, restricted to gray-matter contacts.

    Attributes
    ----------
    participant : str
        Manuscript label (P1-P6).
    during, pre : ndarray, shape (n_surveys, n_features)
        Spectral power in the 30 s after survey onset and in the 30 s
        window starting 60 s before it.
    features : DataFrame, one row per column of ``during``
        ``feature`` (contact_band), ``contact``, ``band``, ``region`` and
        ``group`` (region_band; the unit reduced by PCA in the multivariate
        decoder).
    contacts : DataFrame, one row per gray-matter contact with features
        ``contact``, ``region``, ``MNI_x``, ``MNI_y``, ``MNI_z``.
    vas : ndarray
        Pain intensity rating (0-100) of each survey.
    day : ndarray
        Recording day of each survey, counted from the first day (1, 2, ...).
    hours : ndarray
        Time of each survey in hours from the first survey.
    """

    participant: str
    during: np.ndarray
    pre: np.ndarray
    features: pd.DataFrame
    contacts: pd.DataFrame
    vas: np.ndarray
    day: np.ndarray
    hours: np.ndarray

    @property
    def n_surveys(self) -> int:
        return len(self.vas)

    def detrended(self) -> "Recording":
        """Copy with the linear trend over surveys removed from every feature.

        Pain ratings are left untouched, so the high/low labels are the
        same as in the main analysis.
        """
        return Recording(
            participant=self.participant,
            during=detrend(self.during, axis=0),
            pre=detrend(self.pre, axis=0),
            features=self.features,
            contacts=self.contacts,
            vas=self.vas,
            day=self.day,
            hours=self.hours,
        )


def load_recording(cfg: Config, participant: str) -> Recording:
    """Load one participant (P1-P6) from ``cfg.data_dir``."""
    folder = cfg.data_dir / participant
    # Survey times are stored with millisecond precision; read them exactly.
    surveys = pd.read_csv(folder / "surveys.csv", float_precision="round_trip")
    during = pd.read_csv(folder / "power_during_survey.csv")
    pre = pd.read_csv(folder / "power_pre_survey.csv")
    contacts = pd.read_csv(folder / "contacts.csv")
    if contacts["contact"].duplicated().any():
        raise ValueError(f"{participant}: duplicated contact names in contacts.csv")
    if not len(surveys) == len(during) == len(pre):
        raise ValueError(f"{participant}: the survey and power tables differ in length")

    rows = [
        (f"{contact}_{band}", contact, band, region, f"{region}_{band}")
        for contact, region in zip(contacts["contact"], contacts["region"])
        for band in cfg.bands
    ]
    features = pd.DataFrame(rows, columns=["feature", "contact", "band", "region", "group"])

    return Recording(
        participant=participant,
        during=during[features["feature"]].to_numpy(dtype=float),
        pre=pre[features["feature"]].to_numpy(dtype=float),
        features=features,
        contacts=contacts,
        vas=surveys["pain_vas"].to_numpy(dtype=float),
        day=surveys["day"].to_numpy(),
        hours=(surveys["seconds"] / 3600).to_numpy(),
    )


def binarize_pain(
    vas: np.ndarray, low_fraction: float = 0.33, high_fraction: float = 0.66
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Split ratings into low (0) and high (1) pain, excluding the middle.

    The thresholds are the order statistics at ``int(n * fraction)`` of the
    sorted ratings. Ratings at or below the lower threshold are low pain and
    ratings at or above the upper threshold are high pain.

    Returns
    -------
    keep : ndarray of int
        Indices of the surveys that received a label.
    y : ndarray of int
        Label (0 or 1) of each kept survey.
    low_threshold, high_threshold : float
    """
    ordered = np.sort(vas)
    low_threshold = ordered[int(len(ordered) * low_fraction)]
    high_threshold = ordered[int(len(ordered) * high_fraction)]
    label = np.full(len(vas), -1)
    label[vas <= low_threshold] = 0
    label[vas >= high_threshold] = 1
    keep = np.flatnonzero(label >= 0)
    return keep, label[keep], float(low_threshold), float(high_threshold)
