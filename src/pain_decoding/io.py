"""Reading and writing of intermediate results shared between scripts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .multivariate import DecodingResult

MULTIVARIATE_ANALYSES = ("main", "detrended", "cross_context", "one_day_buffer")


# ---------------------------------------------------------------------------
# Multivariate decoding
# ---------------------------------------------------------------------------

def multivariate_path(cfg: Config, analysis: str, participant: str) -> Path:
    return cfg.out_dir("multivariate", analysis) / f"{participant}.npz"


def save_decoding(
    path: Path,
    result: DecodingResult,
    y: np.ndarray,
    surveys: np.ndarray,
    feature_names: Sequence[str],
    null_auc: Optional[np.ndarray] = None,
) -> None:
    """Store one participant's cross-validated output.

    ``surveys`` maps the labeled surveys back to rows of the feature table.
    Per-fold test indices and scores are stored concatenated, with
    ``test_fold`` giving the fold of each entry.
    """
    arrays = dict(
        fold_auc=result.fold_auc,
        fold_n_train=result.fold_n_train,
        test_index=np.concatenate(result.fold_test),
        test_score=np.concatenate(result.fold_score),
        test_fold=np.concatenate([np.full(len(t), k) for k, t in enumerate(result.fold_test)]),
        y=y,
        surveys=surveys,
        feature_names=np.array(feature_names),
    )
    if result.importance is not None:
        arrays["importance"] = result.importance
    if null_auc is not None:
        arrays["null_auc"] = null_auc
    np.savez_compressed(path, **arrays)


def load_decoding(path: Path) -> dict:
    """Inverse of :func:`save_decoding`; returns a dict with a ``result`` entry."""
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    n_folds = len(out["fold_auc"])
    out["result"] = DecodingResult(
        fold_auc=out["fold_auc"],
        fold_test=[out["test_index"][out["test_fold"] == k] for k in range(n_folds)],
        fold_score=[out["test_score"][out["test_fold"] == k] for k in range(n_folds)],
        fold_n_train=out["fold_n_train"],
        importance=out.get("importance"),
    )
    return out


def decodable_participants(cfg: Config) -> List[str]:
    """Participants with above-chance multivariate decoding in the main analysis.

    Group-level univariate and network analyses are restricted to these
    participants. Requires the output of ``03_multivariate_summary.py``.
    """
    path = cfg.results_dir / "multivariate" / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/02_multivariate_decoding.py --analysis main "
            "and scripts/03_multivariate_summary.py first."
        )
    summary = pd.read_csv(path)
    main = summary[summary["analysis"] == "main"]
    if main["p_fdr"].isna().any():
        raise ValueError("The main analysis has no permutation p-values yet.")
    significant = set(main.loc[main["p_fdr"] < cfg.alpha, "participant"])
    return [label for label in cfg.labels if label in significant]


# ---------------------------------------------------------------------------
# Univariate decoding
# ---------------------------------------------------------------------------

def univariate_models_path(cfg: Config, participant: str) -> Path:
    return cfg.out_dir("univariate", "models") / f"{participant}_models.csv"


def univariate_null_path(cfg: Config, participant: str) -> Path:
    return cfg.out_dir("univariate", "models") / f"{participant}_null_auc.npz"


def load_univariate_models(cfg: Config, participants: Sequence[str]) -> pd.DataFrame:
    """Per-model univariate results of several participants, stacked."""
    tables = []
    for participant in participants:
        table = pd.read_csv(univariate_models_path(cfg, participant))
        table.insert(0, "participant", participant)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def load_ranked_models(cfg: Config, participants: Sequence[str]) -> pd.DataFrame:
    """Univariate models with MNI coordinates and within-participant ranked AUCs.

    Contacts without an MNI coordinate are dropped before ranking.
    """
    from .data import load_recording
    from .network import rank_within_participant

    coordinates = []
    for participant in participants:
        contacts = load_recording(cfg, participant).contacts
        contacts = contacts[["contact", "MNI_x", "MNI_y", "MNI_z"]].copy()
        contacts.insert(0, "participant", participant)
        coordinates.append(contacts)
    models = load_univariate_models(cfg, participants).merge(
        pd.concat(coordinates), on=["participant", "contact"], how="left"
    )
    models = models.dropna(subset=["MNI_x", "MNI_y", "MNI_z"]).reset_index(drop=True)
    models["ranked_auc"] = rank_within_participant(models)
    return models
