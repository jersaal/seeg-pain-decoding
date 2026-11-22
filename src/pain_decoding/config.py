"""Configuration loading and command-line conventions shared by the scripts."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Parsed ``config.yaml``.

    Top-level scalars are attributes; the per-analysis sections
    (``multivariate``, ``univariate``, ...) are plain dictionaries.
    """

    data_dir: Path
    results_dir: Path
    participants: List[str]
    bands: List[str]
    seed: int
    n_workers: int
    alpha: float
    pain_labels: dict
    multivariate: dict
    regression: dict
    univariate: dict
    temporal: dict
    network: dict

    @property
    def labels(self) -> List[str]:
        """Participant labels (P1, P2, ...) in manuscript order."""
        return list(self.participants)

    def out_dir(self, *parts: str) -> Path:
        """Results sub-directory, created if needed."""
        path = self.results_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: Optional[os.PathLike] = None) -> Config:
    """Read the configuration; relative paths are resolved against the file."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f)

    def resolve(p: str) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (path.parent / p).resolve()

    return Config(
        data_dir=resolve(raw["data_dir"]),
        results_dir=resolve(raw["results_dir"]),
        participants=list(raw["participants"]),
        bands=list(raw["frequency_bands"]),
        seed=int(raw["random_seed"]),
        n_workers=int(raw.get("n_workers") or os.cpu_count() or 1),
        alpha=float(raw["alpha"]),
        pain_labels=raw["pain_labels"],
        multivariate=raw["multivariate"],
        regression=raw["regression"],
        univariate=raw["univariate"],
        temporal=raw["temporal"],
        network=raw["network"],
    )


def base_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the options common to every script."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", default=None, help="Path to config.yaml (default: repository root)."
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Participant labels to process, e.g. P1 P4 (default: all).",
    )
    return parser


def selected_participants(cfg: Config, labels: Optional[List[str]]) -> List[str]:
    """Validate a ``--participants`` selection and return it in manuscript order."""
    if not labels:
        return cfg.labels
    unknown = sorted(set(labels) - set(cfg.labels))
    if unknown:
        raise SystemExit(f"Unknown participant(s) {unknown}; expected {cfg.labels}")
    return [lab for lab in cfg.labels if lab in labels]
