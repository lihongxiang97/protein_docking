"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


def load_config(config_path: Optional[Path] = None) -> dict:
    path = Path(config_path) if config_path else Path(__file__).parent.parent / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    _positive_int(data, "docking", "coarse_rotations")
    _positive_int(data, "docking", "coarse_translations")
    _positive_int(data, "docking", "mc_iterations", allow_zero=True)
    _positive_int(data, "docking", "top_n_poses")
    _positive_number(data, "docking", "translation_step")
    _positive_number(data, "docking", "mc_temperature")
    _probability(data, "ppi_prediction", "interaction_threshold")
    return data


def _get(data: dict, section: str, name: str):
    values = data.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping")
    return values.get(name)


def _positive_int(data: dict, section: str, name: str, allow_zero: bool = False) -> None:
    value = _get(data, section, name)
    minimum = 0 if allow_zero else 1
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
    ):
        raise ValueError(f"{section}.{name} must be an integer >= {minimum}")


def _positive_number(data: dict, section: str, name: str) -> None:
    value = _get(data, section, name)
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
    ):
        raise ValueError(f"{section}.{name} must be a positive number")


def _probability(data: dict, section: str, name: str) -> None:
    value = _get(data, section, name)
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{section}.{name} must be between 0 and 1")
