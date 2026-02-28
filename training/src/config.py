from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatureConfig:
    include_product_features: bool = True
    include_kiosk_features: bool = True
    include_behavioral_features: bool = True
    include_personalization_features: bool = False
    include_popularity_features: bool = False
    encode_channel: bool = True
    encode_region: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "FeatureConfig":
        data = _load_yaml(path)
        return cls(
            include_product_features=bool(data.get("include_product_features", cls.include_product_features)),
            include_kiosk_features=bool(data.get("include_kiosk_features", cls.include_kiosk_features)),
            include_behavioral_features=bool(data.get("include_behavioral_features", cls.include_behavioral_features)),
            include_personalization_features=bool(data.get("include_personalization_features", cls.include_personalization_features)),
            include_popularity_features=bool(data.get("include_popularity_features", cls.include_popularity_features)),
            encode_channel=bool(data.get("encode_channel", cls.encode_channel)),
            encode_region=bool(data.get("encode_region", cls.encode_region)),
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def load_yaml_config(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    extends = data.pop("extends", None)
    if not extends:
        return data
    base_paths: list[Path]
    if isinstance(extends, str):
        base_paths = [path.parent / extends]
    elif isinstance(extends, list):
        base_paths = [path.parent / str(p) for p in extends]
    else:
        raise ValueError("extends must be a string or list of strings")
    merged: dict[str, Any] = {}
    for base_path in base_paths:
        merged = _merge_dicts(merged, load_yaml_config(base_path))
    return _merge_dicts(merged, data)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
