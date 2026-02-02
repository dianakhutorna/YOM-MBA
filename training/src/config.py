from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from training.src.paths import EXTERNAL_DIR, INTERIM_DIR


@dataclass(frozen=True)
class CandidateConfig:
    min_cooc: int = 3
    min_lift: float = 2.0
    top_k: int = 20


@dataclass(frozen=True)
class FeatureConfig:
    include_product_features: bool = True
    include_kiosk_features: bool = True
    include_behavioral_features: bool = True
    include_personalization_features: bool = True
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


@dataclass(frozen=True)
class SplitConfig:
    split_date: datetime = datetime(2024, 1, 4)


@dataclass(frozen=True)
class OfflineExperimentConfig:
    orders_path: Path = INTERIM_DIR / "orders_sample.parquet"
    products_path: Path = EXTERNAL_DIR / "products_v2.csv"
    labeled_train_out_path: Path = INTERIM_DIR / "labeled_features_train.parquet"
    labeled_test_out_path: Path = INTERIM_DIR / "labeled_features_test.parquet"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    label_window_days: int | None = 7
    min_cooc: int = 3
    min_lift: float = 2.0
    top_k: int = 20

    @classmethod
    def from_yaml(cls, path: Path) -> "OfflineExperimentConfig":
        data = _load_yaml(path)
        return cls(
            orders_path=Path(data.get("orders_path", cls.orders_path)),
            products_path=Path(data.get("products_path", cls.products_path)),
            labeled_train_out_path=Path(
                data.get("labeled_train_out_path", cls.labeled_train_out_path)
            ),
            labeled_test_out_path=Path(
                data.get("labeled_test_out_path", cls.labeled_test_out_path)
            ),
            train_ratio=float(data.get("train_ratio", cls.train_ratio)),
            val_ratio=float(data.get("val_ratio", cls.val_ratio)),
            test_ratio=float(data.get("test_ratio", cls.test_ratio)),
            label_window_days=data.get("label_window_days", cls.label_window_days),
            min_cooc=int(data.get("min_cooc", cls.min_cooc)),
            min_lift=float(data.get("min_lift", cls.min_lift)),
            top_k=int(data.get("top_k", cls.top_k)),
        )


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"Unsupported datetime value: {value!r}")


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
