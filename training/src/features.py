from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from training.src.config import FeatureConfig
from training.src.steps.add_behavioral_features import add_behavioral_features
from training.src.steps.add_kiosk_features import add_kiosk_history_features
from training.src.steps.add_personalization_features import add_personalization_features
from training.src.steps.add_popularity_features import add_popularity_features
from training.src.steps.add_product_features import add_product_features
from training.src.steps.encode_categorical_features import encode_channel_one_hot
from training.src.steps.encode_region_one_hot import encode_region_one_hot


def add_all_features(
    feature_table: pl.DataFrame,
    *,
    orders: pl.DataFrame,
    products: pl.DataFrame | None,
    commerces: pl.DataFrame | None,
    config: FeatureConfig,
) -> pl.DataFrame:
    """
    Orchestrate feature augmentation based on FeatureConfig flags.
    """
    ft = feature_table

    if config.include_product_features:
        if products is None:
            raise ValueError("products is required when include_product_features=True")
        ft = add_product_features(ft, products)

    if config.include_kiosk_features:
        if commerces is None:
            raise ValueError("commerces is required when include_kiosk_features=True")
        ft = add_kiosk_history_features(ft, orders, commerces)

    if config.include_behavioral_features:
        ft = add_behavioral_features(ft, orders)

    if config.include_personalization_features:
        ft = add_personalization_features(ft, orders)

    if config.include_popularity_features:
        ft = add_popularity_features(ft, orders, commerces=commerces)

    if config.encode_channel:
        ft = encode_channel_one_hot(ft)

    if config.encode_region:
        ft = encode_region_one_hot(ft)

    return ft
