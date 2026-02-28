from __future__ import annotations

import polars as pl

from training.src.config import FeatureConfig
from training.src.steps.add_behavioral_features import add_behavioral_features
from training.src.steps.add_kiosk_features import add_kiosk_history_features
from training.src.steps.add_personalization_features import add_personalization_features
from training.src.steps.add_popularity_features import add_popularity_features
from training.src.steps.add_product_features import add_product_features
from training.src.steps.encode_categorical_features import encode_channel_one_hot
from training.src.steps.encode_region_one_hot import encode_region_one_hot


# ============================================================
# Shared encoding — MUST be used by both training and inference
# to guarantee identical feature representation.
# ============================================================

def lgbm_feature_exprs(
    feature_cols: list[str],
    categorical_cols: list[str],
) -> list[pl.Expr]:
    """
    Build Polars expressions that convert feature columns to the
    numeric representation expected by LightGBM.

    Categorical columns are hashed to UInt64 then cast to Float64.
    Numeric columns are cast to Float64 with null → 0.
    """
    exprs: list[pl.Expr] = []
    cat_set = set(categorical_cols)
    for c in feature_cols:
        if c in cat_set:
            exprs.append(
                pl.col(c)
                .cast(pl.Utf8)
                .fill_null("__MISSING__")
                .hash(seed=42, seed_1=43, seed_2=44, seed_3=45)
                .cast(pl.Float64)
            )
        else:
            exprs.append(pl.col(c).fill_null(0).cast(pl.Float64))
    return exprs


# ============================================================
# Feature orchestration
# ============================================================

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
        # pop_store may already exist from behavioral features; drop to avoid collision
        if "pop_store" in ft.columns:
            ft = ft.drop("pop_store")
        ft = add_popularity_features(ft, orders, commerces=commerces)

    if config.encode_channel:
        ft = encode_channel_one_hot(ft)

    if config.encode_region:
        ft = encode_region_one_hot(ft)

    return ft
