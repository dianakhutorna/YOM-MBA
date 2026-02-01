from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_FEATURE_COLS: tuple[str, ...] = (
    "anchor_product_id",
    "candidate_product_id",
)

REQUIRED_PRODUCT_COLS: tuple[str, ...] = (
    "productid",
    "category",
)


def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def add_product_features(
    features: pl.DataFrame,
    products: pl.DataFrame,
) -> pl.DataFrame:
    """
    Enrich feature table with product-level pair features.

    Adds:
    - same_category (0/1)
    """

    _ensure_columns(features, REQUIRED_FEATURE_COLS)
    _ensure_columns(products, REQUIRED_PRODUCT_COLS)

    LOGGER.info("Adding product category features")

    # Canonical product table
    prod = (
        products
        .select(
            pl.col("productid").alias("product_id"),
            pl.col("category"),
        )
        .unique(subset=["product_id"])
    )

    # Join anchor product category
    features = features.join(
        prod.rename({
            "product_id": "anchor_product_id",
            "category": "anchor_category",
        }),
        on="anchor_product_id",
        how="left",
    )

    # Join candidate product category
    features = features.join(
        prod.rename({
            "product_id": "candidate_product_id",
            "category": "candidate_category",
        }),
        on="candidate_product_id",
        how="left",
    )

    # Pair-wise feature
    features = features.with_columns(
        (pl.col("anchor_category") == pl.col("candidate_category"))
        .cast(pl.Int8)
        .alias("same_category")
    )

    ratio = features.select(pl.col("same_category").mean()).item()
    LOGGER.info("same_category ratio: %s", ratio)

    return features
