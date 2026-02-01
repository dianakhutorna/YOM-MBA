from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_TEST_ORDER_COLS: tuple[str, ...] = (
    "kiosk_id",
    "order_id",
    "product_id",
)

REQUIRED_FEATURE_COLS: tuple[str, ...] = (
    "kiosk_id",
    "anchor_product_id",
    "candidate_product_id",
)


def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def _build_test_pairs(test_orders: pl.DataFrame) -> pl.DataFrame:
    test_baskets = (
        test_orders
        .group_by(["kiosk_id", "order_id"])
        .agg(pl.col("product_id").unique().alias("products"))
        .filter(pl.col("products").list.len() > 1)
    )

    return (
        test_baskets
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .join(
            test_baskets
            .select(["kiosk_id", "products"])
            .explode("products")
            .rename({"products": "candidate_product_id"}),
            on="kiosk_id",
        )
        .filter(pl.col("anchor_product_id") != pl.col("candidate_product_id"))
        .select(
            "kiosk_id",
            "anchor_product_id",
            "candidate_product_id",
        )
        .unique()
        .with_columns(pl.lit(1).alias("label"))
    )


def build_labels(
    feature_table: pl.DataFrame,
    test_orders: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build anchor-based labels for ranking.

    Label = 1 if anchor and candidate were bought together
    in the same basket by the same kiosk in the test period.
    """

    _ensure_columns(feature_table, REQUIRED_FEATURE_COLS)
    _ensure_columns(test_orders, REQUIRED_TEST_ORDER_COLS)

    LOGGER.info("Building labels from feature table: %s", feature_table.shape)

    # 1. Generate anchor–candidate pairs from test baskets
    test_pairs = _build_test_pairs(test_orders)

    # 3. Join with feature table
    labeled = (
        feature_table
        .join(
            test_pairs,
            on=["kiosk_id", "anchor_product_id", "candidate_product_id"],
            how="left",
        )
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )

    return labeled
