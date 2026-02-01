from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_FEATURE_COLS: tuple[str, ...] = (
    "kiosk_id",
    "candidate_product_id",
)

REQUIRED_ORDER_COLS: tuple[str, ...] = (
    "kiosk_id",
    "product_id",
)


def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def add_personalization_features(
    feature_table: pl.DataFrame,
    train_orders: pl.DataFrame,
) -> pl.DataFrame:
    """
    Adds:
    - cand_is_new_for_kiosk
    - anchor_kiosk_frequency
    """

    _ensure_columns(feature_table, REQUIRED_FEATURE_COLS)
    _ensure_columns(train_orders, REQUIRED_ORDER_COLS)

    LOGGER.info("Adding personalization features")

    # ----------------------------------
    # 1. cand_is_new_for_kiosk
    # ----------------------------------
    kiosk_product_history = (
        train_orders
        .select(["kiosk_id", "product_id"])
        .unique()
        .with_columns(pl.lit(1).alias("bought_before"))
    )

    ft = feature_table.join(
        kiosk_product_history,
        left_on=["kiosk_id", "candidate_product_id"],
        right_on=["kiosk_id", "product_id"],
        how="left",
    )

    ft = ft.with_columns(
        pl.when(pl.col("bought_before").is_null())
        .then(1)
        .otherwise(0)
        .alias("cand_is_new_for_kiosk")
    ).drop("bought_before")

    

    LOGGER.info("Personalization features added")

    return ft
