from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_FEATURE_COLS: tuple[str, ...] = (
    "kiosk_id",
    "anchor_product_id",
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


def add_behavioral_features(
    feature_table: pl.DataFrame,
    train_orders: pl.DataFrame,
) -> pl.DataFrame:

    kiosk_product_counts = (
        train_orders
        .group_by(["kiosk_id", "product_id"])
        .len()
        .rename({"len": "pop_store"})
    )

    ft = feature_table.join(
        kiosk_product_counts,
        left_on=["kiosk_id", "candidate_product_id"],
        right_on=["kiosk_id", "product_id"],
        how="left",
    )

    return ft.with_columns(
        pl.col("pop_store").fill_null(0)
    )

