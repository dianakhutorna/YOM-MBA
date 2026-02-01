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
    *,
    kiosk_col: str = "kiosk_id",
    product_col: str = "product_id",
) -> pl.DataFrame:
    """
    Add kiosk-specific behavioral features:

    - pop_store:
        how many times kiosk bought candidate
    - kiosk_bought_candidate_before:
        binary flag (0/1)
    - anchor_kiosk_frequency:
        how many times kiosk bought anchor
    """

    _ensure_columns(feature_table, REQUIRED_FEATURE_COLS)
    _ensure_columns(train_orders, REQUIRED_ORDER_COLS)

    LOGGER.info("Adding behavioral (kiosk-specific) features")

    # ----------------------------------
    # Long orders: kiosk × product
    # ----------------------------------
    orders_long = (
        train_orders
        .select([kiosk_col, product_col])
    )

    # ----------------------------------
    # pop_store: kiosk × candidate
    # ----------------------------------
    pop_store = (
        orders_long
        .group_by([kiosk_col, product_col])
        .len()
        .rename({"len": "pop_store"})
    )

    # ----------------------------------
    # Join pop_store for candidate
    # ----------------------------------
    ft = feature_table.join(
        pop_store,
        left_on=[kiosk_col, "candidate_product_id"],
        right_on=[kiosk_col, product_col],
        how="left",
    )

    # ----------------------------------
    # kiosk_bought_candidate_before (binary)
    # ----------------------------------
    ft = ft.with_columns(
        (pl.col("pop_store") > 0)
        .cast(pl.Int8)
        .alias("kiosk_bought_candidate_before")
    )

    # ----------------------------------
    # anchor_kiosk_frequency
    # ----------------------------------
    anchor_freq = (
        orders_long
        .group_by([kiosk_col, product_col])
        .len()
        .rename({"len": "anchor_kiosk_frequency"})
    )

    ft = ft.join(
        anchor_freq,
        left_on=[kiosk_col, "anchor_product_id"],
        right_on=[kiosk_col, product_col],
        how="left",
    )

    # ----------------------------------
    # Fill missing with 0
    # ----------------------------------
    ft = ft.with_columns(
        [
            pl.col("pop_store").fill_null(0),
            pl.col("anchor_kiosk_frequency").fill_null(0),
        ]
    )

    LOGGER.info("Behavioral features added")

    return ft
