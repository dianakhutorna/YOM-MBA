from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_ORDER_COLS: tuple[str, ...] = (
    "order_id",
    "kiosk_id",
    "product_id",
)


def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def build_baskets(
    orders: pl.DataFrame,
    min_items: int = 2,
) -> pl.DataFrame:
    """
    Build baskets from cleaned orders data.

    Input schema:
    - order_id
    - kiosk_id
    - product_id
    - order_dt
    - quantity

    Output schema:
    - kiosk_id
    - order_id
    - products: list[str]
    """

    _ensure_columns(orders, REQUIRED_ORDER_COLS)

    LOGGER.info("Building baskets from orders: %s", orders.shape)

    baskets = (
        orders
        # group by kiosk + order
        .group_by(["kiosk_id", "order_id"])
        .agg(
            pl.col("product_id")
            .unique()
            .alias("products")
        )
        # keep only baskets with >= min_items
        .filter(pl.col("products").list.len() >= min_items)
    )

    LOGGER.info("Built baskets: %s", baskets.shape)

    avg_size = baskets.select(
        pl.col("products").list.len().mean()
    ).item()

    if avg_size is not None:
        LOGGER.info("Avg basket size: %.2f", avg_size)
    else:
        LOGGER.info("Avg basket size: n/a (no baskets)")

    return baskets
