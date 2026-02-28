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
    max_items: int = 200,
) -> pl.DataFrame:
    """
    Build baskets from cleaned orders data.

    Filters out baskets with fewer than *min_items* or more than
    *max_items* unique products (anomaly guard).

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
        .group_by(["kiosk_id", "order_id"])
        .agg(
            pl.col("product_id")
            .unique()
            .alias("products")
        )
        .with_columns(pl.col("products").list.len().alias("_basket_size"))
    )

    total_before = baskets.height

    # Filter by size bounds
    too_small = baskets.filter(pl.col("_basket_size") < min_items).height
    too_large = baskets.filter(pl.col("_basket_size") > max_items).height
    baskets = baskets.filter(
        (pl.col("_basket_size") >= min_items) & (pl.col("_basket_size") <= max_items)
    )

    # Log size distribution
    if baskets.height > 0:
        size_stats = baskets.select(
            pl.col("_basket_size").min().alias("min"),
            pl.col("_basket_size").mean().alias("mean"),
            pl.col("_basket_size").median().alias("median"),
            pl.col("_basket_size").quantile(0.95).alias("p95"),
            pl.col("_basket_size").quantile(0.99).alias("p99"),
            pl.col("_basket_size").max().alias("max"),
        ).row(0)
        LOGGER.info(
            "Basket size stats — min: %d, mean: %.1f, median: %.0f, "
            "p95: %.0f, p99: %.0f, max: %d",
            *size_stats,
        )

    if too_small > 0 or too_large > 0:
        LOGGER.info(
            "Baskets filtered: %d total -> %d kept "
            "(removed %d with <%d items, %d with >%d items)",
            total_before, baskets.height,
            too_small, min_items, too_large, max_items,
        )
    else:
        LOGGER.info("Built baskets: %s (no anomalous sizes)", baskets.shape)

    baskets = baskets.drop("_basket_size")

    return baskets
