from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_FEATURE_COLS: tuple[str, ...] = (
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


def add_popularity_features(
    feature_table: pl.DataFrame,
    orders: pl.DataFrame,
    commerces: pl.DataFrame | None = None,
    *,
    kiosk_col: str = "kiosk_id",          # column name in orders / feature_table
    product_col: str = "product_id",
    channel_col: str = "channel",
    region_col: str = "region",
    commerce_kiosk_col: str = "userid",   # column name in commerces
) -> pl.DataFrame:
    """
    Add popularity-based features for candidate products.

    Popularity is computed from orders (long format).
    If channel / region are not present in orders,
    they are joined from commerces.

    Added features:
    - pop_global
    - pop_channel
    - pop_region
    - pop_store
    """

    _ensure_columns(feature_table, REQUIRED_FEATURE_COLS)
    _ensure_columns(orders, REQUIRED_ORDER_COLS)

    LOGGER.info("Adding popularity features")

    # ----------------------------------
    # Prepare orders_long
    # ----------------------------------
    orders_long = orders.select([kiosk_col, product_col])

    # Join channel / region if missing
    need_context = {channel_col, region_col} - set(orders.columns)
    if need_context:
        if commerces is None:
            raise ValueError(
                f"Orders missing columns {need_context} and commerces not provided"
            )

        commerces_ctx = (
            commerces
            .select([commerce_kiosk_col, channel_col, region_col])
            .rename({commerce_kiosk_col: kiosk_col})
        )

        orders_long = orders_long.join(
            commerces_ctx,
            on=kiosk_col,
            how="left",
        )
    else:
        orders_long = orders.select(
            [kiosk_col, product_col, channel_col, region_col]
        )

    # ----------------------------------
    # Store popularity (base aggregation)
    # ----------------------------------
    pop_store = (
        orders_long
        .group_by([kiosk_col, product_col])
        .len()
        .rename({"len": "pop_store"})
    )

    # ----------------------------------
    # Global popularity (aggregate from store-level counts)
    # ----------------------------------
    pop_global = (
        pop_store
        .group_by(product_col)
        .agg(pl.col("pop_store").sum().alias("pop_global"))
    )

    # ----------------------------------
    # Channel popularity
    # ----------------------------------
    pop_channel = (
        orders_long
        .group_by([channel_col, product_col])
        .len()
        .rename({"len": "pop_channel"})
    )

    # ----------------------------------
    # Region popularity
    # ----------------------------------
    pop_region = (
        orders_long
        .group_by([region_col, product_col])
        .len()
        .rename({"len": "pop_region"})
    )

    # ----------------------------------
    # Join into feature table
    # ----------------------------------
    ft = feature_table

    # global popularity (always possible)
    ft = ft.join(
        pop_global,
        left_on="candidate_product_id",
        right_on=product_col,
        how="left",
    )

    # channel popularity (only if channel exists)
    if channel_col in ft.columns:
        ft = ft.join(
            pop_channel,
            left_on=[channel_col, "candidate_product_id"],
            right_on=[channel_col, product_col],
            how="left",
        )
    else:
        ft = ft.with_columns(pl.lit(0).alias("pop_channel"))

    # region popularity (only if region exists)
    if region_col in ft.columns:
        ft = ft.join(
            pop_region,
            left_on=[region_col, "candidate_product_id"],
            right_on=[region_col, product_col],
            how="left",
        )
    else:
        ft = ft.with_columns(pl.lit(0).alias("pop_region"))

    # store popularity (always possible)
    ft = ft.join(
        pop_store,
        left_on=[kiosk_col, "candidate_product_id"],
        right_on=[kiosk_col, product_col],
        how="left",
    )


    # ----------------------------------
    # Fill missing with 0
    # ----------------------------------
    ft = ft.with_columns(
        [
            pl.col("pop_global").fill_null(0),
            pl.col("pop_channel").fill_null(0),
            pl.col("pop_region").fill_null(0),
            pl.col("pop_store").fill_null(0),
        ]
    )

    LOGGER.info("Popularity features added")

    return ft
