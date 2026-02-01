from __future__ import annotations

import polars as pl


def add_kiosk_history_features(
    feature_table: pl.DataFrame,
    train_orders: pl.DataFrame,
    commerces: pl.DataFrame,
) -> pl.DataFrame:

    # kiosk-level behavioral stats
    kiosk_stats = (
        train_orders
        .group_by("kiosk_id")
        .agg([
            pl.count().alias("kiosk_product_cnt"),
        ])
    )

    # load static kiosk info (channel + region)
    commerces = commerces.select([
        pl.col("userid").alias("kiosk_id"),
        "channel",
        "region",
    ])


    # join everything
    feature_table = (
        feature_table
        .join(kiosk_stats, on="kiosk_id", how="left")
        .join(commerces, on="kiosk_id", how="left")
    )

    return feature_table
