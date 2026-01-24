from __future__ import annotations
import polars as pl

def add_kiosk_history_features(
    feature_table: pl.DataFrame,
    train_orders: pl.DataFrame,
) -> pl.DataFrame:
    """
    Adds kiosk-specific signals from train period only.

    Requires:
    - feature_table has: kiosk_id, candidate_product_id
    - train_orders has: kiosk_id, product_id
    """

    # 1) how often kiosk bought each product in train
    kiosk_prod = (
        train_orders
        .group_by(["kiosk_id", "product_id"])
        .agg(pl.len().alias("kiosk_product_cnt"))
        .rename({"product_id": "candidate_product_id"})
    )

    out = (
        feature_table
        .join(kiosk_prod, on=["kiosk_id", "candidate_product_id"], how="left")
        .with_columns([
            pl.col("kiosk_product_cnt").fill_null(0),
            (pl.col("kiosk_product_cnt") > 0).cast(pl.Int8).alias("kiosk_bought_candidate_before"),
        ])
    )

    return out
