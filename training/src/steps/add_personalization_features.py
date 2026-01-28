from __future__ import annotations
import polars as pl


def add_personalization_features(
    feature_table: pl.DataFrame,
    train_orders: pl.DataFrame,
) -> pl.DataFrame:
    """
    Adds:
    - cand_is_new_for_kiosk
    - anchor_kiosk_frequency
    """

    print("[INFO] Adding personalization features")

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

    

    print("[INFO] Personalization features added")

    return ft
