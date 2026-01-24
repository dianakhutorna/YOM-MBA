from __future__ import annotations
import polars as pl


def build_labels(
    feature_table: pl.DataFrame,
    test_orders: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add binary labels to feature table using test orders.

    Label = 1 if candidate_product was bought by kiosk in test period.
    """

    print("[INFO] Building labels")

    # --------------------------------------
    # 1. Extract test purchases per kiosk
    # --------------------------------------

    test_purchases = (
        test_orders
        .select(["kiosk_id", "product_id"])
        .unique()
        .with_columns(pl.lit(1).alias("label"))
        .rename({"product_id": "candidate_product_id"})
    )

    print(f"[INFO] Test purchases: {test_purchases.shape}")

    # --------------------------------------
    # 2. Join with feature table
    # --------------------------------------

    labeled = (
        feature_table
        .join(
            test_purchases,
            on=["kiosk_id", "candidate_product_id"],
            how="left",
        )
        .with_columns(
            pl.col("label").fill_null(0)
        )
    )

    print(
        "[INFO] Label distribution:\n",
        labeled.select(pl.col("label").value_counts())
    )

    return labeled
