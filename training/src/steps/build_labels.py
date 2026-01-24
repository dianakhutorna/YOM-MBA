import polars as pl


def build_labels(
    feature_table: pl.DataFrame,
    test_orders: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build anchor-based labels for ranking.

    Label = 1 if anchor and candidate were bought together
    in the same basket by the same kiosk in the test period.
    """

    # 1. Build baskets from test orders
    test_baskets = (
        test_orders
        .group_by(["kiosk_id", "order_id"])
        .agg(pl.col("product_id").unique().alias("products"))
        .filter(pl.col("products").list.len() > 1)
    )

    # 2. Generate anchor–candidate pairs from test baskets
    test_pairs = (
        test_baskets
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .join(
            test_baskets
            .select(["kiosk_id", "products"])
            .explode("products")
            .rename({"products": "candidate_product_id"}),
            on="kiosk_id",
        )
        .filter(pl.col("anchor_product_id") != pl.col("candidate_product_id"))
        .select(
            "kiosk_id",
            "anchor_product_id",
            "candidate_product_id",
        )
        .unique()
        .with_columns(pl.lit(1).alias("label"))
    )

    # 3. Join with feature table
    labeled = (
        feature_table
        .join(
            test_pairs,
            on=["kiosk_id", "anchor_product_id", "candidate_product_id"],
            how="left",
        )
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )

    return labeled

