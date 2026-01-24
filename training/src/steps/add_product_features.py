from __future__ import annotations
import polars as pl


def add_product_features(
    features: pl.DataFrame,
    products: pl.DataFrame,
) -> pl.DataFrame:
    """
    Enrich feature table with product-level pair features.

    Adds:
    - same_category (0/1)
    """

    print("[INFO] Adding product category features")

    # Canonical product table
    prod = (
        products
        .select(
            pl.col("productid").alias("product_id"),
            pl.col("category"),
        )
        .unique(subset=["product_id"])
    )

    # Join anchor product category
    features = features.join(
        prod.rename({
            "product_id": "anchor_product_id",
            "category": "anchor_category",
        }),
        on="anchor_product_id",
        how="left",
    )

    # Join candidate product category
    features = features.join(
        prod.rename({
            "product_id": "candidate_product_id",
            "category": "candidate_category",
        }),
        on="candidate_product_id",
        how="left",
    )

    # Pair-wise feature
    features = features.with_columns(
        (pl.col("anchor_category") == pl.col("candidate_category"))
        .cast(pl.Int8)
        .alias("same_category")
    )

    print(
        "[INFO] same_category ratio:",
        features.select(pl.col("same_category").mean()).item()
    )

    return features
