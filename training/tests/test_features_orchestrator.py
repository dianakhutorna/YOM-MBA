from __future__ import annotations

import polars as pl

from training.src.features import add_all_features
from training.src.steps.build_feature_table import build_feature_table


def test_add_all_features_noop(baskets_df, topk_df, cleaned_orders_df):
    """With no products/commerces, only order-based features are added."""
    base = build_feature_table(baskets_df, topk_df)
    out = add_all_features(
        base,
        orders=cleaned_orders_df,
        products=None,
        commerces=None,
    )
    # order-based features should be added
    assert "pop_store" in out.columns
    assert "pop_global" in out.columns
    assert "kiosk_product_cnt" in out.columns
    assert "cand_is_new" in out.columns
    # product/commerce features should NOT be added
    assert "same_category" not in out.columns
    assert "channel" not in out.columns


def test_add_all_features_product_flag(baskets_df, topk_df, cleaned_orders_df):
    base = build_feature_table(baskets_df, topk_df)
    # minimal products table
    products = (
        cleaned_orders_df
        .select(pl.col("product_id").alias("productid"))
        .unique()
        .with_columns(pl.lit("cat").alias("category"))
    )
    out = add_all_features(
        base,
        orders=cleaned_orders_df,
        products=products,
        commerces=None,
    )
    assert "same_category" in out.columns
