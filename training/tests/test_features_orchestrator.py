from __future__ import annotations

import polars as pl

from training.src.config import FeatureConfig
from training.src.features import add_all_features
from training.src.steps.build_feature_table import build_feature_table


def test_add_all_features_noop(baskets_df, topk_df, cleaned_orders_df):
    base = build_feature_table(baskets_df, topk_df)
    cfg = FeatureConfig(
        include_product_features=False,
        include_kiosk_features=False,
        include_behavioral_features=False,
        include_personalization_features=False,
        include_popularity_features=False,
        encode_channel=False,
        encode_region=False,
    )
    out = add_all_features(
        base,
        orders=cleaned_orders_df,
        products=None,
        commerces=None,
        config=cfg,
    )
    assert out.columns == base.columns


def test_add_all_features_product_flag(baskets_df, topk_df, cleaned_orders_df):
    base = build_feature_table(baskets_df, topk_df)
    cfg = FeatureConfig(
        include_product_features=True,
        include_kiosk_features=False,
        include_behavioral_features=False,
        include_personalization_features=False,
        include_popularity_features=False,
        encode_channel=False,
        encode_region=False,
    )
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
        config=cfg,
    )
    assert "same_category" in out.columns
