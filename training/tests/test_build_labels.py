from __future__ import annotations

import polars as pl


def test_build_labels_schema(baskets_df, topk_df, cleaned_orders_df):
    from training.src.steps.build_feature_table import build_feature_table
    from training.src.steps.build_labels import build_labels

    # Use the same orders as "test" for a simple shape check.
    features = build_feature_table(baskets_df, topk_df)
    labeled = build_labels(features, cleaned_orders_df)

    expected = {"kiosk_id", "anchor_product_id", "candidate_product_id", "label"}
    assert expected.issubset(set(labeled.columns))
    assert labeled.schema["label"] == pl.Int8


def test_build_labels_binary_values(baskets_df, topk_df, cleaned_orders_df):
    from training.src.steps.build_feature_table import build_feature_table
    from training.src.steps.build_labels import build_labels

    features = build_feature_table(baskets_df, topk_df)
    labeled = build_labels(features, cleaned_orders_df)

    values = labeled.select(pl.col("label").unique()).to_series().to_list()
    assert set(values).issubset({0, 1})
