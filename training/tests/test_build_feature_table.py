from __future__ import annotations


def test_build_feature_table_columns(baskets_df, topk_df):
    from training.src.steps.build_feature_table import build_feature_table

    features = build_feature_table(baskets_df, topk_df)
    expected = {
        "kiosk_id",
        "anchor_product_id",
        "candidate_product_id",
    }
    assert expected.issubset(set(features.columns))


def test_build_feature_table_non_empty(baskets_df, topk_df):
    from training.src.steps.build_feature_table import build_feature_table

    features = build_feature_table(baskets_df, topk_df)
    assert features.height > 0
