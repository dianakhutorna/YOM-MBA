from __future__ import annotations

def test_offline_pipeline_smoke(cleaned_orders_df, baskets_df, topk_df):
    from training.src.steps.build_feature_table import build_feature_table
    from training.src.steps.build_labels import build_labels

    features = build_feature_table(baskets_df, topk_df)
    labeled = build_labels(features, cleaned_orders_df)

    assert features.height > 0
    assert labeled.height > 0
