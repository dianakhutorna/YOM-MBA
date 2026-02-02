from __future__ import annotations

def test_offline_pipeline_smoke(cleaned_orders_df, baskets_df, topk_df):
    from training.src.steps.build_feature_table import build_feature_table
    from training.src.steps.build_labels import build_labels
    from training.src.steps.split_orders import split_orders_by_time

    features = build_feature_table(baskets_df, topk_df)
    _train, _val, test_orders = split_orders_by_time(
        cleaned_orders_df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    labeled = build_labels(features, test_orders)

    assert features.height > 0
    assert labeled.height > 0
