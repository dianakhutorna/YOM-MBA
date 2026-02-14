from __future__ import annotations

import polars as pl


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


def test_build_feature_table_kiosk_specific_join():
    from training.src.steps.build_feature_table import build_feature_table

    baskets = pl.DataFrame(
        {
            "kiosk_id": ["k1", "k2"],
            "order_id": ["o1", "o2"],
            "products": [["a", "x"], ["a", "y"]],
        }
    )
    topk_candidates = pl.DataFrame(
        {
            "kiosk_id": ["k1", "k2"],
            "anchor_product_id": ["a", "a"],
            "candidate_product_id": ["c1", "c2"],
            "lift": [2.0, 3.0],
            "cooc_count": [5, 6],
        }
    )
    queries = pl.DataFrame(
        {
            "kiosk_id": ["k1", "k2"],
            "anchor_product_id": ["a", "a"],
        }
    )

    features = build_feature_table(
        baskets=baskets,
        topk_candidates=topk_candidates,
        queries=queries,
    )
    result_pairs = set(
        zip(
            features["kiosk_id"].to_list(),
            features["candidate_product_id"].to_list(),
        )
    )
    assert result_pairs == {("k1", "c1"), ("k2", "c2")}
