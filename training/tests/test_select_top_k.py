from __future__ import annotations

import polars as pl


def test_select_top_k_limit_per_anchor(candidates_df):
    from training.src.steps.select_top_k_candidates import select_top_k_candidates

    k = 10
    topk = select_top_k_candidates(candidates_df, k=k, min_lift=1.5)
    max_count = (
        topk.group_by("anchor_product_id")
        .agg(pl.len().alias("cnt"))
        .select(pl.col("cnt").max())
        .item()
    )
    assert max_count is None or max_count <= k
