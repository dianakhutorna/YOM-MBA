from __future__ import annotations
import polars as pl


def recall_at_k_by_score(
    df: pl.DataFrame,
    k: int = 20,
    score_col: str = "score",
) -> float:
    """
    Recall@K (hit-rate style) computed after ranking by `score_col`.

    For each (kiosk_id, anchor_product_id) group:
    - take top K by score
    - hit = 1 if any label == 1 in topK
    - recall = mean(hit)
    """

    topk = (
        df.sort(
            ["kiosk_id", "anchor_product_id", score_col],
            descending=[False, False, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k)
    )

    recall_df = (
        topk.group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.max("label").alias("hit"))
    )

    recall = recall_df.select(pl.mean("hit")).item()

    # safe print
    if recall is None:
        return 0.0
    return float(recall)
