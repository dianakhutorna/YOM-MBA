from __future__ import annotations
import polars as pl


def recall_at_k(
    labeled_features: pl.DataFrame,
    k: int = 20,
) -> float:
    """
    Compute Recall@K for anchor-based recommendation.
    """

    print(f"[INFO] Computing Recall@{k}")

    # --------------------------------------
    # 1. Take top-K per (kiosk, anchor)
    # --------------------------------------

    topk = (
        labeled_features
        .sort(
            ["kiosk_id", "anchor_product_id", "lift"],
            descending=[False, False, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k)
    )

    # --------------------------------------
    # 2. Recall: hit if any label == 1
    # --------------------------------------

    recall_df = (
        topk
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.max("label").alias("hit"))
    )

    recall = recall_df.select(pl.mean("hit")).item()

    print(f"[RESULT] Recall@{k}: {recall:.4f}")

    return recall
