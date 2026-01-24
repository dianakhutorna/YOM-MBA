from __future__ import annotations
import polars as pl
import numpy as np


def hitrate_at_k_by_score(
    df: pl.DataFrame,
    k: int = 20,
    score_col: str = "score",
) -> float:
    """
    Recall@K (hit-rate style) computed after ranking by `score_col`.

    Only evaluates (kiosk, anchor) groups
    that have at least one positive label in test.
    """

    # 🔴 NEW: keep only anchors with at least one positive label
    valid_groups = (
        df.filter(pl.col("label") == 1)
        .select(["kiosk_id", "anchor_product_id"])
        .unique()
    )

    df = df.join(
        valid_groups,
        on=["kiosk_id", "anchor_product_id"],
        how="inner",
    )

    # --------------------------------------------------

    topk = (
        df.sort(
            ["kiosk_id", "anchor_product_id", score_col],
            descending=[False, False, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k)
    )

    hitrate_df = (
        topk.group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.max("label").alias("hit"))
    )

    hitrate = hitrate_df.select(pl.mean("hit")).item()

    if hitrate is None:
        return 0.0
    return float(hitrate)

def ndcg_at_k_by_score(
    df: pl.DataFrame,
    k: int = 20,
    score_col: str = "score",
) -> float:
    """
    Compute mean NDCG@K over (kiosk_id, anchor_product_id) groups.
    Assumes binary relevance (label ∈ {0,1}).
    """

    ndcgs = []

    grouped = df.group_by(["kiosk_id", "anchor_product_id"])

    for _, group in grouped:
        # sort by predicted score
        group = group.sort(score_col, descending=True).head(k)

        rel = group["label"].to_numpy()

        if rel.sum() == 0:
            continue  # skip groups with no positives

        # DCG
        discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
        dcg = np.sum(rel * discounts)

        # IDCG
        ideal_rel = np.sort(rel)[::-1]
        idcg = np.sum(ideal_rel * discounts)

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    if len(ndcgs) == 0:
        return 0.0

    return float(np.mean(ndcgs))