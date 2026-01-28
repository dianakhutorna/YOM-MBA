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

def positives_at_k_by_score(
    df: pl.DataFrame,
    k: int = 20,
    score_col: str = "score",
) -> float:
    """
    Average number of positive items in top-K recommendations.
    """

    # keep only groups with at least one positive label
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

    # top-K per group
    topk = (
        df.sort(
            ["kiosk_id", "anchor_product_id", score_col],
            descending=[False, False, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k)
    )

    # count positives per group
    per_group = (
        topk.group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.sum("label").alias("positives_at_k"))
    )

    # ⬇️⬇️⬇️ FIX ⬇️⬇️⬇️
    mean_val = per_group.select(pl.mean("positives_at_k")).item()

    if mean_val is None:
        return 0.0

    return float(mean_val)

def quantity_captured_at_k_by_score(
    df: pl.DataFrame,
    test_orders: pl.DataFrame,
    k: int = 20,
    score_col: str = "score",
) -> float:
    """
    Average quantity captured by top-K recommendations.

    For each (kiosk_id, anchor_product_id):
    - take top-K candidates by score
    - find test orders where anchor_product_id was purchased
    - sum quantity of recommended candidates within those orders
    - average over groups with at least one positive label
    """

    # --------------------------------------------------
    # 1. Valid (kiosk, anchor) groups
    # --------------------------------------------------
    valid_groups = (
        df.filter(pl.col("label") == 1)
        .select(["kiosk_id", "anchor_product_id"])
        .unique()
    )

    # --------------------------------------------------
    # 2. Top-K recommendations
    # --------------------------------------------------
    topk = (
        df.join(
            valid_groups,
            on=["kiosk_id", "anchor_product_id"],
            how="inner",
        )
        .sort(
            ["kiosk_id", "anchor_product_id", score_col],
            descending=[False, False, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k)
        .select(["kiosk_id", "anchor_product_id", "candidate_product_id"])
    )

    # --------------------------------------------------
    # 3. Orders where anchor was bought (NO anchor_product_id in test_orders!)
    # --------------------------------------------------
    anchor_orders = (
        test_orders
        .join(
            valid_groups,
            left_on=["kiosk_id", "product_id"],
            right_on=["kiosk_id", "anchor_product_id"],
            how="inner",
        )
        .select(["order_id", "kiosk_id"])
        .unique()
    )

    # --------------------------------------------------
    # 4. Quantities of recommended candidates
    # --------------------------------------------------
    quantities = (
        test_orders
        .join(
            anchor_orders,
            on=["order_id", "kiosk_id"],
            how="inner",
        )
        .join(
            topk,
            on="kiosk_id",
            how="inner",
        )
        .filter(
            (pl.col("product_id") == pl.col("candidate_product_id"))
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.sum("quantity").alias("quantity_captured"))
    )

    # --------------------------------------------------
    # 5. Average quantity
    # --------------------------------------------------
    return float(
        quantities
        .select(pl.mean("quantity_captured"))
        .item()
    )

def quantity_captured_per_order_at_k_by_score(
    df: pl.DataFrame,
    test_orders: pl.DataFrame,
    k: int = 20,
    score_col: str = "score",
) -> float:
    """
    Average quantity captured per anchor-related order.

    For each (kiosk_id, anchor_product_id):
    - take top-K candidates by score
    - find test orders where anchor_product_id was purchased
    - sum quantity of recommended candidates within those orders
    - divide by number of anchor orders
    - average over groups with at least one positive label
    """

    # --------------------------------------------------
    # 1. Valid (kiosk, anchor) groups
    # --------------------------------------------------
    valid_groups = (
        df.filter(pl.col("label") == 1)
        .select(["kiosk_id", "anchor_product_id"])
        .unique()
    )

    # --------------------------------------------------
    # 2. Top-K recommendations
    # --------------------------------------------------
    topk = (
        df.join(
            valid_groups,
            on=["kiosk_id", "anchor_product_id"],
            how="inner",
        )
        .sort(
            ["kiosk_id", "anchor_product_id", score_col],
            descending=[False, False, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k)
        .select(["kiosk_id", "anchor_product_id", "candidate_product_id"])
    )

    # --------------------------------------------------
    # 3. Orders where anchor was purchased
    # --------------------------------------------------
    anchor_orders = (
        test_orders
        .join(
            valid_groups,
            left_on=["kiosk_id", "product_id"],
            right_on=["kiosk_id", "anchor_product_id"],
            how="inner",
        )
        .select(["order_id", "kiosk_id", "anchor_product_id"])
        .unique()
    )

    # number of anchor orders per group
    order_counts = (
        anchor_orders
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.count().alias("n_anchor_orders"))
    )

    # --------------------------------------------------
    # 4. Quantities of recommended candidates
    # --------------------------------------------------
    quantities = (
        test_orders
        .join(
            anchor_orders,
            on=["order_id", "kiosk_id"],
            how="inner",
        )
        .join(
            topk,
            on=["kiosk_id", "anchor_product_id"],
            how="inner",
        )
        .filter(pl.col("product_id") == pl.col("candidate_product_id"))
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.sum("quantity").alias("quantity_captured"))
    )

    # --------------------------------------------------
    # 5. Normalize per order
    # --------------------------------------------------
    per_order = (
        quantities
        .join(
            order_counts,
            on=["kiosk_id", "anchor_product_id"],
            how="inner",
        )
        .with_columns(
            (pl.col("quantity_captured") / pl.col("n_anchor_orders"))
            .alias("quantity_per_order")
        )
    )

    # --------------------------------------------------
    # 6. Average over groups
    # --------------------------------------------------
    return float(
        per_order
        .select(pl.mean("quantity_per_order"))
        .item()
    )

def recall_at_k_by_score(
    df: pl.DataFrame,
    k: int = 20,
    score_col: str = "score",
) -> float:
    """
    True Recall@K for anchor-based recommendation.

    Recall@K = (# relevant items in top-K) / (total # relevant items)
    averaged over (kiosk_id, anchor_product_id).
    """

    # groups with at least one positive
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

    # total relevant items per group
    total_relevant = (
        df.filter(pl.col("label") == 1)
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.count().alias("n_relevant"))
    )

    # top-K per group
    topk = (
        df.sort(
            ["kiosk_id", "anchor_product_id", score_col],
            descending=[False, False, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k)
    )

    # relevant items in top-K
    relevant_in_topk = (
        topk.filter(pl.col("label") == 1)
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.count().alias("n_hit"))
    )

    recall_df = (
        total_relevant
        .join(
            relevant_in_topk,
            on=["kiosk_id", "anchor_product_id"],
            how="left",
        )
        .with_columns(pl.col("n_hit").fill_null(0))
        .with_columns(
            (pl.col("n_hit") / pl.col("n_relevant")).alias("recall")
        )
    )

    return float(recall_df.select(pl.mean("recall")).item())

def category_coverage_lift_at_k(
    df: pl.DataFrame,
    k: int = 20,
    score_col: str = "score",
) -> float:
    """
    Category coverage lift at K.
    """

    # top-K per group
    topk = (
        df.sort(
            ["kiosk_id", "anchor_product_id", score_col],
            descending=[False, False, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k)
    )

    # coverage in recommendations
    rec_cov = (
        topk.group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.col("candidate_category").n_unique().alias("rec_cov"))
    )

    # coverage in ground truth
    gt_cov = (
        df.filter(pl.col("label") == 1)
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.col("candidate_category").n_unique().alias("gt_cov"))
    )

    lift = (
        rec_cov
        .join(gt_cov, on=["kiosk_id", "anchor_product_id"])
        .with_columns(
            (pl.col("rec_cov") / pl.col("gt_cov")).alias("coverage_lift")
        )
    )

    return float(lift.select(pl.mean("coverage_lift")).item())
