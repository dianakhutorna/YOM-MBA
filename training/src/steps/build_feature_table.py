from __future__ import annotations
import polars as pl


def build_feature_table(
    baskets: pl.DataFrame,
    topk_candidates: pl.DataFrame,
    queries: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Build feature table for (kiosk, anchor, candidate).

    Parameters
    ----------
    baskets : pl.DataFrame
        Train baskets (used only to define default kiosk-anchor pairs).
        Schema: [kiosk_id, order_id, products]

    topk_candidates : pl.DataFrame
        Output of select_top_k_candidates().
        Must contain: [anchor_product_id, candidate_product_id, cooc_count, confidence, lift, ...]

    queries : pl.DataFrame | None
        Optional explicit set of (kiosk_id, anchor_product_id) queries.
        If provided, MUST have columns:
            - kiosk_id
            - anchor_product_id

        If None, kiosk-anchor pairs are derived from baskets (BACKWARD COMPATIBLE).

    Returns
    -------
    pl.DataFrame
        Feature table with one row per (kiosk, anchor, candidate).
    """

    print("[INFO] Building feature table")

    # --------------------------------------
    # 1. Define (kiosk, anchor_product) pairs
    # --------------------------------------
    if queries is not None:
        kiosk_anchors = (
            queries
            .select(["kiosk_id", "anchor_product_id"])
            .unique()
        )
        print("[INFO] Using explicit queries")
    else:
        kiosk_anchors = (
            baskets
            .select(["kiosk_id", "products"])
            .explode("products")
            .rename({"products": "anchor_product_id"})
            .unique()
        )
        print("[INFO] Using kiosk-anchor pairs from baskets")

    print(f"[INFO] Kiosk-anchor pairs: {kiosk_anchors.shape}")

    # --------------------------------------
    # 2. Join with top-K candidates (GLOBAL)
    # --------------------------------------
    feature_table = kiosk_anchors.join(
        topk_candidates,
        on="anchor_product_id",
        how="inner",
    )

    print(f"[INFO] Feature table shape: {feature_table.shape}")

    return feature_table

