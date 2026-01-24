from __future__ import annotations
import polars as pl


def build_feature_table(
    baskets: pl.DataFrame,
    topk_candidates: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build feature table for (kiosk, anchor, candidate).

    Output schema:
    - kiosk_id
    - anchor_product_id
    - candidate_product_id
    - cooc_count
    - confidence
    - lift
    - anchor_count
    - candidate_count
    """

    print("[INFO] Building feature table")

    # --------------------------------------
    # 1. Extract (kiosk, anchor_product) pairs
    # --------------------------------------

    kiosk_anchors = (
        baskets
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .unique()
    )

    print(f"[INFO] Kiosk-anchor pairs: {kiosk_anchors.shape}")

    # --------------------------------------
    # 2. Join with top-K candidates
    # --------------------------------------

    feature_table = kiosk_anchors.join(
        topk_candidates,
        on="anchor_product_id",
        how="inner",
    )

    print(f"[INFO] Feature table shape: {feature_table.shape}")

    return feature_table
