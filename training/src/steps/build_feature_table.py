from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_TOPK_COLS: tuple[str, ...] = (
    "anchor_product_id",
    "candidate_product_id",
)

REQUIRED_BASKET_COLS: tuple[str, ...] = (
    "kiosk_id",
    "products",
)

REQUIRED_QUERY_COLS: tuple[str, ...] = (
    "kiosk_id",
    "anchor_product_id",
)


def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


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

    _ensure_columns(topk_candidates, REQUIRED_TOPK_COLS)
    if queries is None:
        _ensure_columns(baskets, REQUIRED_BASKET_COLS)
    else:
        _ensure_columns(queries, REQUIRED_QUERY_COLS)

    LOGGER.info("Building feature table")

    # --------------------------------------
    # 1. Define (kiosk, anchor_product) pairs
    # --------------------------------------
    if queries is not None:
        kiosk_anchors = (
            queries
            .select(["kiosk_id", "anchor_product_id"])
            .unique()
        )
        LOGGER.info("Using explicit queries")
    else:
        kiosk_anchors = (
            baskets
            .select(["kiosk_id", "products"])
            .explode("products")
            .rename({"products": "anchor_product_id"})
            .unique()
        )
        LOGGER.info("Using kiosk-anchor pairs from baskets")

    LOGGER.info("Kiosk-anchor pairs: %s", kiosk_anchors.shape)

    # --------------------------------------
    # 2. Join with top-K candidates (GLOBAL)
    # --------------------------------------
    join_keys = (
        ["kiosk_id", "anchor_product_id"]
        if "kiosk_id" in topk_candidates.columns
        else ["anchor_product_id"]
    )
    feature_table = kiosk_anchors.join(
        topk_candidates,
        on=join_keys,
        how="inner",
    )

    LOGGER.info("Feature table shape: %s", feature_table.shape)

    return feature_table
