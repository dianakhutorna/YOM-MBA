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
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def build_feature_table(
    baskets: pl.DataFrame,
    topk_candidates: pl.DataFrame,
    queries: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Build feature table for (kiosk, anchor, candidate).
    """
    _ensure_columns(topk_candidates, REQUIRED_TOPK_COLS)
    if queries is None:
        _ensure_columns(baskets, REQUIRED_BASKET_COLS)
    else:
        _ensure_columns(queries, REQUIRED_QUERY_COLS)

    LOGGER.info("Build feature table: start")

    # 1) Define (kiosk, anchor) queries
    if queries is not None:
        kiosk_anchors = queries.select(["kiosk_id", "anchor_product_id"]).unique()
        LOGGER.info("Build feature table: using explicit queries (rows=%s)", kiosk_anchors.height)
    else:
        kiosk_anchors = (
            baskets
            .select(["kiosk_id", "products"])
            .explode("products")
            .rename({"products": "anchor_product_id"})
            .unique()
        )
        LOGGER.info("Build feature table: using kiosk-anchor pairs from baskets (rows=%s)", kiosk_anchors.height)

    # 2) Join with candidates
    join_keys = ["kiosk_id", "anchor_product_id"] if "kiosk_id" in topk_candidates.columns else ["anchor_product_id"]
    LOGGER.info("Build feature table: join keys=%s", join_keys)

    feature_table = kiosk_anchors.join(topk_candidates, on=join_keys, how="inner")

    LOGGER.info(
        "Build feature table: done (rows=%s cols=%s)",
        feature_table.height,
        feature_table.width,
    )
    return feature_table
