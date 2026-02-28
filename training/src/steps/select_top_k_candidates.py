from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_COLS: tuple[str, ...] = (
    "anchor_product_id",
    "candidate_product_id",
    "lift",
    "cooc_count",
)


def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def select_top_k_candidates(
    candidates: pl.DataFrame,
    k: int = 50,
    min_lift: float = 1.0,
) -> pl.DataFrame:
    """
    Select top-K candidate products per anchor.

    Output schema:
    - anchor_product_id
    - candidate_product_id
    - cooc_count
    - confidence
    - lift
    """

    _ensure_columns(candidates, REQUIRED_COLS)
    LOGGER.info("Selecting top-%s candidates per anchor", k)

    top_k = (
        candidates
        .filter(pl.col("lift") >= min_lift)
        .sort(
            ["anchor_product_id", "lift", "cooc_count"],
            descending=[False, True, True],
        )
        .group_by("anchor_product_id")
        .head(k)
    )

    # Drop MBA-internal columns that are only needed for sorting/filtering.
    # Keep only key columns + cooc_cosine_sim (used as a model feature).
    _KEEP_COLS = {"anchor_product_id", "candidate_product_id", "cooc_cosine_sim"}
    drop = [c for c in top_k.columns if c not in _KEEP_COLS]
    if drop:
        top_k = top_k.drop(drop)

    LOGGER.info("Top-K candidates shape: %s (columns: %s)", top_k.shape, top_k.columns)

    # Anchor-level statistics
    cands_per_anchor = (
        top_k.group_by("anchor_product_id")
        .agg(pl.len().alias("n_cands"))
    )
    n_anchors = cands_per_anchor.height
    stats = cands_per_anchor.select(
        pl.col("n_cands").min().alias("min"),
        pl.col("n_cands").mean().alias("mean"),
        pl.col("n_cands").median().alias("median"),
        pl.col("n_cands").max().alias("max"),
    ).row(0)
    n_full = int(cands_per_anchor.filter(pl.col("n_cands") >= k).height)
    LOGGER.info(
        "Anchors: %d | candidates per anchor — min: %d, mean: %.1f, "
        "median: %.0f, max: %d | with >=%d candidates: %d (%.1f%%)",
        n_anchors, *stats, k, n_full,
        100.0 * n_full / n_anchors if n_anchors else 0.0,
    )

    return top_k
