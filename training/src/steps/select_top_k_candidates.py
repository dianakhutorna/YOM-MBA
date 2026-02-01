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

    LOGGER.info("Top-K candidates shape: %s", top_k.shape)

    return top_k
