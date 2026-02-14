from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_BASKET_COLS: tuple[str, ...] = (
    "order_id",
    "products",
)


def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def _explode_baskets(baskets: pl.DataFrame) -> pl.DataFrame:
    return (
        baskets
        .select(["order_id", "products"])
        .explode("products")
        .rename({"products": "product_id"})
    )


def _product_counts(exploded: pl.DataFrame) -> pl.DataFrame:
    return (
        exploded
        .group_by("product_id")
        .agg(pl.len().alias("product_count"))
    )


def _pair_products(exploded: pl.DataFrame) -> pl.DataFrame:
    return (
        exploded
        .join(
            exploded,
            on="order_id",
            how="inner",
        )
        .rename({
            "product_id": "anchor_product_id",
            "product_id_right": "candidate_product_id",
        })
        .filter(pl.col("anchor_product_id") != pl.col("candidate_product_id"))
    )


def generate_candidates(
    baskets: pl.DataFrame,
    min_cooc: int = 2,
) -> pl.DataFrame:
    """
    Generate anchor-candidate pairs from baskets and compute
    co-occurrence-based metrics + co-occurrence cosine similarity.

    baskets columns:
    - kiosk_id
    - order_id
    - products (List[str])
    """

    _ensure_columns(baskets, REQUIRED_BASKET_COLS)
    if baskets.is_empty():
        return pl.DataFrame(
            schema={
                "anchor_product_id": pl.Utf8,
                "candidate_product_id": pl.Utf8,
                "cooc_count": pl.Int64,
                "anchor_count": pl.Int64,
                "candidate_count": pl.Int64,
                "support": pl.Float64,
                "confidence": pl.Float64,
                "lift": pl.Float64,
                "cooc_cosine_sim": pl.Float64,
            }
        )

    LOGGER.info("Generating candidates from baskets: %s", baskets.shape)

    # ------------------------------------------------------------------
    # 1. Explode baskets → (order_id, product_id)
    # ------------------------------------------------------------------
    exploded = _explode_baskets(baskets)

    # ------------------------------------------------------------------
    # 2. Product frequencies (global)
    # ------------------------------------------------------------------
    product_counts = _product_counts(exploded)

    total_baskets = baskets.height

    # ------------------------------------------------------------------
    # 3. Anchor–candidate pairs INSIDE THE SAME BASKET
    # ------------------------------------------------------------------
    pairs = _pair_products(exploded)

    # ------------------------------------------------------------------
    # 4. Co-occurrence counts
    # ------------------------------------------------------------------
    cooc = (
        pairs
        .group_by(["anchor_product_id", "candidate_product_id"])
        .agg(pl.len().alias("cooc_count"))
        .filter(pl.col("cooc_count") >= min_cooc)
    )

    # ------------------------------------------------------------------
    # 5. Join product frequencies
    # ------------------------------------------------------------------
    cooc = (
        cooc
        .join(
            product_counts.rename({
                "product_id": "anchor_product_id",
                "product_count": "anchor_count",
            }),
            on="anchor_product_id",
            how="left",
        )
        .join(
            product_counts.rename({
                "product_id": "candidate_product_id",
                "product_count": "candidate_count",
            }),
            on="candidate_product_id",
            how="left",
        )
    )

    # ------------------------------------------------------------------
    # 6. MBA metrics + cosine over basket-incidence vectors
    # ------------------------------------------------------------------
    cooc = cooc.with_columns([
        (pl.col("cooc_count") / total_baskets).alias("support"),
        (pl.col("cooc_count") / pl.col("anchor_count")).alias("confidence"),
        (
            (pl.col("cooc_count") / total_baskets) /
            (
                (pl.col("anchor_count") / total_baskets) *
                (pl.col("candidate_count") / total_baskets)
            )
        ).alias("lift"),
        pl.when(
            (pl.col("anchor_count") > 0) & (pl.col("candidate_count") > 0)
        )
        .then(
            pl.col("cooc_count") /
            (pl.col("anchor_count") * pl.col("candidate_count")).sqrt()
        )
        .otherwise(0.0)
        .alias("cooc_cosine_sim"),
    ])

    return cooc

