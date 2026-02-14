from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates

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


def _anchors_from_baskets(baskets: pl.DataFrame) -> pl.DataFrame:
    return (
        baskets
        .select(["products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .unique()
    )


def _popularity_counts(baskets: pl.DataFrame) -> pl.DataFrame:
    exploded = (
        baskets
        .select(["order_id", "products"])
        .explode("products")
        .rename({"products": "product_id"})
    )
    return (
        exploded
        .group_by("product_id")
        .len()
        .rename({"len": "product_count"})
        .with_columns(pl.col("product_count").cast(pl.Int64))
    )


def generate_candidates_hybrid(
    baskets: pl.DataFrame,
    *,
    products: pl.DataFrame | None,
    min_cooc: int,
    min_lift: float,
    top_k: int,
    pop_top_k_global: int = 50,
    pop_top_k_category: int = 50,
) -> pl.DataFrame:
    """
    Hybrid candidate generation:
    1) Co-occurrence (MBA) candidates with min_lift.
    2) Popularity-based fill:
       - global top-N
       - same-category top-N (if products provided with category)

    Returns top_k candidates per anchor with MBA features preserved,
    popularity candidates filled with zeros for MBA metrics.
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

    LOGGER.info(
        "Generating hybrid candidates: min_cooc=%s min_lift=%s top_k=%s pop_global=%s pop_category=%s",
        min_cooc,
        min_lift,
        top_k,
        pop_top_k_global,
        pop_top_k_category,
    )

    # MBA candidates
    mba = generate_candidates(baskets, min_cooc=min_cooc)
    mba_top = select_top_k_candidates(mba, k=top_k, min_lift=min_lift)

    anchors = _anchors_from_baskets(baskets)
    pop_counts = _popularity_counts(baskets)

    # Global popularity candidates
    global_pop = (
        pop_counts
        .sort("product_count", descending=True)
        .head(pop_top_k_global)
        .select(
            pl.col("product_id").alias("candidate_product_id"),
            pl.col("product_count").cast(pl.Float64).alias("pop_score"),
        )
    )
    global_candidates = anchors.join(global_pop, how="cross")
    global_candidates = global_candidates.filter(
        pl.col("anchor_product_id") != pl.col("candidate_product_id")
    ).with_columns(pl.lit("global_pop").alias("source"))

    # Category popularity candidates (optional)
    category_candidates = pl.DataFrame()
    if products is not None and "category" in products.columns:
        prod_cat = (
            products
            .select(
                pl.col("productid").alias("product_id"),
                pl.col("category"),
            )
            .unique(subset=["product_id"])
        )
        anchors_with_cat = anchors.join(
            prod_cat.rename({"product_id": "anchor_product_id"}),
            on="anchor_product_id",
            how="left",
        )
        pop_with_cat = pop_counts.join(prod_cat, on="product_id", how="left")
        top_by_cat = (
            pop_with_cat
            .sort(["category", "product_count"], descending=[False, True])
            .group_by("category")
            .head(pop_top_k_category)
            .select(
                "category",
                pl.col("product_id").alias("candidate_product_id"),
                pl.col("product_count").cast(pl.Float64).alias("pop_score"),
            )
        )
        category_candidates = (
            anchors_with_cat
            .join(top_by_cat, on="category", how="inner")
            .filter(pl.col("anchor_product_id") != pl.col("candidate_product_id"))
            .select(["anchor_product_id", "candidate_product_id", "pop_score"])
            .with_columns(pl.lit("category_pop").alias("source"))
        )

    # Normalize popularity candidates to MBA schema
    def _to_mba_schema(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        return (
            df
            .with_columns(
                [
                    pl.lit(0).cast(pl.Int64).alias("cooc_count"),
                    pl.lit(0).cast(pl.Int64).alias("anchor_count"),
                    pl.lit(0).cast(pl.Int64).alias("candidate_count"),
                    pl.lit(0.0).cast(pl.Float64).alias("support"),
                    pl.lit(0.0).cast(pl.Float64).alias("confidence"),
                    pl.lit(0.0).cast(pl.Float64).alias("lift"),
                    pl.lit(0.0).cast(pl.Float64).alias("cooc_cosine_sim"),
                ]
            )
            .select(
                [
                    "anchor_product_id",
                    "candidate_product_id",
                    "cooc_count",
                    "anchor_count",
                    "candidate_count",
                    "support",
                    "confidence",
                    "lift",
                    "cooc_cosine_sim",
                    "pop_score",
                    "source",
                ]
            )
        )

    mba_top = mba_top.with_columns(
        [
            pl.col("cooc_count").cast(pl.Int64),
            pl.col("anchor_count").cast(pl.Int64),
            pl.col("candidate_count").cast(pl.Int64),
            pl.col("support").cast(pl.Float64),
            pl.col("confidence").cast(pl.Float64),
            pl.col("lift").cast(pl.Float64),
            pl.col("cooc_cosine_sim").cast(pl.Float64),
            pl.lit(None).cast(pl.Float64).alias("pop_score"),
            pl.lit("mba").alias("source"),
        ]
    )
    global_candidates = _to_mba_schema(global_candidates)
    category_candidates = _to_mba_schema(category_candidates)

    combined = pl.concat([mba_top, category_candidates, global_candidates], how="vertical")
    combined = combined.unique(subset=["anchor_product_id", "candidate_product_id"])

    # Rank candidates: prefer MBA, then category pop, then global pop
    combined = combined.with_columns(
        pl.when(pl.col("source") == "mba")
        .then(1_000_000 + pl.col("lift") * 1_000 + pl.col("cooc_count"))
        .when(pl.col("source") == "category_pop")
        .then(100_000 + pl.col("pop_score").fill_null(0))
        .otherwise(pl.col("pop_score").fill_null(0))
        .alias("_rank_score")
    )

    top_k = (
        combined
        .sort(["anchor_product_id", "_rank_score"], descending=[False, True])
        .group_by("anchor_product_id")
        .head(top_k)
        .drop("_rank_score")
    )

    LOGGER.info("Hybrid Top-K candidates shape: %s", top_k.shape)
    return top_k
