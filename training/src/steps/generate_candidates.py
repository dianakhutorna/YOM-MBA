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


def _empty_candidates_frame() -> pl.DataFrame:
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


def _pair_products_batched(
    exploded: pl.DataFrame,
    *,
    batch_size: int = 200_000,
    min_cooc: int = 2,
) -> pl.DataFrame:
    """
    Generate anchor-candidate pair co-occurrence counts via batched self-join.
    """
    order_ids = exploded.select("order_id").unique().to_series()
    n_orders = len(order_ids)
    if n_orders == 0:
        return pl.DataFrame(
            schema={
                "anchor_product_id": pl.Utf8,
                "candidate_product_id": pl.Utf8,
                "cooc_count": pl.Int64,
            }
        )

    n_batches = max(1, (n_orders + batch_size - 1) // batch_size)
    LOGGER.info(
        "Pair generation: %s orders in %d batches of ~%s",
        f"{n_orders:,}", n_batches, f"{batch_size:,}",
    )

    parts: list[pl.DataFrame] = []
    for i in range(0, n_orders, batch_size):
        batch_ids = order_ids.slice(i, min(batch_size, n_orders - i))
        batch_exploded = exploded.filter(pl.col("order_id").is_in(batch_ids))

        pairs = (
            batch_exploded
            .join(batch_exploded, on="order_id", how="inner")
            .rename({
                "product_id": "anchor_product_id",
                "product_id_right": "candidate_product_id",
            })
            .filter(pl.col("anchor_product_id") != pl.col("candidate_product_id"))
        )

        batch_cooc = (
            pairs
            .group_by(["anchor_product_id", "candidate_product_id"])
            .agg(pl.len().alias("cooc_count"))
        )
        parts.append(batch_cooc)
        del pairs, batch_exploded, batch_cooc

        if (i // batch_size + 1) % 10 == 0:
            LOGGER.info("  batch %d/%d done", i // batch_size + 1, n_batches)

    if not parts:
        return pl.DataFrame(
            schema={
                "anchor_product_id": pl.Utf8,
                "candidate_product_id": pl.Utf8,
                "cooc_count": pl.Int64,
            }
        )

    LOGGER.info("Merging co-occurrence counts from %d batches...", len(parts))
    all_cooc = pl.concat(parts, how="vertical_relaxed")
    del parts

    merged = (
        all_cooc
        .group_by(["anchor_product_id", "candidate_product_id"])
        .agg(pl.col("cooc_count").sum().alias("cooc_count"))
        .filter(pl.col("cooc_count") >= min_cooc)
    )
    del all_cooc
    return merged


def generate_candidates(
    baskets: pl.DataFrame,
    min_cooc: int = 2,
    pair_batch_size: int = 200_000,
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
    LOGGER.info(
        "Exploded: %s rows, %s unique products",
        f"{exploded.height:,}",
        f"{exploded.select(pl.col('product_id').n_unique()).item():,}",
    )

    # ------------------------------------------------------------------
    # 2. Product frequencies (global)
    # ------------------------------------------------------------------
    product_counts = _product_counts(exploded)

    total_baskets = baskets.height

    # ------------------------------------------------------------------
    # 3. Pre-filter products that cannot reach min_cooc
    # ------------------------------------------------------------------
    frequent_products = (
        product_counts
        .filter(pl.col("product_count") >= min_cooc)
        .select("product_id")
    )
    exploded_before = exploded.height
    exploded = exploded.join(frequent_products, on="product_id", how="inner")
    LOGGER.info(
        "Pre-filter: kept %s / %s exploded rows (products with count >= %d)",
        f"{exploded.height:,}", f"{exploded_before:,}", min_cooc,
    )

    if exploded.is_empty():
        return _empty_candidates_frame()

    # ------------------------------------------------------------------
    # 4. Anchor–candidate pairs via batched self-join
    # ------------------------------------------------------------------
    pairs = _pair_products_batched(
        exploded,
        batch_size=pair_batch_size,
        min_cooc=min_cooc,
    )
    del exploded

    if pairs.is_empty():
        return _empty_candidates_frame()

    # ------------------------------------------------------------------
    # 5. Co-occurrence counts (aggregate across batches)
    # ------------------------------------------------------------------
    cooc = (
        pairs
        .group_by(["anchor_product_id", "candidate_product_id"])
        .agg(pl.col("cooc_count").sum().alias("cooc_count"))
        .filter(pl.col("cooc_count") >= min_cooc)
    )
    del pairs

    # ------------------------------------------------------------------
    # 6. Join product frequencies
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
    # 7. MBA metrics + cosine over basket-incidence vectors
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

