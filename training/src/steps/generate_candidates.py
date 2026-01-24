from __future__ import annotations
import polars as pl


def generate_candidates(
    baskets: pl.DataFrame,
    min_cooc: int = 2,
) -> pl.DataFrame:
    """
    Generate anchor-candidate pairs from baskets and compute
    co-occurrence-based metrics + cosine similarity.

    baskets columns:
    - kiosk_id
    - order_id
    - products (List[str])
    """

    # ------------------------------------------------------------------
    # 1. Explode baskets → (order_id, product_id)
    # ------------------------------------------------------------------
    exploded = (
        baskets
        .select(["order_id", "products"])
        .explode("products")
        .rename({"products": "product_id"})
    )

    # ------------------------------------------------------------------
    # 2. Product frequencies (global)
    # ------------------------------------------------------------------
    product_counts = (
        exploded
        .group_by("product_id")
        .agg(pl.count().alias("product_count"))
    )

    total_baskets = baskets.height

    # ------------------------------------------------------------------
    # 3. Anchor–candidate pairs INSIDE THE SAME BASKET
    # ------------------------------------------------------------------
    pairs = (
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

    # ------------------------------------------------------------------
    # 4. Co-occurrence counts
    # ------------------------------------------------------------------
    cooc = (
        pairs
        .group_by(["anchor_product_id", "candidate_product_id"])
        .agg(pl.count().alias("cooc_count"))
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
    # 6. MBA metrics + cosine similarity
    # ------------------------------------------------------------------
    cooc = cooc.with_columns([
        # support
        (pl.col("cooc_count") / total_baskets).alias("support"),

        # confidence P(candidate | anchor)
        (pl.col("cooc_count") / pl.col("anchor_count")).alias("confidence"),

        # lift
        (
            (pl.col("cooc_count") / total_baskets) /
            (
                (pl.col("anchor_count") / total_baskets) *
                (pl.col("candidate_count") / total_baskets)
            )
        ).alias("lift"),

        # cosine similarity
        pl.when(
            (pl.col("anchor_count") > 0) & (pl.col("candidate_count") > 0)
        )
        .then(
            pl.col("cooc_count") /
            (pl.col("anchor_count") * pl.col("candidate_count")).sqrt()
        )
        .otherwise(0.0)
        .alias("cosine_sim"),
    ])

    return cooc




