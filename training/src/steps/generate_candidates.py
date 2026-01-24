from __future__ import annotations
import polars as pl


def generate_candidates(
    baskets: pl.DataFrame,
    min_cooc: int = 2,
) -> pl.DataFrame:
    """
    Generate candidate products using co-occurrence + association metrics.

    Output schema:
    - anchor_product_id
    - candidate_product_id
    - cooc_count
    - anchor_count
    - candidate_count
    - support
    - confidence
    - lift
    """

    print(f"[INFO] Generating candidates from baskets: {baskets.shape}")

    # --------------------------------------------------
    # 0. Pre-compute basic stats
    # --------------------------------------------------

    n_baskets = baskets.height

    # How often each product appears in baskets
    product_counts = (
        baskets
        .explode("products")
        .group_by("products")
        .agg(pl.count().alias("product_count"))
        .rename({"products": "product_id"})
    )

    # --------------------------------------------------
    # 1. Build co-occurring product pairs
    # --------------------------------------------------

    exploded = baskets.explode("products")

    pairs = exploded.join(
        exploded,
        on=["kiosk_id", "order_id"],
        how="inner",
        suffix="_candidate"
    ).filter(
        pl.col("products") != pl.col("products_candidate")
    )

    # --------------------------------------------------
    # 2. Co-occurrence counts
    # --------------------------------------------------

    candidates = (
        pairs
        .group_by([
            pl.col("products").alias("anchor_product_id"),
            pl.col("products_candidate").alias("candidate_product_id"),
        ])
        .agg(pl.count().alias("cooc_count"))
        .filter(pl.col("cooc_count") >= min_cooc)
    )

    # --------------------------------------------------
    # 3. Join product frequencies
    # --------------------------------------------------

    candidates = candidates.join(
        product_counts.rename({
            "product_id": "anchor_product_id",
            "product_count": "anchor_count",
        }),
        on="anchor_product_id",
        how="left"
    )

    candidates = candidates.join(
        product_counts.rename({
            "product_id": "candidate_product_id",
            "product_count": "candidate_count",
        }),
        on="candidate_product_id",
        how="left"
    )

    # --------------------------------------------------
    # 4. Association metrics
    # --------------------------------------------------

    candidates = candidates.with_columns([
        # support(A,B)
        (pl.col("cooc_count") / pl.lit(n_baskets)).alias("support"),

        # confidence(A -> B)
        (pl.col("cooc_count") / pl.col("anchor_count")).alias("confidence"),

        # lift(A,B)
        (
            pl.col("cooc_count") * pl.lit(n_baskets)
            / (pl.col("anchor_count") * pl.col("candidate_count"))
        ).alias("lift"),
    ])

    # --------------------------------------------------
    # 5. Sort for readability / baseline usage
    # --------------------------------------------------

    candidates = candidates.sort(
        ["anchor_product_id", "lift", "cooc_count"],
        descending=[False, True, True],
    )

    print(f"[INFO] Generated candidates with metrics: {candidates.shape}")

    return candidates


