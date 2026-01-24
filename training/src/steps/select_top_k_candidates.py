from __future__ import annotations
import polars as pl


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

    print(f"[INFO] Selecting top-{k} candidates per anchor")

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

    print(f"[INFO] Top-K candidates shape: {top_k.shape}")

    return top_k
