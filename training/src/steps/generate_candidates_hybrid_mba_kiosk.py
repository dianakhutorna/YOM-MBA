from __future__ import annotations

import logging
from typing import Sequence

import polars as pl

from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates

LOGGER = logging.getLogger(__name__)

REQUIRED_BASKET_COLS: tuple[str, ...] = (
    "kiosk_id",
    "order_id",
    "products",
)


def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def _empty_schema_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "kiosk_id": pl.Utf8,
            "anchor_product_id": pl.Utf8,
            "candidate_product_id": pl.Utf8,
            "cooc_count": pl.Int64,
            "anchor_count": pl.Int64,
            "candidate_count": pl.Int64,
            "support": pl.Float64,
            "confidence": pl.Float64,
            "lift": pl.Float64,
            "cooc_cosine_sim": pl.Float64,
            "source": pl.Utf8,
        }
    )


def _compute_kiosk_mba_for_batch(
    exploded_batch: pl.DataFrame,
    basket_counts_batch: pl.DataFrame,
    *,
    min_cooc: int,
    min_lift: float,
    k_kiosk: int,
) -> pl.DataFrame:
    if exploded_batch.is_empty() or k_kiosk <= 0:
        return _empty_schema_df().drop("source")

    product_counts = (
        exploded_batch
        .group_by(["kiosk_id", "product_id"])
        .agg(pl.len().alias("product_count"))
    )

    pairs = (
        exploded_batch
        .join(exploded_batch, on=["kiosk_id", "order_id"], how="inner")
        .rename(
            {
                "product_id": "anchor_product_id",
                "product_id_right": "candidate_product_id",
            }
        )
        .filter(pl.col("anchor_product_id") != pl.col("candidate_product_id"))
        .group_by(["kiosk_id", "anchor_product_id", "candidate_product_id"])
        .agg(pl.len().alias("cooc_count"))
        .filter(pl.col("cooc_count") >= min_cooc)
    )
    if pairs.is_empty():
        return _empty_schema_df().drop("source")

    cooc = (
        pairs
        .join(
            product_counts.rename(
                {
                    "product_id": "anchor_product_id",
                    "product_count": "anchor_count",
                }
            ),
            on=["kiosk_id", "anchor_product_id"],
            how="left",
        )
        .join(
            product_counts.rename(
                {
                    "product_id": "candidate_product_id",
                    "product_count": "candidate_count",
                }
            ),
            on=["kiosk_id", "candidate_product_id"],
            how="left",
        )
        .join(basket_counts_batch, on="kiosk_id", how="left")
        .with_columns(
            [
                (pl.col("cooc_count") / pl.col("kiosk_basket_count")).alias("support"),
                (pl.col("cooc_count") / pl.col("anchor_count")).alias("confidence"),
                (
                    (pl.col("cooc_count") / pl.col("kiosk_basket_count")) /
                    (
                        (pl.col("anchor_count") / pl.col("kiosk_basket_count")) *
                        (pl.col("candidate_count") / pl.col("kiosk_basket_count"))
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
            ]
        )
        .filter(pl.col("lift") >= min_lift)
        .sort(
            ["kiosk_id", "anchor_product_id", "lift", "cooc_count"],
            descending=[False, False, True, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k_kiosk)
        .with_columns(
            [
                pl.col("cooc_count").cast(pl.Int64),
                pl.col("anchor_count").cast(pl.Int64),
                pl.col("candidate_count").cast(pl.Int64),
                pl.col("support").cast(pl.Float64),
                pl.col("confidence").cast(pl.Float64),
                pl.col("lift").cast(pl.Float64),
                pl.col("cooc_cosine_sim").cast(pl.Float64),
            ]
        )
        .select(
            [
                "kiosk_id",
                "anchor_product_id",
                "candidate_product_id",
                "cooc_count",
                "anchor_count",
                "candidate_count",
                "support",
                "confidence",
                "lift",
                "cooc_cosine_sim",
            ]
        )
    )
    return cooc


def generate_candidates_hybrid_mba_kiosk(
    baskets: pl.DataFrame,
    *,
    min_cooc: int,
    min_lift: float,
    top_k: int,
    kiosk_share: float = 0.5,
    kiosk_batch_size: int = 100,
) -> pl.DataFrame:
    """
    Hybrid generator:
    - kiosk MBA part (personalized)
    - global MBA part (fallback/fill)

    For each (kiosk_id, anchor_product_id), returns up to top_k candidates.
    """

    _ensure_columns(baskets, REQUIRED_BASKET_COLS)
    if baskets.is_empty() or top_k <= 0:
        return _empty_schema_df()

    kiosk_share = float(max(0.0, min(1.0, kiosk_share)))
    k_kiosk = int(round(top_k * kiosk_share))
    k_global_default = top_k - k_kiosk

    exploded = (
        baskets
        .select(["kiosk_id", "order_id", "products"])
        .explode("products")
        .rename({"products": "product_id"})
        .with_columns(
            [
                pl.col("kiosk_id").cast(pl.Utf8),
                pl.col("product_id").cast(pl.Utf8),
            ]
        )
        .unique(subset=["kiosk_id", "order_id", "product_id"])
    )
    if exploded.is_empty():
        return _empty_schema_df()

    kiosk_anchors = (
        exploded
        .select(
            [
                pl.col("kiosk_id"),
                pl.col("product_id").alias("anchor_product_id"),
            ]
        )
        .unique()
    )

    basket_counts_all = (
        baskets
        .with_columns(pl.col("kiosk_id").cast(pl.Utf8))
        .group_by("kiosk_id")
        .agg(pl.len().alias("kiosk_basket_count"))
    )

    global_mba = generate_candidates(baskets, min_cooc=min_cooc)
    global_top = (
        select_top_k_candidates(global_mba, k=top_k, min_lift=min_lift)
        .with_columns(
            [
                pl.col("anchor_product_id").cast(pl.Utf8),
                pl.col("candidate_product_id").cast(pl.Utf8),
                pl.col("cooc_count").cast(pl.Int64),
                pl.col("anchor_count").cast(pl.Int64),
                pl.col("candidate_count").cast(pl.Int64),
                pl.col("support").cast(pl.Float64),
                pl.col("confidence").cast(pl.Float64),
                pl.col("lift").cast(pl.Float64),
                pl.col("cooc_cosine_sim").cast(pl.Float64),
            ]
        )
    )

    kiosks = basket_counts_all.select("kiosk_id").to_series().to_list()
    kiosk_batch_size = max(1, int(kiosk_batch_size))
    kiosk_parts: list[pl.DataFrame] = []

    for start in range(0, len(kiosks), kiosk_batch_size):
        chunk = kiosks[start:start + kiosk_batch_size]
        exploded_batch = exploded.filter(pl.col("kiosk_id").is_in(chunk))
        basket_counts_batch = basket_counts_all.filter(pl.col("kiosk_id").is_in(chunk))
        part = _compute_kiosk_mba_for_batch(
            exploded_batch,
            basket_counts_batch,
            min_cooc=min_cooc,
            min_lift=min_lift,
            k_kiosk=k_kiosk,
        )
        if not part.is_empty():
            kiosk_parts.append(part)

    if kiosk_parts:
        kiosk_top = pl.concat(kiosk_parts, how="vertical").with_columns(pl.lit("kiosk_mba").alias("source"))
    else:
        kiosk_top = _empty_schema_df()

    kiosk_counts = (
        kiosk_top
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.len().alias("kiosk_count"))
    )
    needed_global = (
        kiosk_anchors
        .join(kiosk_counts, on=["kiosk_id", "anchor_product_id"], how="left")
        .with_columns(
            (
                pl.lit(top_k) - pl.col("kiosk_count").fill_null(0)
            )
            .clip(lower_bound=0, upper_bound=top_k)
            .cast(pl.Int64)
            .alias("need_global")
        )
        .drop("kiosk_count")
    )
    if k_global_default == 0:
        needed_global = needed_global.with_columns(pl.lit(0).alias("need_global"))

    global_pool = (
        kiosk_anchors
        .join(global_top, on="anchor_product_id", how="inner")
        .join(
            kiosk_top.select(["kiosk_id", "anchor_product_id", "candidate_product_id"]),
            on=["kiosk_id", "anchor_product_id", "candidate_product_id"],
            how="anti",
        )
        .sort(
            ["kiosk_id", "anchor_product_id", "lift", "cooc_count"],
            descending=[False, False, True, True],
        )
        .with_columns(
            pl.col("candidate_product_id")
            .cum_count()
            .over(["kiosk_id", "anchor_product_id"])
            .alias("_global_rank")
        )
        .join(needed_global, on=["kiosk_id", "anchor_product_id"], how="left")
        .filter(pl.col("_global_rank") <= pl.col("need_global"))
        .drop(["_global_rank", "need_global"])
        .with_columns(pl.lit("global_mba").alias("source"))
    )

    combined = (
        pl.concat([kiosk_top, global_pool], how="vertical_relaxed")
        .with_columns(
            pl.when(pl.col("source") == "kiosk_mba")
            .then(1)
            .otherwise(0)
            .alias("_source_rank")
        )
        .sort(
            ["kiosk_id", "anchor_product_id", "_source_rank", "lift", "cooc_count"],
            descending=[False, False, True, True, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(top_k)
        .drop("_source_rank")
    )

    LOGGER.info(
        "Hybrid MBA kiosk candidates shape: %s (kiosk_share=%.2f top_k=%s)",
        combined.shape,
        kiosk_share,
        top_k,
    )
    return combined
