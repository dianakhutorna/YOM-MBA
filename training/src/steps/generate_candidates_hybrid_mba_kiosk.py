from __future__ import annotations

import logging
import time
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
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _empty_schema_df() -> pl.DataFrame:
    """
    Minimal unified schema for this generator.

    Important: kiosk_top and global_pool must share the same columns for concat.
    """
    return pl.DataFrame(
        schema={
            "kiosk_id": pl.Utf8,
            "anchor_product_id": pl.Utf8,
            "candidate_product_id": pl.Utf8,
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
        .rename({"product_id": "anchor_product_id", "product_id_right": "candidate_product_id"})
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
            product_counts.rename({"product_id": "anchor_product_id", "product_count": "anchor_count"}),
            on=["kiosk_id", "anchor_product_id"],
            how="left",
        )
        .join(
            product_counts.rename({"product_id": "candidate_product_id", "product_count": "candidate_count"}),
            on=["kiosk_id", "candidate_product_id"],
            how="left",
        )
        .join(basket_counts_batch, on="kiosk_id", how="left")
        .with_columns(
            [
                (pl.col("cooc_count") / pl.col("kiosk_basket_count")).alias("support"),
                (pl.col("cooc_count") / pl.col("anchor_count")).alias("confidence"),
                (
                    (pl.col("cooc_count") / pl.col("kiosk_basket_count"))
                    / (
                        (pl.col("anchor_count") / pl.col("kiosk_basket_count"))
                        * (pl.col("candidate_count") / pl.col("kiosk_basket_count"))
                    )
                ).alias("lift"),
                pl.when((pl.col("anchor_count") > 0) & (pl.col("candidate_count") > 0))
                .then(pl.col("cooc_count") / (pl.col("anchor_count") * pl.col("candidate_count")).sqrt())
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
        .with_columns(pl.col("cooc_cosine_sim").cast(pl.Float64))
        .select(
            [
                "kiosk_id",
                "anchor_product_id",
                "candidate_product_id",
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
      - global MBA part (fallback)

    Output columns:
      - kiosk_id
      - anchor_product_id
      - candidate_product_id
      - cooc_cosine_sim
      - source
    """
    _ensure_columns(baskets, REQUIRED_BASKET_COLS)
    if baskets.is_empty() or top_k <= 0:
        return _empty_schema_df()

    t0 = time.perf_counter()

    kiosk_share = float(max(0.0, min(1.0, kiosk_share)))
    k_kiosk = int(round(top_k * kiosk_share))
    k_global_default = top_k - k_kiosk
    kiosk_batch_size = max(1, int(kiosk_batch_size))

    LOGGER.info(
        "HybridMBA(kiosk): start (rows=%s) min_cooc=%s min_lift=%.3f top_k=%s kiosk_share=%.2f (k_kiosk=%s k_global=%s) batch=%s",
        baskets.height, min_cooc, float(min_lift), top_k, kiosk_share, k_kiosk, k_global_default, kiosk_batch_size
    )

    # ---- explode baskets ----
    t_explode = time.perf_counter()
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
        LOGGER.info("HybridMBA(kiosk): exploded empty -> returning empty")
        return _empty_schema_df()

    basket_counts_all = (
        baskets
        .with_columns(pl.col("kiosk_id").cast(pl.Utf8))
        .group_by("kiosk_id")
        .agg(pl.len().alias("kiosk_basket_count"))
    )
    kiosks = basket_counts_all.select("kiosk_id").to_series().to_list()

    kiosk_anchors = (
        exploded
        .select([pl.col("kiosk_id"), pl.col("product_id").alias("anchor_product_id")])
        .unique()
    )

    LOGGER.info(
        "HybridMBA(kiosk): exploded done in %.2fs (rows=%s). kiosk_anchors=%s kiosks=%s",
        time.perf_counter() - t_explode,
        exploded.height,
        kiosk_anchors.height,
        len(kiosks),
    )

    # ---- global MBA (light) ----
    global_top = pl.DataFrame(
        schema={
            "anchor_product_id": pl.Utf8,
            "candidate_product_id": pl.Utf8,
            "cooc_cosine_sim": pl.Float64,
        }
    )
    if k_global_default > 0:
        t_global = time.perf_counter()
        global_mba = generate_candidates(baskets, min_cooc=min_cooc)
        global_top = (
            select_top_k_candidates(global_mba, k=top_k, min_lift=min_lift)
            .select(["anchor_product_id", "candidate_product_id", "cooc_cosine_sim"])
            .with_columns(
                [
                    pl.col("anchor_product_id").cast(pl.Utf8),
                    pl.col("candidate_product_id").cast(pl.Utf8),
                    pl.col("cooc_cosine_sim").cast(pl.Float64),
                ]
            )
        )
        LOGGER.info(
            "HybridMBA(kiosk): global MBA done in %.2fs (anchors=%s rows=%s)",
            time.perf_counter() - t_global,
            global_top.select("anchor_product_id").n_unique(),
            global_top.height,
        )
    else:
        LOGGER.info("HybridMBA(kiosk): global MBA skipped (k_global_default=0)")

    # ---- kiosk MBA (batched) ----
    kiosk_top = _empty_schema_df()
    if k_kiosk > 0:
        t_kiosk = time.perf_counter()
        kiosk_parts: list[pl.DataFrame] = []
        n_batches = (len(kiosks) + kiosk_batch_size - 1) // kiosk_batch_size

        for b, start in enumerate(range(0, len(kiosks), kiosk_batch_size), start=1):
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

            if n_batches <= 10 or b in {1, n_batches} or (b % max(1, n_batches // 10) == 0):
                LOGGER.info("HybridMBA(kiosk): kiosk batches progress %s/%s", b, n_batches)

        if kiosk_parts:
            kiosk_top = (
                pl.concat(kiosk_parts, how="vertical_relaxed")
                .with_columns(pl.lit("kiosk_mba").alias("source"))
            )
        else:
            kiosk_top = _empty_schema_df()

        LOGGER.info(
            "HybridMBA(kiosk): kiosk MBA done in %.2fs (rows=%s)",
            time.perf_counter() - t_kiosk,
            kiosk_top.height,
        )
    else:
        LOGGER.info("HybridMBA(kiosk): kiosk MBA skipped (k_kiosk=0)")

    # ---- compute needed global per (kiosk, anchor) ----
    needed_global = None
    if k_global_default > 0:
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
                .clip(lower_bound=0, upper_bound=k_global_default)
                .cast(pl.Int64)
                .alias("need_global")
            )
            .drop("kiosk_count")
        )

        need_sum = int(needed_global.select(pl.col("need_global").sum()).item())
        LOGGER.info("HybridMBA(kiosk): need_global total=%s (pairs=%s)", need_sum, needed_global.height)
        if need_sum == 0:
            needed_global = None
            LOGGER.info("HybridMBA(kiosk): global fill skipped (need_global=0)")

    # ---- build global pool (light) ----
    global_pool = _empty_schema_df()
    if k_global_default > 0 and needed_global is not None and global_top.height > 0:
        t_pool = time.perf_counter()

        # join global candidates per anchor onto each kiosk-anchor pair
        global_pool = (
            kiosk_anchors
            .join(global_top, on="anchor_product_id", how="inner")
            .join(
                kiosk_top.select(["kiosk_id", "anchor_product_id", "candidate_product_id"]),
                on=["kiosk_id", "anchor_product_id", "candidate_product_id"],
                how="anti",
            )
            .sort(
                ["kiosk_id", "anchor_product_id", "cooc_cosine_sim"],
                descending=[False, False, True],
            )
            .with_columns(
                (
                    pl.col("candidate_product_id")
                    .cum_count()
                    .over(["kiosk_id", "anchor_product_id"]) + 1
                ).alias("_global_rank")
            )
            .join(needed_global, on=["kiosk_id", "anchor_product_id"], how="left")
            .with_columns(pl.col("need_global").fill_null(0).alias("need_global"))
            .filter(pl.col("_global_rank") <= pl.col("need_global"))
            .drop(["_global_rank", "need_global"])
            .with_columns(pl.lit("global_mba").alias("source"))
            .select(["kiosk_id", "anchor_product_id", "candidate_product_id", "cooc_cosine_sim", "source"])
        )

        LOGGER.info(
            "HybridMBA(kiosk): global fill done in %.2fs (rows=%s)",
            time.perf_counter() - t_pool,
            global_pool.height,
        )

    # ---- combine & final top_k ----
    combined = (
        pl.concat([kiosk_top, global_pool], how="vertical_relaxed")
        .with_columns(
            pl.when(pl.col("source") == "kiosk_mba").then(1).otherwise(0).alias("_source_rank")
        )
        .sort(
            ["kiosk_id", "anchor_product_id", "_source_rank", "cooc_cosine_sim"],
            descending=[False, False, True, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(top_k)
        .drop("_source_rank")
    )

    if combined.height > 0 and "source" in combined.columns:
        src = combined.group_by("source").agg(pl.len().alias("rows")).sort("rows", descending=True)
        LOGGER.info("HybridMBA(kiosk): source mix: %s", src.to_dicts())

    LOGGER.info(
        "HybridMBA(kiosk): done in %.2fs (rows=%s cols=%s)",
        time.perf_counter() - t0,
        combined.height,
        combined.width,
    )
    return combined
