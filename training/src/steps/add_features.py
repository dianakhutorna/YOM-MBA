"""
Compute all features for the bundle recommendation LightGBM model.

Input:  base feature table with (kiosk_id, anchor_product_id, candidate_product_id, cooc_cosine_sim)
Output: same table enriched with features ready for LightGBM.

Features added:
  pop_store          — how many times this kiosk ordered the candidate product
  pop_global         — how many times the candidate was ordered across all kiosks
  kiosk_product_cnt  — total order rows for this kiosk (proxy for kiosk size)
  same_category      — 1 if anchor and candidate share the same product category
  cand_is_new        — 1 if the kiosk has never ordered the candidate before
  channel            — kiosk sales channel  (categorical, kept as string)
  region             — kiosk geographic region (categorical, kept as string)
"""

from __future__ import annotations

import polars as pl

KEY_COLS = ["kiosk_id", "anchor_product_id", "candidate_product_id"]


def add_features(
    feature_table: pl.DataFrame,
    *,
    orders: pl.DataFrame,
    products: pl.DataFrame | None = None,
    commerces: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Add all features to the base feature table in a single pass."""
    ft = feature_table

    # ---- popularity: store-level (kiosk × candidate) ----
    kiosk_product_counts = (
        orders
        .group_by(["kiosk_id", "product_id"])
        .len()
        .rename({"len": "pop_store"})
    )
    ft = ft.join(
        kiosk_product_counts,
        left_on=["kiosk_id", "candidate_product_id"],
        right_on=["kiosk_id", "product_id"],
        how="left",
    ).with_columns(pl.col("pop_store").fill_null(0))

    # ---- popularity: global (candidate across all kiosks) ----
    pop_global = (
        kiosk_product_counts
        .group_by("product_id")
        .agg(pl.col("pop_store").sum().alias("pop_global"))
    )
    ft = ft.join(
        pop_global,
        left_on="candidate_product_id",
        right_on="product_id",
        how="left",
    ).with_columns(pl.col("pop_global").fill_null(0))

    # ---- kiosk volume ----
    kiosk_stats = (
        orders
        .group_by("kiosk_id")
        .agg(pl.len().alias("kiosk_product_cnt"))
    )
    ft = ft.join(kiosk_stats, on="kiosk_id", how="left").with_columns(
        pl.col("kiosk_product_cnt").fill_null(0)
    )

    # ---- is-new: kiosk never ordered this candidate ----
    ft = ft.with_columns(
        (pl.col("pop_store") == 0).cast(pl.Int8).alias("cand_is_new")
    )

    # ---- product category pair feature ----
    if products is not None and "productid" in products.columns and "category" in products.columns:
        prod = (
            products
            .select(
                pl.col("productid").alias("product_id"),
                pl.col("category"),
            )
            .unique(subset=["product_id"])
        )
        ft = (
            ft
            .join(
                prod.rename({"product_id": "anchor_product_id", "category": "anchor_cat"}),
                on="anchor_product_id",
                how="left",
            )
            .join(
                prod.rename({"product_id": "candidate_product_id", "category": "cand_cat"}),
                on="candidate_product_id",
                how="left",
            )
            .with_columns(
                (pl.col("anchor_cat") == pl.col("cand_cat")).cast(pl.Int8).alias("same_category")
            )
            .drop(["anchor_cat", "cand_cat"])
        )

    # ---- kiosk metadata (channel, region) ----
    if commerces is not None and "userid" in commerces.columns:
        comm = commerces.select(
            pl.col("userid").alias("kiosk_id"),
            "channel",
            "region",
        )
        ft = ft.join(comm, on="kiosk_id", how="left")

    return ft
