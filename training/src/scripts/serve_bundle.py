from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from training.src.config import load_yaml_config
from training.src.io import load_parquet, load_products_csv
from training.src.logging_utils import setup_logging
from training.src.paths import EXTERNAL_DIR, INTERIM_DIR


LOGGER = logging.getLogger(__name__)


def _normalize_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in {"", "none", "null"}:
        return None
    return stripped


def _parse_list(raw: str | None) -> list[str]:
    raw = _normalize_none(raw)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def apply_bundle_rules(
    df_scored: pl.DataFrame,
    products: pl.DataFrame | None,
    *,
    kiosk_id: str,
    anchor_product_id: str,
    included_products: list[str],
    excluded_products: list[str],
    allowed_categories: list[str],
    n_group_key: int | None,
    n_min: int,
    n_max: int,
) -> pl.DataFrame:
    prod_map = None
    if products is not None:
        prod_map = products.select(
            [
                pl.col("productid").cast(pl.Utf8).alias("product_id"),
                pl.col("category").cast(pl.Utf8),
            ]
        )

    df = (
        df_scored
        .with_columns(pl.col("candidate_product_id").cast(pl.Utf8))
    )
    if "category" not in df.columns and prod_map is not None:
        df = df.join(
            prod_map,
            left_on="candidate_product_id",
            right_on="product_id",
            how="left",
        )

    if excluded_products:
        df = df.filter(~pl.col("candidate_product_id").is_in(excluded_products))

    if allowed_categories:
        df = df.filter(pl.col("category").is_in(allowed_categories))

    df = df.sort("score", descending=True)

    if n_group_key is not None and n_group_key > 0:
        df = (
            df
            .with_columns(
                pl.col("category")
                .cum_count()
                .over("category")
                .alias("_cat_rank")
            )
            .filter(pl.col("_cat_rank") <= n_group_key)
            .drop("_cat_rank")
        )

    if included_products:
        included_set = set(included_products)
        present = set(df.select("candidate_product_id").to_series().to_list())
        missing = list(included_set - present)
        if missing:
            max_score = df.select(pl.max("score")).item()
            bonus = (max_score if max_score is not None else 0.0) + 1e6
            add_rows = (
                pl.DataFrame(
                    {
                        "kiosk_id": [kiosk_id] * len(missing),
                        "anchor_product_id": [anchor_product_id] * len(missing),
                        "candidate_product_id": missing,
                        "score": [bonus] * len(missing),
                    }
                )
            )
            if prod_map is not None:
                add_rows = add_rows.join(
                    prod_map,
                    left_on="candidate_product_id",
                    right_on="product_id",
                    how="left",
                )
            df = pl.concat([add_rows, df], how="diagonal").sort("score", descending=True)

    df = df.head(n_max)
    if df.height < n_min:
        LOGGER.warning("Returned only %s items (< N_min=%s).", df.height, n_min)

    return df


def _fill_from_fallback(
    df: pl.DataFrame,
    fallback_rows: pl.DataFrame,
    *,
    kiosk_id: str,
    anchor_product_id: str,
    excluded_products: list[str],
    allowed_categories: list[str],
    n_max: int,
) -> pl.DataFrame:
    """Append items from *fallback_rows* to *df* until it reaches *n_max*.

    Deduplicates against items already in *df* and applies exclusion /
    category filters.  Scores are set below the current minimum so that
    fallback items always appear after model-scored items.
    """
    if fallback_rows.is_empty():
        return df

    fb = fallback_rows.with_columns(
        pl.lit(kiosk_id).alias("kiosk_id"),
        pl.lit(anchor_product_id).alias("anchor_product_id"),
    )

    # Assign scores below the current lowest so order is preserved
    if df.height > 0:
        min_score = df.select(pl.min("score")).item() or 0.0
        fb = fb.with_columns(
            (pl.lit(min_score) - 1.0 - pl.arange(0, pl.len()).cast(pl.Float64) * 0.001).alias("score")
        )
    else:
        fb = fb.with_columns((-pl.arange(0, pl.len()).cast(pl.Float64)).alias("score"))

    if "category" not in fb.columns:
        fb = fb.with_columns(pl.lit(None).cast(pl.Utf8).alias("category"))

    # Dedup + exclusions
    seen = set(df.select("candidate_product_id").to_series().to_list())
    fb = fb.filter(~pl.col("candidate_product_id").is_in(list(seen)))
    fb = fb.filter(pl.col("candidate_product_id") != anchor_product_id)
    if excluded_products:
        fb = fb.filter(~pl.col("candidate_product_id").is_in(excluded_products))
    if allowed_categories and "category" in fb.columns:
        fb = fb.filter(pl.col("category").is_in(allowed_categories))

    need = max(0, n_max - df.height)
    if need == 0:
        return df
    fb = fb.sort("score", descending=True).head(need)

    # Align schemas
    base_cols = df.columns if df.height > 0 else [
        "kiosk_id", "anchor_product_id", "candidate_product_id", "category", "score",
    ]
    for col in base_cols:
        if col not in fb.columns:
            fb = fb.with_columns(pl.lit(None).alias(col))
    fb = fb.select(base_cols)
    if df.height > 0:
        df = df.select(base_cols)
    return pl.concat([df, fb], how="vertical").sort("score", descending=True)


def _get_anchor_fallback(
    fallback: pl.DataFrame, anchor_product_id: str,
) -> pl.DataFrame:
    """Filter per-anchor fallback; return empty DataFrame if anchor unknown."""
    if "anchor_product_id" in fallback.columns:
        return fallback.filter(pl.col("anchor_product_id") == anchor_product_id)
    return fallback


def _get_category_fallback(
    cat_fallback: pl.DataFrame | None,
    products: pl.DataFrame | None,
    anchor_product_id: str,
) -> pl.DataFrame:
    """Get category-level popular items matching the anchor's category."""
    empty = pl.DataFrame(schema={"candidate_product_id": pl.Utf8, "category": pl.Utf8, "score": pl.Float64})
    if cat_fallback is None or cat_fallback.is_empty():
        return empty

    # Look up anchor's category from products table
    anchor_category: str | None = None
    if products is not None and "category" in products.columns:
        row = products.filter(
            pl.col("productid").cast(pl.Utf8) == anchor_product_id
        ).select("category").head(1)
        if row.height > 0:
            anchor_category = row.item()

    if anchor_category is None:
        return empty

    return cat_fallback.filter(pl.col("category") == anchor_category)


def build_bundle(
    preds: pl.DataFrame,
    fallback: pl.DataFrame,
    products: pl.DataFrame | None,
    *,
    kiosk_id: str,
    anchor_product_id: str,
    included_products: list[str],
    excluded_products: list[str],
    allowed_categories: list[str],
    n_group_key: int | None,
    n_min: int,
    n_max: int,
    category_fallback: pl.DataFrame | None = None,
    global_fallback: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a bundle with multi-level fallback.

    Priority:
      1. LightGBM predictions (personalized)
      2. Per-anchor MBA co-purchase fallback
      3. Per-category popularity fallback
      4. Global popularity fallback
    """
    anchor_fb = _get_anchor_fallback(fallback, anchor_product_id)
    if "category" not in anchor_fb.columns:
        anchor_fb = anchor_fb.with_columns(pl.lit(None).cast(pl.Utf8).alias("category"))

    # --- Level 1: model predictions ---
    df_scored = preds.filter(
        (pl.col("kiosk_id") == kiosk_id) &
        (pl.col("anchor_product_id") == anchor_product_id)
    )

    # --- Level 2: per-anchor MBA fallback ---
    if df_scored.is_empty():
        LOGGER.info("No predictions for kiosk+anchor. Trying per-anchor fallback.")
        df_scored = (
            anchor_fb
            .with_columns(
                pl.lit(kiosk_id).alias("kiosk_id"),
                pl.lit(anchor_product_id).alias("anchor_product_id"),
            )
            .select(["kiosk_id", "anchor_product_id", "candidate_product_id", "category", "score"])
        )

    final = apply_bundle_rules(
        df_scored,
        products,
        kiosk_id=kiosk_id,
        anchor_product_id=anchor_product_id,
        included_products=included_products,
        excluded_products=excluded_products,
        allowed_categories=allowed_categories,
        n_group_key=n_group_key,
        n_min=n_min,
        n_max=n_max,
    )

    # --- Level 2b: fill from anchor fallback if not enough ---
    if final.height < n_max:
        final = _fill_from_fallback(
            final, anchor_fb,
            kiosk_id=kiosk_id, anchor_product_id=anchor_product_id,
            excluded_products=excluded_products, allowed_categories=allowed_categories,
            n_max=n_max,
        )

    # --- Level 3: per-category popularity fallback ---
    if final.height < n_max and category_fallback is not None:
        cat_fb = _get_category_fallback(category_fallback, products, anchor_product_id)
        if not cat_fb.is_empty():
            LOGGER.info("Filling from category fallback (%s items available).", cat_fb.height)
            final = _fill_from_fallback(
                final, cat_fb,
                kiosk_id=kiosk_id, anchor_product_id=anchor_product_id,
                excluded_products=excluded_products, allowed_categories=allowed_categories,
                n_max=n_max,
            )

    # --- Level 4: global popularity fallback ---
    if final.height < n_max and global_fallback is not None:
        LOGGER.info("Filling from global fallback.")
        final = _fill_from_fallback(
            final, global_fallback,
            kiosk_id=kiosk_id, anchor_product_id=anchor_product_id,
            excluded_products=excluded_products, allowed_categories=allowed_categories,
            n_max=n_max,
        )

    # --- Final pass: enforce n_group_key across all sources ---
    if n_group_key is not None and n_group_key > 0 and final.height > 0:
        if "category" not in final.columns and products is not None:
            prod_map = products.select(
                pl.col("productid").cast(pl.Utf8).alias("product_id"),
                pl.col("category").cast(pl.Utf8),
            )
            final = final.join(prod_map, left_on="candidate_product_id", right_on="product_id", how="left")
        if "category" in final.columns:
            final = (
                final
                .sort("score", descending=True)
                .with_columns(
                    pl.col("category").cum_count().over("category").alias("_cat_rank")
                )
                .filter(pl.col("_cat_rank") <= n_group_key)
                .drop("_cat_rank")
                .head(n_max)
            )

    if final.height == 0:
        LOGGER.warning("Bundle is empty after all fallbacks.")

    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve bundle from predictions.parquet with fallback")
    parser.add_argument("--config", default="training/configs/serve_bundle.yaml")
    parser.add_argument("--kiosk-id", required=False, default="")
    parser.add_argument("--anchor-product-id", required=False, default="")
    parser.add_argument("--included-products", default="")
    parser.add_argument("--excluded-products", default="")
    parser.add_argument("--allowed-categories", default="")
    parser.add_argument("--n-group-key", type=int, default=0)
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=20)
    args = parser.parse_args()

    setup_logging("serve_bundle")

    cfg = load_yaml_config(Path(args.config)) if args.config else {}

    kiosk_id = _normalize_none(args.kiosk_id) or str(cfg.get("kiosk_id", "")).strip()
    anchor_product_id = _normalize_none(args.anchor_product_id) or str(cfg.get("anchor_product_id", "")).strip()
    if not kiosk_id or not anchor_product_id:
        raise ValueError("kiosk_id and anchor_product_id are required.")

    included_products = _parse_list(args.included_products or cfg.get("included_products"))
    excluded_products = _parse_list(args.excluded_products or cfg.get("excluded_products"))
    allowed_categories = _parse_list(args.allowed_categories or cfg.get("allowed_categories") or cfg.get("agg_key"))

    n_group_key = args.n_group_key if args.n_group_key > 0 else int(cfg.get("n_group_key", 0)) or None
    n_min = max(1, args.n_min if args.n_min else int(cfg.get("n_min", 10)))
    n_max = max(n_min, args.n_max if args.n_max else int(cfg.get("n_max", 20)))

    predictions_path = Path(cfg.get("predictions_path", INTERIM_DIR / "predictions.parquet"))
    popularity_path = Path(cfg.get("popularity_path", INTERIM_DIR / "popularity_fallback.parquet"))
    category_fallback_path = Path(cfg.get("category_fallback_path", INTERIM_DIR / "category_fallback.parquet"))
    global_fallback_path = Path(cfg.get("global_fallback_path", INTERIM_DIR / "global_fallback.parquet"))
    products_path = Path(cfg.get("products_path", EXTERNAL_DIR / "products_v2.csv"))

    preds = load_parquet(predictions_path, label="Predictions parquet")
    fallback = load_parquet(popularity_path, label="Anchor fallback")

    # Always load products for category fallback lookup
    products = load_products_csv(products_path)

    cat_fb = load_parquet(category_fallback_path, label="Category fallback") if category_fallback_path.exists() else None
    glob_fb = load_parquet(global_fallback_path, label="Global fallback") if global_fallback_path.exists() else None

    final = build_bundle(
        preds,
        fallback,
        products,
        kiosk_id=kiosk_id,
        anchor_product_id=anchor_product_id,
        included_products=included_products,
        excluded_products=excluded_products,
        allowed_categories=allowed_categories,
        n_group_key=n_group_key,
        n_min=n_min,
        n_max=n_max,
        category_fallback=cat_fb,
        global_fallback=glob_fb,
    )

    print(
        final.select(
            [
                "kiosk_id",
                "anchor_product_id",
                "candidate_product_id",
                "category",
                pl.col("score").round(6).alias("score"),
            ]
        )
    )


if __name__ == "__main__":
    main()
