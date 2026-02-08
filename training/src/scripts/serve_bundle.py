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
            df = pl.concat([add_rows, df], how="vertical").sort("score", descending=True)

    df = df.head(n_max)
    if df.height < n_min:
        LOGGER.warning("Returned only %s items (< N_min=%s).", df.height, n_min)

    return df


def _fill_with_popularity(
    df: pl.DataFrame,
    fallback: pl.DataFrame,
    *,
    kiosk_id: str,
    anchor_product_id: str,
    excluded_products: list[str],
    allowed_categories: list[str],
    n_max: int,
) -> pl.DataFrame:
    if fallback.is_empty():
        return df

    fallback_df = fallback.with_columns(
        [
            pl.lit(kiosk_id).alias("kiosk_id"),
            pl.lit(anchor_product_id).alias("anchor_product_id"),
        ]
    )
    if df.height > 0:
        min_score = df.select(pl.min("score")).item()
        if min_score is None:
            min_score = 0.0
        fallback_df = fallback_df.with_columns(pl.lit(min_score - 1.0).alias("score"))
    else:
        # No model scores; use rank-based small negative scores to keep scale reasonable
        fallback_df = fallback_df.with_columns(
            (-pl.arange(0, pl.len()).cast(pl.Float64)).alias("score")
        )
    if "category" not in fallback_df.columns:
        fallback_df = fallback_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("category"))

    seen = set(df.select("candidate_product_id").to_series().to_list())
    fallback_df = fallback_df.filter(~pl.col("candidate_product_id").is_in(list(seen)))
    fallback_df = fallback_df.filter(pl.col("candidate_product_id") != anchor_product_id)
    if excluded_products:
        fallback_df = fallback_df.filter(~pl.col("candidate_product_id").is_in(excluded_products))

    if allowed_categories and "category" in fallback_df.columns:
        filtered = fallback_df.filter(pl.col("category").is_in(allowed_categories))
        if filtered.height == 0:
            LOGGER.warning("Popularity fallback has no items in allowed_categories; using unfiltered fallback.")
        else:
            fallback_df = filtered

    need = max(0, n_max - df.height)
    if need == 0:
        return df
    fallback_df = fallback_df.sort("score", descending=True).head(need)
    # Align schemas before concat
    if df.height > 0:
        base_cols = df.columns
    else:
        base_cols = ["kiosk_id", "anchor_product_id", "candidate_product_id", "category", "score"]
    for col in base_cols:
        if col not in fallback_df.columns:
            fallback_df = fallback_df.with_columns(pl.lit(None).alias(col))
    fallback_df = fallback_df.select(base_cols)
    if df.height > 0:
        df = df.select(base_cols)
    out = pl.concat([df, fallback_df], how="vertical").sort("score", descending=True)
    return out


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
) -> pl.DataFrame:
    if "category" not in fallback.columns:
        fallback = fallback.with_columns(pl.lit(None).cast(pl.Utf8).alias("category"))
    df_scored = preds.filter(
        (pl.col("kiosk_id") == kiosk_id) &
        (pl.col("anchor_product_id") == anchor_product_id)
    )

    if df_scored.is_empty():
        LOGGER.warning("No predictions for kiosk+anchor. Using popularity fallback.")
        df_scored = (
            fallback
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

    if final.height < n_min:
        final = _fill_with_popularity(
            final,
            fallback,
            kiosk_id=kiosk_id,
            anchor_product_id=anchor_product_id,
            excluded_products=excluded_products,
            allowed_categories=allowed_categories,
            n_max=n_max,
        )
    if final.height == 0:
        LOGGER.warning("Bundle is empty after fallback; returning raw popularity fallback.")
        final = (
            fallback
            .with_columns(
                [
                    pl.lit(kiosk_id).alias("kiosk_id"),
                    pl.lit(anchor_product_id).alias("anchor_product_id"),
                ]
            )
            .select(["kiosk_id", "anchor_product_id", "candidate_product_id", "category", "score"])
            .head(n_max)
        )
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
    products_path = Path(cfg.get("products_path", EXTERNAL_DIR / "products_v2.csv"))

    preds = load_parquet(predictions_path, label="Predictions parquet")
    fallback = load_parquet(popularity_path, label="Popularity fallback")
    products = None
    need_products = (
        "category" not in preds.columns or
        (included_products and "category" not in preds.columns) or
        (allowed_categories and "category" not in preds.columns) or
        (n_group_key and "category" not in preds.columns)
    )
    if need_products:
        products = load_products_csv(products_path)

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
