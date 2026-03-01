"""Batch scoring → precomputed predictions.parquet + popularity fallback.

Designed to run on **more data** than training used.
The model learns co-purchase patterns from a representative sample;
at inference we want to cover as many active kiosks as possible.

Typical cadence: daily or weekly.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from training.src.config import load_yaml_config
from training.src.features import add_all_features, lgbm_feature_exprs
from training.src.io import (
    load_orders_parquet,
    load_products_csv,
    load_commerces_csv,
    save_parquet,
)
from training.src.logging_utils import setup_logging
from training.src.paths import EXTERNAL_DIR, INTERIM_DIR, MODELS_DIR
from training.src.steps.build_baskets import build_baskets
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates


LOGGER = logging.getLogger(__name__)


def _load_feature_list(model_path: Path, ranker: lgb.Booster) -> list[str]:
    """Load the saved feature list; fall back to model-embedded names."""
    feature_path = model_path.with_suffix(".features.json")
    names = list(ranker.feature_name() or [])
    has_generic = bool(names) and all(str(n).startswith("Column_") for n in names)

    if feature_path.exists():
        file_names = json.loads(feature_path.read_text(encoding="utf-8"))
        if has_generic or not names:
            return list(file_names)

    return names


def _predict_scores_batched(
    ranker: lgb.Booster,
    df: pl.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    batch_size: int,
) -> np.ndarray:
    """Predict in batches using the **same** encoding as training."""
    if df.height == 0:
        return np.array([], dtype=np.float64)
    batch_size = max(1, int(batch_size))
    out: list[np.ndarray] = []
    for start in range(0, df.height, batch_size):
        chunk = (
            df.slice(start, batch_size)
            .select(lgbm_feature_exprs(feature_cols, categorical_cols))
            .to_numpy()
        )
        out.append(np.asarray(ranker.predict(chunk)))
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions.parquet from a trained model")
    parser.add_argument("--config", default="training/configs/generate_predictions.yaml")
    args = parser.parse_args()

    setup_logging("generate_predictions")

    cfg = load_yaml_config(Path(args.config))

    # ---- paths ----
    orders_path = Path(cfg.get("orders_path", INTERIM_DIR / "orders_sample.parquet"))
    products_path = Path(cfg.get("products_path", EXTERNAL_DIR / "products_v2.csv"))
    commerces_path = Path(cfg.get("commerces_path", EXTERNAL_DIR / "commerces.csv"))
    model_path = Path(cfg.get("model_path", MODELS_DIR / "lgbm_ranker.txt"))
    predictions_path = Path(cfg.get("predictions_path", INTERIM_DIR / "predictions.parquet"))
    popularity_path = Path(cfg.get("popularity_path", INTERIM_DIR / "popularity_fallback.parquet"))

    # ---- data selection ----
    inference_last_n_days = int(cfg.get("inference_last_n_days", 30))
    inference_max_rows = int(cfg.get("inference_max_rows", 0))
    query_sample_n = int(cfg.get("query_sample_n", 0))

    # ---- MBA candidate params ----
    min_cooc = int(cfg.get("min_cooc", 3))
    min_lift = float(cfg.get("min_lift", 2.0))
    top_k_candidates = int(cfg.get("top_k_candidates", 250))
    catalog_top_k = int(cfg.get("catalog_top_k", 100))
    predict_batch_size = int(cfg.get("predict_batch_size", 200_000))

    # ---- load data ----
    orders = load_orders_parquet(orders_path)
    products = load_products_csv(products_path)
    commerces = load_commerces_csv(commerces_path)

    # Filter to active kiosks
    n_total_active_kiosks: int = 0
    if "active" in commerces.columns:
        active_kiosks = (
            commerces
            .filter(pl.col("active") == True)  # noqa: E712
            .select(pl.col("userid").cast(pl.Utf8).alias("kiosk_id"))
            .drop_nulls()
            .unique()
        )
        n_total_active_kiosks = active_kiosks.height
        orders_before = orders.height
        kiosks_before = orders.select(pl.col("kiosk_id").n_unique()).item()
        orders = orders.join(active_kiosks, on="kiosk_id", how="inner")
        commerces = commerces.filter(pl.col("active") == True)  # noqa: E712
        orders_after = orders.height
        kiosks_after = orders.select(pl.col("kiosk_id").n_unique()).item() if orders_after > 0 else 0
        LOGGER.info(
            "Filtered to active kiosks: rows %s -> %s, kiosks %s -> %s (total active: %s)",
            orders_before, orders_after, kiosks_before, kiosks_after, n_total_active_kiosks,
        )
    else:
        LOGGER.warning("Column 'active' not found in commerces; skipping active kiosk filter.")

    # ---- select inference window ----
    max_dt = orders.select(pl.col("order_dt").max()).item()
    if max_dt is not None and inference_last_n_days > 0:
        cutoff = max_dt - pl.duration(days=inference_last_n_days)
        train_orders = orders.filter(pl.col("order_dt") >= cutoff)
    else:
        train_orders = orders

    if inference_max_rows and train_orders.height > inference_max_rows:
        train_orders = train_orders.tail(inference_max_rows)

    LOGGER.info("Inference orders: rows=%s kiosks=%s",
                train_orders.height,
                train_orders.select(pl.col("kiosk_id").n_unique()).item() if train_orders.height > 0 else 0)

    # ---- build baskets + candidates ----
    baskets_train = build_baskets(train_orders)

    ranker = lgb.Booster(model_file=str(model_path))
    model_feature_cols = _load_feature_list(model_path, ranker)
    if not model_feature_cols:
        raise ValueError("Model feature list is empty; retrain to persist features.")

    categorical_feature_cols = [c for c in ("channel", "region") if c in model_feature_cols]
    numeric_feature_cols = [c for c in model_feature_cols if c not in categorical_feature_cols]

    # MBA candidates
    candidates = generate_candidates(baskets_train, min_cooc=min_cooc)
    topk_candidates = select_top_k_candidates(candidates, k=top_k_candidates, min_lift=min_lift)

    # ---- build feature table ----
    queries = None
    if query_sample_n and query_sample_n > 0:
        queries = (
            baskets_train
            .select(["kiosk_id", "products"])
            .explode("products")
            .rename({"products": "anchor_product_id"})
            .unique()
            .sample(n=min(query_sample_n, baskets_train.height), seed=42)
        )

    feature_table = build_feature_table(
        baskets=baskets_train, topk_candidates=topk_candidates, queries=queries,
    )

    feature_table = add_all_features(
        feature_table, orders=train_orders, products=products, commerces=commerces,
    )

    # ---- align columns to model expectations ----
    missing_cols = [col for col in model_feature_cols if col not in feature_table.columns]
    if missing_cols:
        LOGGER.warning(
            "Missing features in inference: %s. Filling defaults.", missing_cols,
        )
        for col in missing_cols:
            fill_value = "__MISSING__" if col in categorical_feature_cols else 0
            feature_table = feature_table.with_columns(pl.lit(fill_value).alias(col))

    feature_table = feature_table.with_columns(
        [pl.col(c).fill_null(0) for c in numeric_feature_cols if c in feature_table.columns]
        + [pl.col(c).cast(pl.Utf8).fill_null("__MISSING__") for c in categorical_feature_cols if c in feature_table.columns]
    )

    # Diagnostic: check for zero-only features
    if numeric_feature_cols:
        feature_max_abs = feature_table.select(
            [pl.col(c).abs().max().alias(c) for c in numeric_feature_cols if c in feature_table.columns]
        ).row(0)
        cols_present = [c for c in numeric_feature_cols if c in feature_table.columns]
        zero_only = [name for name, max_abs in zip(cols_present, feature_max_abs) if max_abs == 0 or max_abs is None]
        if zero_only:
            LOGGER.warning("Zero-only features in inference (%s): %s", len(zero_only), zero_only[:10])

    # ---- product name / category lookup ----
    prod_info = (
        products
        .select([
            pl.col("productid").cast(pl.Utf8).alias("product_id"),
            pl.col("name").cast(pl.Utf8).alias("product_name"),
            pl.col("category").cast(pl.Utf8),
        ])
        .unique(subset=["product_id"])
    )

    # ---- predict ----
    scores = _predict_scores_batched(
        ranker, feature_table, model_feature_cols, categorical_feature_cols, predict_batch_size,
    )
    scored = feature_table.with_columns(pl.Series("score", scores))

    score_range = scored.select(
        pl.col("score").min().alias("min"),
        pl.col("score").max().alias("max"),
        pl.col("score").mean().alias("mean"),
    ).row(0)
    LOGGER.info("Score stats: min=%.6f max=%.6f mean=%.6f", *score_range)

    # ---- save predictions catalog ----
    final = (
        scored
        .sort(["kiosk_id", "anchor_product_id", "score"], descending=[False, False, True])
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(catalog_top_k)
        .select(["kiosk_id", "anchor_product_id", "candidate_product_id", "score"])
    )

    # Attach candidate name + category
    final = final.join(
        prod_info.rename({"product_id": "candidate_product_id",
                          "product_name": "candidate_name"}),
        on="candidate_product_id", how="left",
    )
    # Attach anchor name
    final = final.join(
        prod_info.select([
            pl.col("product_id").alias("anchor_product_id"),
            pl.col("product_name").alias("anchor_name"),
        ]),
        on="anchor_product_id", how="left",
    )

    final = final.select([
        "kiosk_id",
        "anchor_product_id", "anchor_name",
        "candidate_product_id", "candidate_name", "category",
        "score",
    ])

    save_parquet(final, predictions_path)

    # ---- coverage report ----
    catalog_kiosks = final.select(pl.col("kiosk_id").n_unique()).item()
    catalog_queries = final.select(
        pl.struct(["kiosk_id", "anchor_product_id"]).n_unique()
    ).item()
    total_active = n_total_active_kiosks if n_total_active_kiosks > 0 else catalog_kiosks
    coverage_pct = 100.0 * catalog_kiosks / total_active if total_active > 0 else 0.0

    LOGGER.info(
        "Catalog saved: %s rows | %s queries | %s kiosks | coverage %.1f%% (%s/%s active) | %s",
        f"{final.height:,}", f"{catalog_queries:,}", f"{catalog_kiosks:,}",
        coverage_pct, catalog_kiosks, total_active,
        predictions_path,
    )

    # ---- per-anchor co-purchase fallback ----
    # For kiosks not in the catalog we still know the anchor, so we provide
    # anchor-specific recommendations based on MBA co-occurrence (top-K by
    # cosine similarity).  This is much better than a flat global popularity list.
    anchor_fallback = (
        topk_candidates
        .sort(["anchor_product_id", "cooc_cosine_sim"], descending=[False, True])
        .group_by("anchor_product_id")
        .head(catalog_top_k)
    )
    # Normalize cosine similarity to model score range so serve_bundle can
    # compare with model scores seamlessly.
    score_min, score_max = float(score_range[0]), float(score_range[1])

    sim_min = anchor_fallback.select(pl.col("cooc_cosine_sim").min()).item()
    sim_max = anchor_fallback.select(pl.col("cooc_cosine_sim").max()).item()
    if sim_min is None or sim_max is None or sim_min == sim_max:
        anchor_fallback = anchor_fallback.with_columns(pl.lit(score_min).alias("score"))
    else:
        anchor_fallback = anchor_fallback.with_columns(
            ((pl.col("cooc_cosine_sim") - sim_min) / (sim_max - sim_min)
             * (score_max - score_min) + score_min).alias("score")
        )

    # Add product names
    anchor_fallback = (
        anchor_fallback
        .join(
            prod_info.rename({"product_id": "candidate_product_id",
                              "product_name": "candidate_name"}),
            on="candidate_product_id", how="left",
        )
        .join(
            prod_info.select([
                pl.col("product_id").alias("anchor_product_id"),
                pl.col("product_name").alias("anchor_name"),
            ]),
            on="anchor_product_id", how="left",
        )
        .select([
            "anchor_product_id", "anchor_name",
            "candidate_product_id", "candidate_name", "category",
            "score",
        ])
    )

    save_parquet(anchor_fallback, popularity_path)
    n_fb_anchors = anchor_fallback.select(pl.col("anchor_product_id").n_unique()).item()
    LOGGER.info(
        "Saved per-anchor fallback: %s rows | %s anchors | top-%s per anchor | %s",
        f"{anchor_fallback.height:,}", n_fb_anchors, catalog_top_k, popularity_path,
    )

    # ---- per-category popularity fallback ----
    # If anchor is unknown, we recommend popular items from the anchor's category.
    category_fallback_path = Path(
        cfg.get("category_fallback_path", INTERIM_DIR / "category_fallback.parquet")
    )
    cat_pop = (
        train_orders
        .join(
            prod_info.select([pl.col("product_id"), pl.col("category")]),
            on="product_id", how="left",
        )
        .filter(pl.col("category").is_not_null())
        .group_by(["category", "product_id"])
        .agg(pl.len().alias("purchase_count"))
        .sort(["category", "purchase_count"], descending=[False, True])
        .group_by("category")
        .head(catalog_top_k)
    )
    # Normalize scores to model range
    cat_pop = cat_pop.with_columns(pl.col("purchase_count").cast(pl.Float64).log1p().alias("_pop"))
    cpop_min = cat_pop.select(pl.col("_pop").min()).item()
    cpop_max = cat_pop.select(pl.col("_pop").max()).item()
    if cpop_min is None or cpop_max is None or cpop_min == cpop_max:
        cat_pop = cat_pop.with_columns(pl.lit(score_min).alias("score"))
    else:
        cat_pop = cat_pop.with_columns(
            ((pl.col("_pop") - cpop_min) / (cpop_max - cpop_min)
             * (score_max - score_min) + score_min).alias("score")
        )
    cat_pop = (
        cat_pop
        .drop(["_pop", "purchase_count"])
        .rename({"product_id": "candidate_product_id"})
        .join(
            prod_info.rename({"product_id": "candidate_product_id",
                              "product_name": "candidate_name"})
            .select(["candidate_product_id", "candidate_name"]),
            on="candidate_product_id", how="left",
        )
        .select(["category", "candidate_product_id", "candidate_name", "score"])
    )
    save_parquet(cat_pop, category_fallback_path)
    n_cats = cat_pop.select(pl.col("category").n_unique()).item()
    LOGGER.info(
        "Saved category fallback: %s rows | %s categories | top-%s per category | %s",
        f"{cat_pop.height:,}", n_cats, catalog_top_k, category_fallback_path,
    )

    # ---- global popularity fallback ----
    # Absolute last resort: top-N most purchased products overall.
    global_fallback_path = Path(
        cfg.get("global_fallback_path", INTERIM_DIR / "global_fallback.parquet")
    )
    global_pop = (
        train_orders
        .group_by("product_id")
        .agg(pl.len().alias("purchase_count"))
        .sort("purchase_count", descending=True)
        .head(catalog_top_k)
        .rename({"product_id": "candidate_product_id"})
    )
    gpop_min = global_pop.select(pl.col("purchase_count").cast(pl.Float64).log1p().min()).item()
    gpop_max = global_pop.select(pl.col("purchase_count").cast(pl.Float64).log1p().max()).item()
    if gpop_min is None or gpop_max is None or gpop_min == gpop_max:
        global_pop = global_pop.with_columns(pl.lit(score_min).alias("score"))
    else:
        global_pop = global_pop.with_columns(
            ((pl.col("purchase_count").cast(pl.Float64).log1p() - gpop_min) / (gpop_max - gpop_min)
             * (score_max - score_min) + score_min).alias("score")
        )
    global_pop = (
        global_pop
        .drop("purchase_count")
        .join(
            prod_info.rename({"product_id": "candidate_product_id",
                              "product_name": "candidate_name"}),
            on="candidate_product_id", how="left",
        )
        .select(["candidate_product_id", "candidate_name", "category", "score"])
    )
    save_parquet(global_pop, global_fallback_path)
    LOGGER.info(
        "Saved global fallback: %s items | %s", global_pop.height, global_fallback_path,
    )


if __name__ == "__main__":
    main()
