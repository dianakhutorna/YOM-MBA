"""
Batch scoring: load a trained model and generate a precomputed
predictions.parquet + popularity fallback for the serving layer.

This script is designed to run on **more data** than training used.
The model learns co-purchase patterns from a representative sample;
at inference we want to cover as many active kiosks as possible,
using the freshest available order history.

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
from training.src.steps.split_orders import split_orders_by_time


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
    # Use inference_last_n_days > 0 to select a recent window.
    # Use inference_max_rows = 0 (unlimited) to cover all active kiosks.
    train_ratio = float(cfg.get("train_ratio", 0.7))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    test_ratio = float(cfg.get("test_ratio", 0.2))
    inference_last_n_days = int(cfg.get("inference_last_n_days", 0))
    inference_max_rows = int(cfg.get("inference_max_rows", 0))
    query_sample_n = int(cfg.get("query_sample_n", 0))

    # ---- MBA candidate params ----
    min_cooc = int(cfg.get("min_cooc", 3))
    min_lift = float(cfg.get("min_lift", 2.0))
    top_k_candidates = int(cfg.get("top_k_candidates", 250))
    catalog_top_k = int(cfg.get("catalog_top_k", 100))
    predict_batch_size = int(cfg.get("predict_batch_size", 200_000))
    normalize_popularity = bool(cfg.get("normalize_popularity", True))

    # ---- load data ----
    orders = load_orders_parquet(orders_path)
    products = load_products_csv(products_path)
    commerces = load_commerces_csv(commerces_path)

    # Filter to active kiosks
    if "active" in commerces.columns:
        active_kiosks = (
            commerces
            .filter(pl.col("active") == True)  # noqa: E712
            .select(pl.col("userid").cast(pl.Utf8).alias("kiosk_id"))
            .drop_nulls()
            .unique()
        )
        orders_before = orders.height
        kiosks_before = orders.select(pl.col("kiosk_id").n_unique()).item()
        orders = orders.join(active_kiosks, on="kiosk_id", how="inner")
        commerces = commerces.filter(pl.col("active") == True)  # noqa: E712
        orders_after = orders.height
        kiosks_after = orders.select(pl.col("kiosk_id").n_unique()).item() if orders_after > 0 else 0
        LOGGER.info(
            "Filtered to active kiosks: rows %s -> %s, kiosks %s -> %s",
            orders_before, orders_after, kiosks_before, kiosks_after,
        )
    else:
        LOGGER.warning("Column 'active' not found in commerces; skipping active kiosk filter.")

    # ---- select inference window ----
    if inference_last_n_days and "order_dt" in orders.columns:
        max_dt = orders.select(pl.col("order_dt").max()).item()
        if max_dt is not None:
            cutoff = max_dt - pl.duration(days=inference_last_n_days)
            train_orders = orders.filter(pl.col("order_dt") >= cutoff)
        else:
            train_orders = orders
    else:
        train_orders, _, _ = split_orders_by_time(
            orders, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        )

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

    # ---- predict ----
    scores = _predict_scores_batched(
        ranker, feature_table, model_feature_cols, categorical_feature_cols, predict_batch_size,
    )
    scored = feature_table.with_columns(pl.Series("score", scores))

    # Attach category from products
    prod_map = products.select(
        [
            pl.col("productid").cast(pl.Utf8).alias("candidate_product_id"),
            pl.col("category").cast(pl.Utf8),
        ]
    ).unique(subset=["candidate_product_id"])
    scored = scored.join(prod_map, on="candidate_product_id", how="left")

    score_range = scored.select(
        pl.col("score").min().alias("min"),
        pl.col("score").max().alias("max"),
        pl.col("score").mean().alias("mean"),
    ).row(0)
    LOGGER.info("Score stats: min=%.6f max=%.6f mean=%.6f", score_range[0], score_range[1], score_range[2])

    # Check score spread per query
    score_spread = (
        scored.group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.col("score").std().alias("score_std"))
    )
    zero_std = score_spread.filter(
        (pl.col("score_std") == 0) | (pl.col("score_std").is_null())
    ).height
    if score_spread.height > 0:
        LOGGER.info(
            "Queries with zero score spread: %s/%s (%.2f%%)",
            zero_std, score_spread.height, 100.0 * zero_std / score_spread.height,
        )

    # ---- save predictions ----
    final = (
        scored
        .sort(["kiosk_id", "anchor_product_id", "score"], descending=[False, False, True])
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(catalog_top_k)
        .select(["kiosk_id", "anchor_product_id", "candidate_product_id", "category", "score"])
    )
    save_parquet(final, predictions_path)
    LOGGER.info("Saved predictions to %s", predictions_path)

    # ---- popularity fallback ----
    popularity = (
        train_orders
        .group_by("product_id")
        .agg(pl.len().alias("purchase_count"))
        .sort("purchase_count", descending=True)
        .head(catalog_top_k)
        .select(["product_id", "purchase_count"])
        .rename({"product_id": "candidate_product_id"})
        .join(prod_map, on="candidate_product_id", how="left")
    )
    if normalize_popularity:
        pop = popularity.with_columns(pl.col("purchase_count").cast(pl.Float64).log1p().alias("_pop"))
        pop_min = pop.select(pl.col("_pop").min()).item()
        pop_max = pop.select(pl.col("_pop").max()).item()
        score_min, score_max = float(score_range[0]), float(score_range[1])
        if pop_min is None or pop_max is None or pop_min == pop_max:
            pop = pop.with_columns(pl.lit(score_min).alias("score"))
        else:
            pop = pop.with_columns(
                ((pl.col("_pop") - pop_min) / (pop_max - pop_min) * (score_max - score_min) + score_min).alias("score")
            )
        popularity = pop.drop("_pop")
        LOGGER.info("Popularity fallback normalized to model score range.")
    else:
        popularity = popularity.with_columns(pl.col("purchase_count").cast(pl.Float64).alias("score"))

    popularity = popularity.select(["candidate_product_id", "category", "score"])
    save_parquet(popularity, popularity_path)
    LOGGER.info("Saved cold-start popularity fallback to %s", popularity_path)


if __name__ == "__main__":
    main()
