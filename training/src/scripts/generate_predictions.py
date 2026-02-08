from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import polars as pl

from training.src.config import FeatureConfig, load_yaml_config
from training.src.features import add_all_features
from training.src.io import load_orders_parquet, load_products_csv, load_commerces_csv, save_parquet
from training.src.logging_utils import setup_logging
from training.src.paths import EXTERNAL_DIR, INTERIM_DIR, MODELS_DIR
from training.src.steps.build_baskets import build_baskets
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.generate_candidates_hybrid import generate_candidates_hybrid
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.split_orders import split_orders_by_time


LOGGER = logging.getLogger(__name__)


def _load_feature_list(model_path: Path, ranker: lgb.Booster) -> list[str]:
    names = ranker.feature_name()
    if names:
        return list(names)
    feature_path = model_path.with_suffix(".features.json")
    if feature_path.exists():
        return json.loads(feature_path.read_text(encoding="utf-8"))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions.parquet from a trained model")
    parser.add_argument("--config", default="training/configs/generate_predictions.yaml")
    args = parser.parse_args()

    setup_logging("generate_predictions")

    cfg = load_yaml_config(Path(args.config))
    orders_path = Path(cfg.get("orders_path", INTERIM_DIR / "orders_sample.parquet"))
    products_path = Path(cfg.get("products_path", EXTERNAL_DIR / "products_v2.csv"))
    commerces_path = Path(cfg.get("commerces_path", EXTERNAL_DIR / "commerces.csv"))
    model_path = Path(cfg.get("model_path", MODELS_DIR / "lgbm_ranker.txt"))
    features_config_path = cfg.get("features_config_path", "")

    predictions_path = Path(cfg.get("predictions_path", INTERIM_DIR / "predictions.parquet"))
    popularity_path = Path(cfg.get("popularity_path", INTERIM_DIR / "popularity_fallback.parquet"))

    train_ratio = float(cfg.get("train_ratio", 0.8))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    test_ratio = float(cfg.get("test_ratio", 0.1))
    inference_last_n_days = int(cfg.get("inference_last_n_days", 0))

    min_cooc = int(cfg.get("min_cooc", 3))
    min_lift = float(cfg.get("min_lift", 2.0))
    top_k_candidates = int(cfg.get("top_k_candidates", 250))
    catalog_top_k = int(cfg.get("catalog_top_k", 100))
    candidate_generator = str(cfg.get("candidate_generator", "mba")).lower().strip()
    hybrid_pop_top_k_global = int(cfg.get("hybrid_pop_top_k_global", 50))
    hybrid_pop_top_k_category = int(cfg.get("hybrid_pop_top_k_category", 50))

    orders = load_orders_parquet(orders_path)
    products = load_products_csv(products_path)
    commerces = load_commerces_csv(commerces_path)

    if inference_last_n_days and "order_dt" in orders.columns:
        max_dt = orders.select(pl.col("order_dt").max()).item()
        if max_dt is not None:
            cutoff = max_dt - pl.duration(days=inference_last_n_days)
            train_orders = orders.filter(pl.col("order_dt") >= cutoff)
        else:
            train_orders = orders
    else:
        train_orders, _, _ = split_orders_by_time(
            orders,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    baskets_train = build_baskets(train_orders)
    if candidate_generator == "hybrid":
        topk_candidates = generate_candidates_hybrid(
            baskets_train,
            products=products,
            min_cooc=min_cooc,
            min_lift=min_lift,
            top_k=top_k_candidates,
            pop_top_k_global=hybrid_pop_top_k_global,
            pop_top_k_category=hybrid_pop_top_k_category,
        )
    else:
        candidates = generate_candidates(baskets_train, min_cooc=min_cooc)
        topk_candidates = select_top_k_candidates(
            candidates,
            k=top_k_candidates,
            min_lift=min_lift,
        )

    feature_table = build_feature_table(
        baskets=baskets_train,
        topk_candidates=topk_candidates,
    )

    feature_config = (
        FeatureConfig.from_yaml(Path(features_config_path))
        if features_config_path
        else FeatureConfig()
    )
    feature_table = add_all_features(
        feature_table,
        orders=train_orders,
        products=products,
        commerces=commerces,
        config=feature_config,
    )

    ranker = lgb.Booster(model_file=str(model_path))
    model_feature_cols = _load_feature_list(model_path, ranker)
    if not model_feature_cols:
        raise ValueError("Model feature list is empty; retrain to persist features.")

    missing_cols = [col for col in model_feature_cols if col not in feature_table.columns]
    if missing_cols:
        LOGGER.warning("Missing features in inference: %s. Adding zeros (this may impact predictions!)", missing_cols)
        for col in missing_cols:
            feature_table = feature_table.with_columns(pl.lit(0).alias(col))
    
    feature_table = feature_table.with_columns([pl.col(c).fill_null(0) for c in model_feature_cols])

    feature_max_abs = feature_table.select(
        [pl.col(c).abs().max().alias(c) for c in model_feature_cols]
    ).row(0)
    zero_only = [
        name for name, max_abs in zip(model_feature_cols, feature_max_abs)
        if max_abs == 0 or max_abs is None
    ]
    if zero_only:
        LOGGER.warning(
            "Zero-only features in inference (%s): %s",
            len(zero_only),
            zero_only[:10],
        )

    scores = ranker.predict(feature_table.select(model_feature_cols).to_pandas())
    scored = feature_table.with_columns(pl.Series("score", scores))
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

    score_spread = (
        scored
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(pl.col("score").std().alias("score_std"))
    )
    zero_std = score_spread.filter(
        (pl.col("score_std") == 0) | (pl.col("score_std").is_null())
    ).height
    if score_spread.height > 0:
        LOGGER.info(
            "Queries with zero score spread: %s/%s (%.2f%%)",
            zero_std,
            score_spread.height,
            100.0 * zero_std / score_spread.height,
        )

    final = (
        scored
        .sort(["kiosk_id", "anchor_product_id", "score"], descending=[False, False, True])
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(catalog_top_k)
        .select(["kiosk_id", "anchor_product_id", "candidate_product_id", "category", "score"])
    )

    save_parquet(final, predictions_path)
    LOGGER.info("Saved predictions to %s", predictions_path)

    popularity = (
        train_orders
        .group_by("product_id")
        .agg(pl.len().alias("purchase_count"))
        .sort("purchase_count", descending=True)
        .head(catalog_top_k)
        .select(["product_id", "purchase_count"])
        .rename({"product_id": "candidate_product_id"})
        .with_columns(pl.col("purchase_count").cast(pl.Float64).alias("score"))
        .join(prod_map, on="candidate_product_id", how="left")
        .select(["candidate_product_id", "category", "score"])
    )
    save_parquet(popularity, popularity_path)
    LOGGER.info("Saved cold-start popularity fallback to %s", popularity_path)


if __name__ == "__main__":
    main()
