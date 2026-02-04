from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import json

import lightgbm as lgb
import polars as pl
import pandas as pd

from training.src.config import load_yaml_config
from training.src.features import add_all_features
from training.src.io import load_orders_csv_sample, load_products_csv, load_commerces_csv, save_parquet
from training.src.paths import RAW_DIR, INTERIM_DIR, EXTERNAL_DIR, MODELS_DIR
from training.src.steps.build_baskets import build_baskets
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.build_labels import build_labels
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.preprocessing import preprocess_orders
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.split_orders import split_orders_by_time
from training.src.config import FeatureConfig
from training.src.steps.rank_eval_at_k import (
    hitrate_at_k_by_score,
    recall_at_k_by_score,
    ndcg_at_k_by_score,
    positives_at_k_by_score,
    quantity_captured_at_k_by_score,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingPipelineConfig:
    raw_paths: list[Path]
    n_rows: int
    interim_path: Path
    products_path: Path
    commerces_path: Path
    model_path: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    test_eval_ratio: float
    min_cooc: int
    min_lift: float
    top_k: int
    label_window_days: int | None
    min_cooc_label: int
    max_neg_per_group: int
    features_config_path: Path | None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingPipelineConfig":
        data = load_yaml_config(path) if path.exists() else {}
        raw_paths = [Path(p) for p in data.get("raw_paths", [])]
        if not raw_paths:
            raw_paths = [RAW_DIR / "2024-20250001_part_00-001.csv"]
        return cls(
            raw_paths=raw_paths,
            n_rows=int(data.get("n_rows", 500_000)),
            interim_path=Path(data.get("interim_path", INTERIM_DIR / "orders_sample.parquet")),
            products_path=Path(data.get("products_path", EXTERNAL_DIR / "products_v2.csv")),
            commerces_path=Path(data.get("commerces_path", EXTERNAL_DIR / "commerces.csv")),
            model_path=Path(data.get("model_path", MODELS_DIR / "lgbm_ranker.txt")),
            train_ratio=float(data.get("train_ratio", 0.8)),
            val_ratio=float(data.get("val_ratio", 0.1)),
            test_ratio=float(data.get("test_ratio", 0.1)),
            test_eval_ratio=float(data.get("test_eval_ratio", 0.5)),
            min_cooc=int(data.get("min_cooc", 3)),
            min_lift=float(data.get("min_lift", 2.0)),
            top_k=int(data.get("top_k", 100)),
            label_window_days=data.get("label_window_days", 7),
            min_cooc_label=int(data.get("min_cooc_label", 1)),
            max_neg_per_group=int(data.get("max_neg_per_group", 60)),
            features_config_path=Path(data["features_config_path"]) if data.get("features_config_path") else None,
        )


def run(config: TrainingPipelineConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    per_file = max(1, config.n_rows // len(config.raw_paths))
    raw_frames = [
        load_orders_csv_sample(path, n_rows=per_file)
        for path in config.raw_paths
    ]
    raw_orders = pl.concat(raw_frames, how="vertical")
    if "order_dt" in raw_orders.columns:
        dt_stats = raw_orders.select(
            pl.col("order_dt").min().alias("min_dt"),
            pl.col("order_dt").max().alias("max_dt"),
        ).row(0)
        LOGGER.info("Raw orders date range: %s to %s", dt_stats[0], dt_stats[1])
    clean_orders = preprocess_orders(raw_orders)
    save_parquet(clean_orders, config.interim_path)
    LOGGER.info("Saved interim orders to %s", config.interim_path)

    train_orders, val_orders, test_holdout_orders = split_orders_by_time(
        clean_orders,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    for name, df in (("Train", train_orders), ("Val", val_orders), ("TestHoldout", test_holdout_orders)):
        if df.height == 0:
            LOGGER.info("%s orders: rows=0", name)
            continue
        dt_stats = df.select(
            pl.col("order_dt").min().alias("min_dt"),
            pl.col("order_dt").max().alias("max_dt"),
        ).row(0)
        LOGGER.info("%s orders date range: %s to %s (rows=%s)", name, dt_stats[0], dt_stats[1], df.height)

    test_holdout_sorted = test_holdout_orders.sort("order_dt")
    test_eval_ratio = config.test_eval_ratio
    if not 0.0 < test_eval_ratio < 1.0:
        raise ValueError("test_eval_ratio must be between 0 and 1.")
    eval_size = int(test_holdout_sorted.height * test_eval_ratio)
    eval_size = max(1, min(eval_size, max(0, test_holdout_sorted.height - 1)))
    label_size = test_holdout_sorted.height - eval_size
    test_label_orders = test_holdout_sorted.head(label_size)
    test_eval_orders = test_holdout_sorted.tail(eval_size)
    for name, df in (("TestLabel", test_label_orders), ("TestEval", test_eval_orders)):
        if df.height == 0:
            LOGGER.info("%s orders: rows=0", name)
            continue
        dt_stats = df.select(
            pl.col("order_dt").min().alias("min_dt"),
            pl.col("order_dt").max().alias("max_dt"),
        ).row(0)
        LOGGER.info("%s orders date range: %s to %s (rows=%s)", name, dt_stats[0], dt_stats[1], df.height)

    baskets_train = build_baskets(train_orders)
    candidates = generate_candidates(baskets_train, min_cooc=config.min_cooc)
    topk_candidates = select_top_k_candidates(
        candidates,
        k=config.top_k,
        min_lift=config.min_lift,
    )

    feature_table = build_feature_table(
        baskets=baskets_train,
        topk_candidates=topk_candidates,
    )

    products = load_products_csv(config.products_path)
    commerces = load_commerces_csv(config.commerces_path)
    feature_config = (
        FeatureConfig.from_yaml(config.features_config_path)
        if config.features_config_path and config.features_config_path.exists()
        else FeatureConfig()
    )
    feature_table = add_all_features(
        feature_table,
        orders=train_orders,
        products=products,
        commerces=commerces,
        config=feature_config,
    )

    # -----------------------------
    # Drop unwanted features (keep only minimal set)
    # -----------------------------
    unwanted = [
        "anchor_kiosk_frequency",
        "kiosk_bought_candidate_before",
        "candidate_count",
        "support",
        "lift",
        "confidence",
        "same_category",
        "cooc_count",
        "anchor_count",
    ]
    present_unwanted = [c for c in unwanted if c in feature_table.columns]
    if present_unwanted:
        LOGGER.info("Dropping unwanted feature columns: %s", present_unwanted)
        feature_table = feature_table.drop(present_unwanted)

    labeled_train = build_labels(
        feature_table,
        train_orders,
        window_days=config.label_window_days,
        min_cooc_label=config.min_cooc_label,
    )
    labeled_val = build_labels(
        feature_table,
        val_orders,
        window_days=config.label_window_days,
        min_cooc_label=config.min_cooc_label,
    )
    labeled_test = build_labels(
        feature_table,
        test_label_orders,
        window_days=config.label_window_days,
        min_cooc_label=config.min_cooc_label,
    )

    for col in ("label",):
        labeled_train = labeled_train.with_columns(pl.col(col).fill_null(0).cast(pl.Int8))
        labeled_val = labeled_val.with_columns(pl.col(col).fill_null(0).cast(pl.Int8))
        labeled_test = labeled_test.with_columns(pl.col(col).fill_null(0).cast(pl.Int8))

    non_feature_cols = {"kiosk_id", "anchor_product_id", "candidate_product_id", "label"}
    numeric_dtypes = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Boolean,
    }
    feature_cols = [
        c for c, dtype in labeled_train.schema.items()
        if c not in non_feature_cols and dtype in numeric_dtypes
    ]
    feature_cols = sorted(feature_cols)

    def _fill_missing_features(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([pl.col(c).fill_null(0) for c in feature_cols])

    labeled_train = _fill_missing_features(labeled_train)
    labeled_val = _fill_missing_features(labeled_val)
    labeled_test = _fill_missing_features(labeled_test)

    def _add_query_id(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            (pl.col("kiosk_id").cast(pl.Utf8) + pl.lit("::") + pl.col("anchor_product_id").cast(pl.Utf8))
            .alias("query_id")
        )

    def _sample_negatives(df: pl.DataFrame, max_neg_per_group: int, seed: int = 42) -> pl.DataFrame:
        if max_neg_per_group <= 0:
            return df
        df = _add_query_id(df)
        if df.height == 0:
            return df
        cols = df.columns
        pos = df.filter(pl.col("label") == 1)
        neg = df.filter(pl.col("label") == 0)
        if neg.height == 0:
            return df
        neg = (
            neg
            .with_columns(pl.arange(0, pl.len()).shuffle(seed=seed).alias("_rand"))
            .sort(["query_id", "_rand"])
            .group_by("query_id")
            .head(max_neg_per_group)
            .drop("_rand")
        )
        return pl.concat([pos.select(cols), neg.select(cols)], how="vertical")

    def _to_lgbm_arrays(df: pl.DataFrame):
        pdf = _add_query_id(df).select(["query_id", "label"] + feature_cols).to_pandas()
        pdf = pdf.sort_values("query_id", kind="mergesort")
        group = pdf.groupby("query_id", sort=False).size().to_numpy()
        X = pdf[feature_cols]
        y = pdf["label"].astype(int).to_numpy()
        return X, y, group

    def _log_label_stats(name: str, df: pl.DataFrame) -> None:
        if df.height == 0:
            LOGGER.info("%s label stats: rows=0 positives=0 ratio=0.0000 queries=0", name)
            return
        stats = df.select(
            pl.len().alias("rows"),
            pl.col("label").sum().alias("positives"),
            pl.struct(["kiosk_id", "anchor_product_id"]).n_unique().alias("queries"),
        ).row(0)
        rows, positives, queries = stats
        negatives = int(rows) - int(positives)
        ratio = float(positives) / float(rows) if rows else 0.0
        LOGGER.info(
            "%s label stats: rows=%s positives=%s negatives=%s ratio=%.4f queries=%s",
            name,
            rows,
            positives,
            negatives,
            ratio,
            queries,
        )

    _log_label_stats("Train", labeled_train)
    _log_label_stats("Val", labeled_val)
    _log_label_stats("TestLabel", labeled_test)

    labeled_train = _sample_negatives(labeled_train, config.max_neg_per_group, seed=42)
    labeled_val = _sample_negatives(labeled_val, config.max_neg_per_group, seed=43)
    _log_label_stats("TrainSampled", labeled_train)
    _log_label_stats("ValSampled", labeled_val)

    X_train, y_train, g_train = _to_lgbm_arrays(labeled_train)
    X_val, y_val, g_val = _to_lgbm_arrays(labeled_val)
    if len(g_train) == 0 or len(g_val) == 0:
        raise ValueError("Empty train/val groups; adjust split ratios or data volume.")

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train)
    valid_set = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_set)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    }

    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=50)],
    )

    # ---------- Feature importance ----------
    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    LOGGER.info("Feature importance (gain):")
    LOGGER.info("\n" + imp_df.to_string(index=False))

    # ---------- Offline evaluation on test_eval ----------
    eval_queries = (
        build_baskets(test_eval_orders)
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .unique()
    )
    eval_feature_table = build_feature_table(
        baskets=baskets_train,
        topk_candidates=topk_candidates,
        queries=eval_queries,
    )
    eval_feature_table = add_all_features(
        eval_feature_table,
        orders=train_orders,
        products=products,
        commerces=commerces,
        config=feature_config,
    )
    eval_labeled = build_labels(
        eval_feature_table,
        test_label_orders,
        window_days=config.label_window_days,
        min_cooc_label=config.min_cooc_label,
    )
    for c in feature_cols:
        if c not in eval_labeled.columns:
            eval_labeled = eval_labeled.with_columns(pl.lit(0).alias(c))
    eval_labeled = eval_labeled.with_columns(
        [pl.col(c).fill_null(0) for c in feature_cols] +
        [pl.col("label").fill_null(0).cast(pl.Int8)]
    )
    eval_pdf = eval_labeled.select(["kiosk_id", "anchor_product_id", "candidate_product_id", "label"] + feature_cols).to_pandas()
    eval_pdf["score"] = booster.predict(eval_pdf[feature_cols])
    eval_scored = pl.from_pandas(eval_pdf)

    _log_label_stats("TestEval", eval_labeled)
    LOGGER.info("[TEST] HitRate@20: %.4f", hitrate_at_k_by_score(eval_scored, k=20))
    LOGGER.info("[TEST] Recall@20: %.4f", recall_at_k_by_score(eval_scored, k=20))
    LOGGER.info("[TEST] NDCG@20: %.4f", ndcg_at_k_by_score(eval_scored, k=20))
    LOGGER.info("[TEST] Positives@20: %.4f", positives_at_k_by_score(eval_scored, k=20))
    LOGGER.info("[TEST] QuantityCaptured@20: %.4f", quantity_captured_at_k_by_score(eval_scored, test_eval_orders, k=20))

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(config.model_path))
    LOGGER.info("Model saved to %s", config.model_path)

    feature_path = config.model_path.with_suffix(".features.json")
    feature_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Feature list saved to %s", feature_path)
