from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from training.src.config import FeatureConfig, load_yaml_config
from training.src.features import add_all_features
from training.src.io import load_orders_csv_sample, load_products_csv, load_commerces_csv, save_parquet
from training.src.logging_utils import setup_logging
from training.src.paths import RAW_DIR, INTERIM_DIR, EXTERNAL_DIR, MODELS_DIR
from training.src.steps.build_baskets import build_baskets
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.build_labels import build_labels
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.generate_candidates_hybrid import generate_candidates_hybrid
from training.src.steps.preprocessing import preprocess_orders
from training.src.steps.rank_eval_at_k import (
    hitrate_at_k_by_score,
    recall_at_k_by_score,
    ndcg_at_k_by_score,
    positives_at_k_by_score,
    precision_at_k_by_score,
)
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.split_orders import split_orders_by_time

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingClassifierConfig:
    raw_paths: list[Path]
    n_rows: int
    sample_position: str
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
    eval_ks: list[int]
    eval_extra_neg_per_query: int
    features_config_path: Path | None
    candidate_generator: str
    hybrid_pop_top_k_global: int
    hybrid_pop_top_k_category: int
    lgbm_params: dict
    num_boost_round: int
    early_stopping_rounds: int
    eval_log_path: Path | None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingClassifierConfig":
        data = load_yaml_config(path) if path.exists() else {}
        raw_paths = [Path(p) for p in data.get("raw_paths", [])]
        if not raw_paths:
            raw_paths = [RAW_DIR / "2024-20250001_part_00-001.csv"]
        return cls(
            raw_paths=raw_paths,
            n_rows=int(data.get("n_rows", 500_000)),
            sample_position=str(data.get("sample_position", "head")),
            interim_path=Path(data.get("interim_path", INTERIM_DIR / "orders_sample.parquet")),
            products_path=Path(data.get("products_path", EXTERNAL_DIR / "products_v2.csv")),
            commerces_path=Path(data.get("commerces_path", EXTERNAL_DIR / "commerces.csv")),
            model_path=Path(data.get("model_path", MODELS_DIR / "lgbm_classifier.txt")),
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
            eval_ks=[int(k) for k in data.get("eval_ks", [20])],
            eval_extra_neg_per_query=int(data.get("eval_extra_neg_per_query", 0)),
            features_config_path=Path(data["features_config_path"]) if data.get("features_config_path") else None,
            candidate_generator=str(data.get("candidate_generator", "mba")),
            hybrid_pop_top_k_global=int(data.get("hybrid_pop_top_k_global", 50)),
            hybrid_pop_top_k_category=int(data.get("hybrid_pop_top_k_category", 50)),
            lgbm_params=dict(data.get("lgbm_params", {})),
            num_boost_round=int(data.get("num_boost_round", 1000)),
            early_stopping_rounds=int(data.get("early_stopping_rounds", 50)),
            eval_log_path=Path(data["eval_log_path"]) if data.get("eval_log_path") else None,
        )


def run(config: TrainingClassifierConfig) -> None:
    setup_logging("training_classifier")

    per_file = max(1, config.n_rows // len(config.raw_paths))
    LOGGER.info(
        "Loading raw paths: %s (per_file=%s, sample_position=%s)",
        [str(p) for p in config.raw_paths],
        per_file,
        config.sample_position,
    )
    raw_frames = [
        load_orders_csv_sample(path, n_rows=per_file, sample_position=config.sample_position)
        for path in config.raw_paths
    ]
    raw_orders = pl.concat(raw_frames, how="vertical")
    clean_orders = preprocess_orders(raw_orders)
    save_parquet(clean_orders, config.interim_path)
    LOGGER.info("Saved interim orders to %s", config.interim_path)

    products = load_products_csv(config.products_path)
    commerces = load_commerces_csv(config.commerces_path)

    train_orders, val_orders, test_holdout_orders = split_orders_by_time(
        clean_orders,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    test_holdout_sorted = test_holdout_orders.sort("order_dt")
    test_eval_ratio = config.test_eval_ratio
    if not 0.0 < test_eval_ratio < 1.0:
        raise ValueError("test_eval_ratio must be between 0 and 1.")
    eval_size = int(test_holdout_sorted.height * test_eval_ratio)
    eval_size = max(1, min(eval_size, max(0, test_holdout_sorted.height - 1)))
    label_size = test_holdout_sorted.height - eval_size
    test_eval_orders = test_holdout_sorted.head(eval_size)
    test_label_orders = test_holdout_sorted.tail(label_size)

    def _build_queries(orders: pl.DataFrame) -> pl.DataFrame:
        return (
            build_baskets(orders)
            .select(["kiosk_id", "products"])
            .explode("products")
            .rename({"products": "anchor_product_id"})
            .unique()
        )

    baskets_train = build_baskets(train_orders)
    candidate_generator = config.candidate_generator.lower().strip()
    if candidate_generator == "hybrid":
        topk_candidates = generate_candidates_hybrid(
            baskets_train,
            products=products,
            min_cooc=config.min_cooc,
            min_lift=config.min_lift,
            top_k=config.top_k,
            pop_top_k_global=config.hybrid_pop_top_k_global,
            pop_top_k_category=config.hybrid_pop_top_k_category,
        )
    else:
        candidates = generate_candidates(baskets_train, min_cooc=config.min_cooc)
        topk_candidates = select_top_k_candidates(
            candidates,
            k=config.top_k,
            min_lift=config.min_lift,
        )

    train_queries = _build_queries(train_orders)
    val_queries = _build_queries(val_orders)
    test_queries = _build_queries(test_label_orders)

    feature_table_train = build_feature_table(
        baskets=baskets_train,
        topk_candidates=topk_candidates,
        queries=train_queries,
    )
    feature_table_val = build_feature_table(
        baskets=baskets_train,
        topk_candidates=topk_candidates,
        queries=val_queries,
    )
    feature_table_test = build_feature_table(
        baskets=baskets_train,
        topk_candidates=topk_candidates,
        queries=test_queries,
    )

    feature_config = (
        FeatureConfig.from_yaml(config.features_config_path)
        if config.features_config_path and config.features_config_path.exists()
        else FeatureConfig()
    )

    def _add_features(ft: pl.DataFrame) -> pl.DataFrame:
        return add_all_features(
            ft,
            orders=train_orders,
            products=products,
            commerces=commerces,
            config=feature_config,
        )

    feature_table_train = _add_features(feature_table_train)
    feature_table_val = _add_features(feature_table_val)
    feature_table_test = _add_features(feature_table_test)

    unwanted = [
        "cand_is_new_for_kiosk",
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

    def _drop_unwanted(ft: pl.DataFrame) -> pl.DataFrame:
        present = [c for c in unwanted if c in ft.columns]
        if present:
            LOGGER.info("Dropping unwanted feature columns: %s", present)
            ft = ft.drop(present)
        return ft

    feature_table_train = _drop_unwanted(feature_table_train)
    feature_table_val = _drop_unwanted(feature_table_val)
    feature_table_test = _drop_unwanted(feature_table_test)

    labeled_train = build_labels(
        feature_table_train,
        train_orders,
        window_days=config.label_window_days,
        min_cooc_label=config.min_cooc_label,
    )
    labeled_val = build_labels(
        feature_table_val,
        val_orders,
        window_days=config.label_window_days,
        min_cooc_label=config.min_cooc_label,
    )
    labeled_test = build_labels(
        feature_table_test,
        test_label_orders,
        window_days=config.label_window_days,
        min_cooc_label=config.min_cooc_label,
    )

    labeled_train = labeled_train.with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    labeled_val = labeled_val.with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    labeled_test = labeled_test.with_columns(pl.col("label").fill_null(0).cast(pl.Int8))

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

    def _filter_good_queries(df: pl.DataFrame) -> pl.DataFrame:
        df = _add_query_id(df)
        stats = (
            df.group_by("query_id")
            .agg(
                pl.len().alias("q_size"),
                pl.sum("label").alias("q_pos"),
            )
        )
        good = stats.filter((pl.col("q_size") > 1) & (pl.col("q_pos") > 0)).select("query_id")
        out = df.join(good, on="query_id", how="inner")
        removed = stats.height - good.height
        LOGGER.info("Queries total: %s, kept: %s, removed: %s", stats.height, good.height, removed)
        return out

    def _shuffle_within_query(df: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
        if df.height == 0:
            return df
        df = _add_query_id(df)
        df = df.with_columns(pl.arange(0, pl.len()).shuffle(seed=seed).alias("_rand"))
        return df.sort(["query_id", "_rand"]).drop("_rand")

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
        pdf = df.select(["label"] + feature_cols).to_pandas()
        X = pdf[feature_cols]
        y = pdf["label"].astype(int).to_numpy()
        return X, y

    def _log_label_stats(name: str, df: pl.DataFrame) -> None:
        if df.height == 0:
            LOGGER.info("%s label stats: rows=0 positives=0 ratio=0.0000", name)
            return
        stats = df.select(
            pl.len().alias("rows"),
            pl.col("label").sum().alias("positives"),
        ).row(0)
        rows, positives = stats
        negatives = int(rows) - int(positives)
        ratio = float(positives) / float(rows) if rows else 0.0
        LOGGER.info(
            "%s label stats: rows=%s positives=%s negatives=%s ratio=%.4f",
            name,
            rows,
            positives,
            negatives,
            ratio,
        )

    _log_label_stats("Train", labeled_train)
    _log_label_stats("Val", labeled_val)
    _log_label_stats("TestLabel", labeled_test)

    labeled_train = _filter_good_queries(labeled_train)
    labeled_val = _filter_good_queries(labeled_val)
    labeled_train = _sample_negatives(labeled_train, config.max_neg_per_group, seed=42)
    labeled_train = _shuffle_within_query(labeled_train, seed=42)
    labeled_val = _shuffle_within_query(labeled_val, seed=43)
    _log_label_stats("TrainSampled", labeled_train)
    _log_label_stats("ValSampled", labeled_val)

    X_train, y_train = _to_lgbm_arrays(labeled_train)
    X_val, y_val = _to_lgbm_arrays(labeled_val)
    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError("Empty train/val data; adjust split ratios or data volume.")

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss", "average_precision"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "verbosity": -1,
    }
    if config.lgbm_params:
        params.update(config.lgbm_params)

    evals_result: dict = {}
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ],
    )

    if config.eval_log_path:
        config.eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for dataset, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                for idx, val in enumerate(values, start=1):
                    rows.append(
                        {
                            "iteration": idx,
                            "dataset": dataset,
                            "metric": metric_name,
                            "value": val,
                        }
                    )
        if rows:
            pl.DataFrame(rows).write_csv(config.eval_log_path)
            LOGGER.info("Eval curves saved to %s", config.eval_log_path)

    if booster.best_iteration:
        best_iter = booster.best_iteration
        best = {}
        for dataset, metrics in evals_result.items():
            best[dataset] = {m: metrics[m][best_iter - 1] for m in metrics if len(metrics[m]) >= best_iter}
        LOGGER.info("Best iteration: %s", best_iter)
        LOGGER.info("Best metrics: %s", best)

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
    eval_queries = _build_queries(test_eval_orders)
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
    eval_pdf = eval_labeled.select(
        ["kiosk_id", "anchor_product_id", "candidate_product_id", "label"] + feature_cols
    ).to_pandas()
    eval_pdf["score"] = booster.predict(eval_pdf[feature_cols])

    # Binary metrics
    y_true = eval_pdf["label"].astype(int).to_numpy()
    y_score = eval_pdf["score"].to_numpy()
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        LOGGER.info("[TEST] AUC: %.4f", roc_auc_score(y_true, y_score))
        LOGGER.info("[TEST] PR-AUC: %.4f", average_precision_score(y_true, y_score))
        LOGGER.info("[TEST] LogLoss: %.4f", log_loss(y_true, y_score, labels=[0, 1]))
    else:
        LOGGER.warning("[TEST] AUC/PR-AUC not computed (only one class present).")

    eval_scored = pl.from_pandas(eval_pdf)
    for k in sorted({k for k in config.eval_ks if k > 0}):
        LOGGER.info("[TEST] HitRate@%s: %.4f", k, hitrate_at_k_by_score(eval_scored, k=k))
        LOGGER.info("[TEST] Recall@%s: %.4f", k, recall_at_k_by_score(eval_scored, k=k))
        LOGGER.info("[TEST] NDCG@%s: %.4f", k, ndcg_at_k_by_score(eval_scored, k=k))
        LOGGER.info("[TEST] Precision@%s: %.4f", k, precision_at_k_by_score(eval_scored, k=k))
        LOGGER.info("[TEST] Positives@%s: %.4f", k, positives_at_k_by_score(eval_scored, k=k))

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(config.model_path))
    LOGGER.info("Model saved to %s", config.model_path)

    feature_path = config.model_path.with_suffix(".features.json")
    feature_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Feature list saved to %s", feature_path)
