from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging
from typing import Tuple

import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb

from training.src.io import load_commerces_csv, load_orders_parquet, load_products_csv
from training.src.paths import DATA_DIR, INTERIM_DIR, LOGS_DIR, MODELS_DIR
from training.src.steps.build_baskets import build_baskets
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.build_labels import build_labels
from training.src.cli import parse_config_args
from training.src.config import FeatureConfig, load_yaml_config
from training.src.features import add_all_features

from training.src.steps.rank_eval_at_k import (
    hitrate_at_k_by_score,
    recall_at_k_by_score,
    ndcg_at_k_by_score,
    positives_at_k_by_score,
    quantity_captured_at_k_by_score,
)

# ======================================================
# CONFIG
# ======================================================
ORDERS_PATH = INTERIM_DIR / "orders_sample.parquet"
PRODUCTS_PATH = DATA_DIR / "products_v2.csv"
COMMERCES_PATH = DATA_DIR / "commerces.csv"
FEATURES_CONFIG_PATH = Path("training/configs/features.yaml")
SCRIPT_CONFIG_PATH = Path("training/configs/train_ranker_lgbm.yaml")

K_CANDIDATES = 100
K_EVAL = 20

MIN_COOC = 3
MIN_LIFT = 2.0

RANDOM_SEED = 42
TRAIN_KIOSK_RATIO = 0.9
MAX_NEG_PER_GROUP = 60

# ======================================================
# FEATURES
# ======================================================
FEATURE_COLS_BASE = [
    "cosine_sim",
    "kiosk_product_cnt",
    "pop_store",
    "kiosk_bought_candidate_before",
    "anchor_kiosk_frequency",
    "cand_is_new_for_kiosk",
    "channel_Mayorista",
    "channel_Ruta",
    "channel_Foodservice",
    "channel_Distribuidores",
    "channel_Supermercados",
]

BASE_COLS = ["kiosk_id", "anchor_product_id", "candidate_product_id", "label"]

# ======================================================
# LOGGING
# ======================================================
def setup_logging():
    logs_dir = LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_ranker_lgbm_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_path}")

# ======================================================
# UTILS
# ======================================================
def _add_query_id(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("kiosk_id").cast(pl.Utf8) + "::" +
         pl.col("anchor_product_id").cast(pl.Utf8)).alias("query_id")
    )

def _filter_good_queries(df: pl.DataFrame) -> pl.DataFrame:
    stats = (
        df.group_by("query_id")
        .agg(
            pl.len().alias("q_size"),
            pl.sum("label").alias("q_pos"),
        )
    )
    good = stats.filter(
        (pl.col("q_size") > 1) & (pl.col("q_pos") > 0)
    ).select("query_id")

    logging.info(
        f"Queries total: {stats.shape[0]}, kept: {good.shape[0]}, "
        f"removed: {stats.shape[0] - good.shape[0]}"
    )
    return df.join(good, on="query_id", how="inner")

def _sample_negatives(df: pl.DataFrame, max_neg: int, seed: int) -> pl.DataFrame:
    if max_neg <= 0:
        return df

    rng = np.random.default_rng(seed)
    pdf = df.select(["query_id"] + BASE_COLS + FEATURE_COLS).to_pandas()

    parts = []
    for _, g in pdf.groupby("query_id", sort=False):
        pos = g[g.label == 1]
        neg = g[g.label == 0]

        if len(neg) > max_neg:
            neg = neg.sample(max_neg, random_state=seed)

        parts.append(pd.concat([pos, neg]))

    return pl.from_pandas(pd.concat(parts, ignore_index=True))

def _shuffle_within_query(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    pdf = df.to_pandas()

    parts = []
    for _, g in pdf.groupby("query_id", sort=False):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        parts.append(pdf.loc[idx])

    return pl.from_pandas(pd.concat(parts, ignore_index=True))

def _to_lgbm_arrays(df: pl.DataFrame):
    pdf = df.select(["query_id"] + BASE_COLS + FEATURE_COLS).to_pandas()
    pdf = pdf.sort_values("query_id", kind="mergesort")

    group = pdf.groupby("query_id", sort=False).size().to_numpy()
    X = pdf[FEATURE_COLS]
    y = pdf["label"].astype(int).to_numpy()

    return X, y, group

# ======================================================
# MAIN
# ======================================================
def main():
    config_path, features_config_path = parse_config_args(
        default_config=SCRIPT_CONFIG_PATH,
        default_features_config=FEATURES_CONFIG_PATH,
        description="Train LightGBM ranker (lgbm script)",
    )
    cfg = load_yaml_config(config_path) if config_path.exists() else {}
    global SPLIT_DATE, MIN_COOC, MIN_LIFT, K_CANDIDATES, K_EVAL, TRAIN_KIOSK_RATIO, MAX_NEG_PER_GROUP, FEATURE_COLS_BASE
    if "split_date" in cfg:
        SPLIT_DATE = datetime.fromisoformat(cfg["split_date"])
    MIN_COOC = int(cfg.get("min_cooc", MIN_COOC))
    MIN_LIFT = float(cfg.get("min_lift", MIN_LIFT))
    K_CANDIDATES = int(cfg.get("k_candidates", K_CANDIDATES))
    K_EVAL = int(cfg.get("k_eval", K_EVAL))
    TRAIN_KIOSK_RATIO = float(cfg.get("train_kiosk_ratio", TRAIN_KIOSK_RATIO))
    MAX_NEG_PER_GROUP = int(cfg.get("max_neg_per_group", MAX_NEG_PER_GROUP))
    if "feature_cols_base" in cfg:
        FEATURE_COLS_BASE = list(cfg["feature_cols_base"])
    setup_logging()
    logging.info("Starting LightGBM ranker training")

    # ---------- Load orders ----------
    orders_path = Path(cfg.get("orders_path", ORDERS_PATH))
    orders = load_orders_parquet(orders_path).with_columns(
        pl.col("order_dt").cast(pl.Datetime)
    )

    SPLIT_DATE = pl.datetime(2024, 1, 4)
    train_orders = orders.filter(pl.col("order_dt") < SPLIT_DATE)
    test_orders = orders.filter(pl.col("order_dt") >= SPLIT_DATE)

    # ---------- Baskets ----------
    baskets_train = build_baskets(train_orders)
    baskets_test = build_baskets(test_orders)

    queries = (
        baskets_test
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .unique()
    )

    # ---------- Candidates ----------
    candidates = generate_candidates(baskets_train, min_cooc=MIN_COOC)
    topk_candidates = select_top_k_candidates(
        candidates, k=K_CANDIDATES, min_lift=MIN_LIFT
    )

    # ---------- Feature table ----------
    feature_table = build_feature_table(
        baskets=baskets_train,
        topk_candidates=topk_candidates,
        queries=queries,
    )

    products_path = Path(cfg.get("products_path", PRODUCTS_PATH))
    commerces_path = Path(cfg.get("commerces_path", COMMERCES_PATH))
    products = load_products_csv(products_path)
    commerces = load_commerces_csv(commerces_path)
    feature_config = (
        FeatureConfig.from_yaml(features_config_path)
        if features_config_path and features_config_path.exists()
        else FeatureConfig()
    )
    feature_table = add_all_features(
        feature_table,
        orders=train_orders,
        products=products,
        commerces=commerces,
        config=feature_config,
    )

    feature_table = feature_table.with_columns(
        pl.col("cooc_count").log1p()
    )

    # ---------- Features ----------
    global FEATURE_COLS
    region_cols = [c for c in feature_table.columns if c.startswith("region_")]
    FEATURE_COLS = FEATURE_COLS_BASE + region_cols

    # save feature schema
    features_path = Path(cfg.get("features_path", MODELS_DIR / "lgbm_features.txt"))
    features_path.parent.mkdir(parents=True, exist_ok=True)
    with open(features_path, "w") as f:
        for c in FEATURE_COLS:
            f.write(c + "\n")
    logging.info(f"Saved feature list to {features_path}")

    # ---------- Labels ----------
    labeled = build_labels(feature_table, test_orders)
    labeled = labeled.with_columns(
        [pl.col(c).fill_null(0) for c in FEATURE_COLS] +
        [pl.col("label").fill_null(0).cast(pl.Int8)]
    )

    labeled = _add_query_id(labeled)

    # ---------- Split by kiosks ----------
    kiosks = labeled.select("kiosk_id").unique().to_series().to_list()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(kiosks)

    cut = int(len(kiosks) * TRAIN_KIOSK_RATIO)
    train_k = set(kiosks[:cut])

    train_df = labeled.filter(pl.col("kiosk_id").is_in(train_k))
    valid_df = labeled.filter(~pl.col("kiosk_id").is_in(train_k))

    train_df = _filter_good_queries(train_df)
    valid_df = _filter_good_queries(valid_df)

    train_df = _sample_negatives(train_df, MAX_NEG_PER_GROUP, RANDOM_SEED)
    train_df = _shuffle_within_query(train_df, RANDOM_SEED)
    valid_df = _shuffle_within_query(valid_df, RANDOM_SEED + 1)

    # ---------- Arrays ----------
    X_train, y_train, g_train = _to_lgbm_arrays(train_df)
    X_valid, y_valid, g_valid = _to_lgbm_arrays(valid_df)

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, group=g_valid)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [K_EVAL],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(20),
        ],
    )

    logging.info(
        f"[LGBM INTERNAL] Best iteration: {booster.best_iteration}, "
        f"best valid ndcg@{K_EVAL}: {booster.best_score['valid'][f'ndcg@{K_EVAL}']:.4f}"
    )

    # ---------- Validation metrics ----------
    valid_pdf = valid_df.select(BASE_COLS + FEATURE_COLS).to_pandas()
    valid_pdf["score"] = booster.predict(valid_pdf[FEATURE_COLS])

    valid_scored = pl.from_pandas(valid_pdf)

    logging.info("===== VALIDATION METRICS (HELD-OUT KIOSKS) =====")
    logging.info(f"HitRate@{K_EVAL}: {hitrate_at_k_by_score(valid_scored, k=K_EVAL):.4f}")
    logging.info(f"Recall@{K_EVAL}: {recall_at_k_by_score(valid_scored, k=K_EVAL):.4f}")
    logging.info(f"NDCG@{K_EVAL}: {ndcg_at_k_by_score(valid_scored, k=K_EVAL):.4f}")
    logging.info(f"Positives@{K_EVAL}: {positives_at_k_by_score(valid_scored, k=K_EVAL):.4f}")
    logging.info(
        f"QuantityCaptured@{K_EVAL}: "
        f"{quantity_captured_at_k_by_score(valid_scored, test_orders, k=K_EVAL):.4f}"
    )

    # ---------- Feature importance ----------

    imp = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    logging.info("===== FEATURE IMPORTANCE (TOP 20, GAIN) =====")
    logging.info("\n" + imp.to_string(index=False))


    # ---------- Save model ----------
    model_path = Path(cfg.get("model_path", MODELS_DIR / "lgbm_ranker.txt"))
    booster.save_model(str(model_path))
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
