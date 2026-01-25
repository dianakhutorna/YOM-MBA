from __future__ import annotations

from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple

import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb

from training.src.steps.build_baskets import build_baskets
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.build_labels import build_labels
from training.src.steps.add_product_features import add_product_features
from training.src.steps.add_kiosk_features import add_kiosk_history_features
from training.src.steps.encode_categorical_features import encode_channel_one_hot
from training.src.steps.rank_eval_at_k import (
    hitrate_at_k_by_score,
    ndcg_at_k_by_score,
    positives_at_k_by_score,
    quantity_captured_at_k_by_score,
)
from training.src.steps.add_popularity_features import add_popularity_features


# ======================================================
# CONFIG
# ======================================================
ORDERS_PATH = Path("training/data/interim/orders_sample.parquet")
PRODUCTS_PATH = Path("training/data/products_v2.csv")
COMMERCES_PATH = Path("training/data/commerces.csv")

commerces = pl.read_csv(COMMERCES_PATH, separator=";")


K_CANDIDATES = 100
K_EVAL = 20

MIN_COOC = 3
MIN_LIFT = 2.0

RANDOM_SEED = 42
TRAIN_KIOSK_RATIO = 0.9

# Ограничение числа негативов на группу (ускоряет и стабилизирует)
# Оставляем все positives, а negatives семплим до MAX_NEG_PER_GROUP
MAX_NEG_PER_GROUP = 60

# ======================================================
# FEATURE SELECTION (ТОЛЬКО ЗДЕСЬ!)
# ======================================================
FEATURE_COLS = [
    # --- MBA ---
    "cooc_count",
    # "confidence",
    # "lift",

    # "cosine_sim",

    # --- kiosk history ---
    "kiosk_product_cnt",

    # popularity
    #"pop_global",
    "pop_channel",
    "pop_region",
    "pop_store",

    # --- channel one-hot ---
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
    logs_dir = Path("training/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_ranker_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_path}")


# ======================================================
# RANKING UTILS
# ======================================================
def _add_query_id(df: pl.DataFrame) -> pl.DataFrame:
    # Строковый query_id (устойчиво и удобно)
    return df.with_columns(
        (pl.col("kiosk_id").cast(pl.Utf8) + pl.lit("::") + pl.col("anchor_product_id").cast(pl.Utf8)).alias("query_id")
    )


def _filter_good_queries(df: pl.DataFrame) -> pl.DataFrame:
    """
    Убираем группы, где:
    - размер 1 (ранжировать нечего)
    - нет ни одного positive (LambdaRank не получает сигнал)
    """
    stats = (
        df.group_by("query_id")
        .agg(
            pl.len().alias("q_size"),
            pl.sum("label").alias("q_pos"),
        )
    )

    good = stats.filter((pl.col("q_size") > 1) & (pl.col("q_pos") > 0)).select("query_id")
    out = df.join(good, on="query_id", how="inner")

    removed = stats.shape[0] - good.shape[0]
    logging.info(f"Queries total: {stats.shape[0]}, kept: {good.shape[0]}, removed: {removed}")
    return out


def _sample_negatives_within_query(df: pl.DataFrame, max_neg_per_group: int, seed: int) -> pl.DataFrame:
    """
    Оставляем все positives.
    Negatives (label=0) семплим в каждой группе до max_neg_per_group.
    Делает обучение быстрее и уменьшает перекос.
    """
    if max_neg_per_group is None or max_neg_per_group <= 0:
        return df

    rng = np.random.default_rng(seed)

    # Делать это чисто в Polars можно, но проще и надежнее через group-wise pandas,
    # потому что размер большой, а логика семплинга точная.
    pdf = df.select(["query_id"] + BASE_COLS + FEATURE_COLS).to_pandas()

    kept_parts = []
    for qid, g in pdf.groupby("query_id", sort=False):
        pos = g[g["label"] == 1]
        neg = g[g["label"] == 0]

        if len(neg) > max_neg_per_group:
            take_idx = rng.choice(neg.index.to_numpy(), size=max_neg_per_group, replace=False)
            neg = neg.loc[take_idx]

        kept_parts.append(pd.concat([pos, neg], axis=0))

    out_pdf = pd.concat(kept_parts, axis=0, ignore_index=True)
    return pl.from_pandas(out_pdf)


def _shuffle_within_query(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """
    Важно: если строки внутри группы уже “по MBA” отсортированы,
    то даже при константных score метрики будут “красивые”.
    Поэтому перед обучением/валидацией шеффлим порядок кандидатов внутри query.
    """
    rng = np.random.default_rng(seed)
    pdf = df.to_pandas()

    parts = []
    for qid, g in pdf.groupby("query_id", sort=False):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        parts.append(pdf.loc[idx])

    out = pd.concat(parts, axis=0, ignore_index=True)
    return pl.from_pandas(out)


def _to_lgbm_ranking_arrays(df: pl.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Возвращаем:
    - X (pd.DataFrame)
    - y (np.ndarray int)
    - group (np.ndarray query sizes)
    """
    # Гарантируем порядок: все строки одной query подряд
    pdf = df.select(["query_id"] + BASE_COLS + FEATURE_COLS).to_pandas()
    pdf = pdf.sort_values(["query_id"], kind="mergesort")

    group = pdf.groupby("query_id", sort=False).size().to_numpy()

    X = pdf[FEATURE_COLS]
    y = pdf["label"].astype(int).to_numpy()
    return X, y, group


# ======================================================
# MAIN
# ======================================================
def main():
    setup_logging()
    logging.info("Starting train_ranker (group-aware)")

    # ---------- Load orders ----------
    orders = pl.read_parquet(ORDERS_PATH).with_columns(pl.col("order_dt").cast(pl.Datetime))

    SPLIT_DATE = pl.datetime(2024, 1, 4)
    train_orders = orders.filter(pl.col("order_dt") < SPLIT_DATE)
    test_orders = orders.filter(pl.col("order_dt") >= SPLIT_DATE)

    logging.info(f"Train orders: {train_orders.shape}")
    logging.info(f"Test orders:  {test_orders.shape}")

    # ---------- Baskets ----------
    baskets_train = build_baskets(train_orders)

    # ---------- Test queries (REAL anchor situations) ----------
    baskets_test = build_baskets(test_orders)

    test_queries = (
        baskets_test
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .unique()
    )

    logging.info(f"Test queries (kiosk, anchor): {test_queries.shape}")


    # ---------- Candidates ----------
    candidates = generate_candidates(baskets_train, min_cooc=MIN_COOC)
    topk_candidates = select_top_k_candidates(
        candidates,
        k=K_CANDIDATES,
        min_lift=MIN_LIFT,
    )
    logging.info(f"Top-K candidates shape: {topk_candidates.shape}")

    # ---------- Feature table ----------
    feature_table = build_feature_table(
    baskets=baskets_train,
    topk_candidates=topk_candidates,
    queries=test_queries,   # <<< ВОТ КЛЮЧЕВОЕ
    )

    logging.info(f"Feature table shape: {feature_table.shape}")

    products = pl.read_csv(PRODUCTS_PATH, separator=";")
    feature_table = add_product_features(feature_table, products)
    feature_table = add_kiosk_history_features(feature_table=feature_table, train_orders=train_orders)

    feature_table = add_popularity_features(
        feature_table=feature_table,
        orders=train_orders,
        commerces=commerces,
    )


    feature_table = encode_channel_one_hot(feature_table)

    # Логарифмируем популярности
    feature_table = feature_table.with_columns(pl.col("pop_global").log1p().alias("pop_global"))


    # ---------- Labels ----------
    labeled = build_labels(feature_table, test_orders)

    # ---------- Fill missing features ----------
    for c in FEATURE_COLS:
        if c not in labeled.columns:
            logging.warning(f"Feature missing, filling with 0: {c}")
            labeled = labeled.with_columns(pl.lit(0).alias(c))

    labeled = labeled.with_columns(
        [pl.col(c).fill_null(0) for c in FEATURE_COLS] +
        [pl.col("label").fill_null(0).cast(pl.Int8)]
    )

    # ---------- Add query_id ----------
    labeled = _add_query_id(labeled)

    # ---------- Split by kiosks ----------
    kiosks = labeled.select("kiosk_id").unique().to_pandas()["kiosk_id"].tolist()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(kiosks)

    cut = int(len(kiosks) * TRAIN_KIOSK_RATIO)
    train_kiosks = set(kiosks[:cut])

    train_df = labeled.filter(pl.col("kiosk_id").is_in(train_kiosks))
    valid_df = labeled.filter(~pl.col("kiosk_id").is_in(train_kiosks))

    logging.info(f"Raw train rows: {train_df.shape}, raw valid rows: {valid_df.shape}")

    # ---------- Filter good queries ----------
    train_df = _filter_good_queries(train_df)
    valid_df = _filter_good_queries(valid_df)

    # ---------- Negative sampling (train only) ----------
    train_df = _sample_negatives_within_query(train_df, MAX_NEG_PER_GROUP, seed=RANDOM_SEED)

    # ---------- Shuffle within query (train + valid) ----------
    train_df = _shuffle_within_query(train_df, seed=RANDOM_SEED)
    valid_df = _shuffle_within_query(valid_df, seed=RANDOM_SEED + 1)

    logging.info(f"Final train rows: {train_df.shape}, final valid rows: {valid_df.shape}")

    # ---------- Prepare ranking arrays ----------
    X_train, y_train, group_train = _to_lgbm_ranking_arrays(train_df)
    X_valid, y_valid, group_valid = _to_lgbm_ranking_arrays(valid_df)

    logging.info(f"Train queries: {len(group_train)}, avg q size: {group_train.mean():.2f}")
    logging.info(f"Valid queries: {len(group_valid)}, avg q size: {group_valid.mean():.2f}")
    logging.info(f"Train positives ratio: {y_train.mean():.6f}")
    logging.info(f"Valid positives ratio: {y_valid.mean():.6f}")

    # ---------- Train (native lightgbm for ranking) ----------
    train_set = lgb.Dataset(X_train, label=y_train, group=group_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, group=group_valid, reference=train_set)

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
        params=params,
        train_set=train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=20),
        ],
    )

    # ---------- Evaluation (by score) ----------
    valid_pdf = valid_df.select(["query_id"] + BASE_COLS + FEATURE_COLS).to_pandas()
    valid_pdf["score"] = booster.predict(valid_pdf[FEATURE_COLS])

    # Важно: rank_eval_at_k сортирует по score. Если есть ties, порядок будет "как есть".
    # Мы уже шеффлили внутри query, поэтому ties не дают "MBA-порядок" как подсказку.
    valid_scored = pl.from_pandas(valid_pdf.drop(columns=["query_id"]))

    logging.info(f"[FINAL RESULT] HitRate@{K_EVAL} = {hitrate_at_k_by_score(valid_scored, k=K_EVAL):.4f}")
    logging.info(f"[FINAL RESULT] NDCG@{K_EVAL}    = {ndcg_at_k_by_score(valid_scored, k=K_EVAL):.4f}")
    logging.info(f"[FINAL RESULT] Positives@{K_EVAL} = {positives_at_k_by_score(valid_scored, k=K_EVAL):.4f}")
    logging.info(
        f"[FINAL RESULT] QuantityCaptured@{K_EVAL} = "
        f"{quantity_captured_at_k_by_score(valid_scored, test_orders, k=K_EVAL):.4f}"
    )

    # ---------- Feature importance ----------
    imp = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": booster.feature_importance(importance_type="gain")}
    ).sort_values("importance", ascending=False)
    logging.info("Feature importance (gain):")
    logging.info(imp)

    # ---------- Save model ----------
    model_path = Path("training/models/lgbm_ranker.txt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    logging.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
