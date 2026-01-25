from __future__ import annotations

from pathlib import Path
from datetime import datetime
import logging

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
from training.src.steps.rank_eval_at_k import (
    hitrate_at_k_by_score,
    ndcg_at_k_by_score,
    positives_at_k_by_score,
    quantity_captured_at_k_by_score,
)

# ==========================
# FEATURE CONFIG
# ==========================

USE_BASE_MBA = True
USE_KIOSK_HISTORY = True
USE_CHANNEL = True

FEATURE_COLS = []

if USE_BASE_MBA:
    FEATURE_COLS += [
        "cooc_count",
        "anchor_count",
        "candidate_count",
        "support",
        "confidence",
        "lift",
    ]

if USE_KIOSK_HISTORY:
    FEATURE_COLS += [
        "kiosk_product_cnt",
        "kiosk_bought_candidate_before",
    ]

if USE_CHANNEL:
    FEATURE_COLS += [
        "channel_Mayorista",
        "channel_Distribuidores",
        "channel_Ruta",
        "channel_Foodservice",
        "channel_Supermercados",
    ]


# ==========================
# Config
# ==========================
ORDERS_PATH = Path("training/data/interim/orders_sample.parquet")
PRODUCTS_PATH = Path("training/data/products_v2.csv")

K_CANDIDATES = 100
K_EVAL = 20

MIN_COOC = 3
MIN_LIFT = 2.0


# ==========================
# Utils
# ==========================
def _make_group_sizes(df_pd: pd.DataFrame) -> np.ndarray:
    return (
        df_pd.groupby(["kiosk_id", "anchor_product_id"], sort=False)
        .size()
        .to_numpy()
    )


def setup_logging():
    logs_dir = Path("training/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_ranker_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Logging to {log_path}")


# ==========================
# Main
# ==========================
def main():
    setup_logging()
    logging.info("Starting train_ranker")

    # ---------- Load orders ----------
    orders = pl.read_parquet(ORDERS_PATH).with_columns(
        pl.col("order_dt").cast(pl.Datetime)
    )

    SPLIT_DATE = pl.datetime(2024, 1, 4)

    train_orders = orders.filter(pl.col("order_dt") < SPLIT_DATE)
    test_orders = orders.filter(pl.col("order_dt") >= SPLIT_DATE)

    logging.info(f"Train orders: {train_orders.shape}")
    logging.info(f"Test orders:  {test_orders.shape}")

    # ---------- Baskets ----------
    baskets_train = build_baskets(train_orders)

    # ---------- Candidates ----------
    candidates = generate_candidates(baskets_train, min_cooc=MIN_COOC)

    topk_candidates = select_top_k_candidates(
        candidates,
        k=K_CANDIDATES,
        min_lift=MIN_LIFT,
    )

    # ---------- Feature table ----------
    feature_table = build_feature_table(baskets_train, topk_candidates)

    products = pl.read_csv(PRODUCTS_PATH, separator=";")
    feature_table = add_product_features(feature_table, products)

    feature_table = add_kiosk_history_features(
        feature_table=feature_table,
        train_orders=train_orders,
    )

    # ---------- Labels ----------
    labeled = build_labels(feature_table, test_orders)

    out_path = Path("training/data/interim/labeled_features_for_ranker.parquet")
    labeled.write_parquet(out_path)
    logging.info(f"[OK] Saved labeled dataset to {out_path}")

    # ---------- Kiosk split ----------
    kiosks = labeled.select("kiosk_id").unique().to_pandas()
    rng = np.random.default_rng(42)
    mask = rng.random(len(kiosks)) < 0.9

    train_kiosks = set(kiosks.loc[mask, "kiosk_id"])
    model_train = labeled.filter(pl.col("kiosk_id").is_in(train_kiosks))
    model_valid = labeled.filter(~pl.col("kiosk_id").is_in(train_kiosks))

    logging.info(f"Model train rows: {model_train.shape}")
    logging.info(f"Model valid rows: {model_valid.shape}")

    # ---------- Columns ----------
    base_cols = [
        "kiosk_id",
        "anchor_product_id",
        "candidate_product_id",
        "label",
        "channel",
        "cooc_count",
        "kiosk_product_cnt",
    ]

    train_pd = model_train.select(base_cols).to_pandas()
    valid_pd = model_valid.select(base_cols).to_pandas()

    # ---------- One-hot channel ----------
    train_pd = pd.get_dummies(train_pd, columns=["channel"])
    valid_pd = pd.get_dummies(valid_pd, columns=["channel"])

    train_pd, valid_pd = train_pd.align(
        valid_pd,
        join="left",
        axis=1,
        fill_value=0,
    )

    feature_cols = [
        c for c in train_pd.columns
        if c not in {
            "kiosk_id",
            "anchor_product_id",
            "candidate_product_id",
            "label",
        }
    ]

    # ---------- Sort & groups ----------
    train_pd = train_pd.sort_values(
        ["kiosk_id", "anchor_product_id"], kind="mergesort"
    )
    valid_pd = valid_pd.sort_values(
        ["kiosk_id", "anchor_product_id"], kind="mergesort"
    )

    group_train = _make_group_sizes(train_pd)
    group_valid = _make_group_sizes(valid_pd)

    X_train = train_pd[feature_cols]
    y_train = train_pd["label"].astype(int)

    X_valid = valid_pd[feature_cols]
    y_valid = valid_pd["label"].astype(int)

    # ---------- Train ----------
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    ranker.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[K_EVAL],
    )

    # ---------- Eval ----------
    valid_pd["score"] = ranker.predict(X_valid)
    valid_scored = pl.from_pandas(valid_pd)

    hitrate = hitrate_at_k_by_score(valid_scored, k=K_EVAL)
    ndcg = ndcg_at_k_by_score(valid_scored, k=K_EVAL)
    positives = positives_at_k_by_score(valid_scored, k=K_EVAL)
    quantity = quantity_captured_at_k_by_score(
        df=valid_scored,
        test_orders=test_orders,
        k=K_EVAL,
    )

    logging.info(f"[FINAL RESULT] HitRate@{K_EVAL} = {hitrate:.4f}")
    logging.info(f"[FINAL RESULT] NDCG@{K_EVAL}    = {ndcg:.4f}")
    logging.info(f"[FINAL RESULT] Positives@{K_EVAL} = {positives:.4f}")
    logging.info(f"[FINAL RESULT] QuantityCaptured@{K_EVAL} = {quantity:.4f}")

    # ---------- Feature importance ----------
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": ranker.feature_importances_,
    }).sort_values("importance", ascending=False)

    logging.info("Feature importance:")
    logging.info(feat_imp)

    model_path = Path("training/models/lgbm_ranker.txt")
    ranker.booster_.save_model(str(model_path))
    logging.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

