from __future__ import annotations

from pathlib import Path
from datetime import datetime

import polars as pl

from training.src.io import load_commerces_csv, load_orders_parquet, load_products_csv
from training.src.paths import DATA_DIR, INTERIM_DIR
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
    ndcg_at_k_by_score,
)

# ==========================
# Config
# ==========================
ORDERS_PATH = INTERIM_DIR / "orders_sample.parquet"
PRODUCTS_PATH = DATA_DIR / "products_v2.csv"
COMMERCES_PATH = DATA_DIR / "commerces.csv"
FEATURES_CONFIG_PATH = Path("training/configs/features.yaml")
SCRIPT_CONFIG_PATH = Path("training/configs/run_baseline.yaml")

SPLIT_DATE = pl.datetime(2024, 1, 4)

K_CANDIDATES = 100
K_EVAL = 20

MIN_COOC = 3
MIN_LIFT = 2.0


def main():
    config_path, features_path = parse_config_args(
        default_config=SCRIPT_CONFIG_PATH,
        default_features_config=FEATURES_CONFIG_PATH,
        description="Run MBA baseline",
    )
    cfg = load_yaml_config(config_path) if config_path.exists() else {}
    global SPLIT_DATE, MIN_COOC, MIN_LIFT, K_CANDIDATES, K_EVAL
    if "split_date" in cfg:
        SPLIT_DATE = datetime.fromisoformat(cfg["split_date"])
    MIN_COOC = int(cfg.get("min_cooc", MIN_COOC))
    MIN_LIFT = float(cfg.get("min_lift", MIN_LIFT))
    K_CANDIDATES = int(cfg.get("k_candidates", K_CANDIDATES))
    K_EVAL = int(cfg.get("k_eval", K_EVAL))
    print("[INFO] Running MBA baseline (rank by lift)")

    # ---------- load data ----------
    orders_path = Path(cfg.get("orders_path", ORDERS_PATH))
    orders = load_orders_parquet(orders_path).with_columns(
        pl.col("order_dt").cast(pl.Datetime)
    )
    products_path = Path(cfg.get("products_path", PRODUCTS_PATH))
    commerces_path = Path(cfg.get("commerces_path", COMMERCES_PATH))
    products = load_products_csv(products_path)

    # ---------- time split ----------
    train_orders = orders.filter(pl.col("order_dt") < SPLIT_DATE)
    test_orders = orders.filter(pl.col("order_dt") >= SPLIT_DATE)

    print(f"[INFO] Train orders: {train_orders.shape}")
    print(f"[INFO] Test orders:  {test_orders.shape}")

    # ---------- baskets ----------
    baskets_train = build_baskets(train_orders)

    # ---------- candidate generation ----------
    candidates = generate_candidates(
        baskets_train,
        min_cooc=MIN_COOC,
    )

    topk_candidates = select_top_k_candidates(
        candidates,
        k=K_CANDIDATES,
        min_lift=MIN_LIFT,
    )

    # ---------- feature table ----------
    feature_table = build_feature_table(
        baskets_train,
        topk_candidates,
    )

    commerces = load_commerces_csv(commerces_path)
    feature_config = (
        FeatureConfig.from_yaml(features_path)
        if features_path and features_path.exists()
        else FeatureConfig()
    )
    feature_table = add_all_features(
        feature_table=feature_table,
        orders=train_orders,
        products=products,
        commerces=commerces,
        config=feature_config,
    )

    # ---------- labels from test ----------
    labeled = build_labels(
        feature_table,
        test_orders,
    )

    print(f"[INFO] Labeled rows: {labeled.shape}")

    # ---------- MBA scoring (rank by lift) ----------
    scored = labeled.with_columns(
        pl.col("lift").alias("score")
    )

    # ---------- evaluation ----------
    hitrate = hitrate_at_k_by_score(
        scored,
        k=K_EVAL,
        score_col="score",
    )

    ndcg = ndcg_at_k_by_score(
        scored,
        k=K_EVAL,
        score_col="score",
    )

    print(f"\n[BASELINE RESULT] MBA HitRate@{K_EVAL} = {hitrate:.4f}")
    print(f"[BASELINE RESULT] MBA NDCG@{K_EVAL}    = {ndcg:.4f}")


if __name__ == "__main__":
    main()
