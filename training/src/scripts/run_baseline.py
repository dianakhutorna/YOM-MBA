from __future__ import annotations

from pathlib import Path
import polars as pl

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
)

# ==========================
# Config
# ==========================
ORDERS_PATH = Path("training/data/interim/orders_sample.parquet")
PRODUCTS_PATH = Path("training/data/products_v2.csv")

SPLIT_DATE = pl.datetime(2024, 1, 4)

K_CANDIDATES = 100
K_EVAL = 20

MIN_COOC = 3
MIN_LIFT = 2.0


def main():
    print("[INFO] Running MBA baseline (rank by lift)")

    # ---------- load data ----------
    orders = pl.read_parquet(ORDERS_PATH).with_columns(
        pl.col("order_dt").cast(pl.Datetime)
    )
    products = pl.read_csv(PRODUCTS_PATH, separator=";")

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

    feature_table = add_product_features(feature_table, products)

    feature_table = add_kiosk_history_features(
        feature_table=feature_table,
        train_orders=train_orders,
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
