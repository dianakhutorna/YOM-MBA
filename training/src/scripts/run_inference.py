from __future__ import annotations

from pathlib import Path
import polars as pl
import lightgbm as lgb
import pandas as pd

from training.src.steps.build_baskets import build_baskets
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.add_product_features import add_product_features
from training.src.steps.add_kiosk_features import add_kiosk_history_features

# ==========================
# Config
# ==========================
ORDERS_PATH = Path("training/data/interim/orders_sample.parquet")
PRODUCTS_PATH = Path("training/data/products_v2.csv")
MODEL_PATH = Path("training/models/lgbm_ranker.txt")

INFERENCE_DATE = pl.datetime(2024, 1, 5)
TOP_K_CANDIDATES = 20
FINAL_N = 10
MIN_COOC = 3
MIN_LIFT = 2.0


FEATURE_COLS = [
    "cooc_count",
    "anchor_count",
    "candidate_count",
    "support",
    "confidence",
    "lift",
    "cosine_sim",
    "same_category",
    "kiosk_product_cnt",
    "kiosk_bought_candidate_before",
]


# ==========================
# Main
# ==========================
def main():
    print("[INFO] Loading trained LightGBM model")
    ranker = lgb.Booster(model_file=str(MODEL_PATH))

    print("[INFO] Loading orders for inference")
    orders = pl.read_parquet(ORDERS_PATH).with_columns(
        pl.col("order_dt").cast(pl.Datetime)
    )

    orders = orders.filter(pl.col("order_dt") < INFERENCE_DATE)
    print(f"[INFO] Orders used for inference: {orders.shape}")

    # ---------- baskets ----------
    baskets = build_baskets(orders)

    # ---------- candidates ----------
    candidates = generate_candidates(baskets, min_cooc=MIN_COOC)

    topk_candidates = select_top_k_candidates(
        candidates,
        k=TOP_K_CANDIDATES,
        min_lift=MIN_LIFT,
    )

    # ---------- feature table ----------
    feature_table = build_feature_table(baskets, topk_candidates)

    products = pl.read_csv(PRODUCTS_PATH, separator=";")
    feature_table = add_product_features(feature_table, products)

    # ---------- ADD KIOSK-SPECIFIC FEATURES ----------
    feature_table = add_kiosk_history_features(
        feature_table=feature_table,
        train_orders=orders,   
    )

    # ---------- scoring ----------
    feature_table = feature_table.with_columns(
        [pl.col(c).fill_null(0) for c in FEATURE_COLS]
    )

    X = feature_table.select(FEATURE_COLS).to_pandas()
    scores = ranker.predict(X)

    scored = feature_table.with_columns(
        pl.Series("score", scores)
    )

    # ---------- final top-N ----------
    final = (
        scored
        .sort(["kiosk_id", "anchor_product_id", "score"], descending=[False, False, True])
        .group_by(["kiosk_id", "anchor_product_id"], maintain_order=True)
        .head(FINAL_N)
        .with_columns(pl.col("score").round(6))
    )


    print(f"[INFO] Final inference result shape: {final.shape}")

    # ---------- save ----------
    out_path = Path("training/data/interim/inference_results.parquet")
    final.write_parquet(out_path)
    print(f"[OK] Saved inference results to {out_path}")


if __name__ == "__main__":
    main()
