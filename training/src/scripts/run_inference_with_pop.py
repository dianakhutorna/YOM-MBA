from __future__ import annotations

from pathlib import Path
import polars as pl
import lightgbm as lgb
import numpy as np

from training.src.steps.add_popularity_features import add_popularity_features
from training.src.steps.build_baskets import build_baskets
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.add_product_features import add_product_features
from training.src.steps.add_kiosk_features import add_kiosk_history_features
from training.src.steps.encode_categorical_features import encode_channel_one_hot


# ==========================
# Config
# ==========================
ORDERS_PATH = Path("training/data/interim/orders_sample.parquet")
PRODUCTS_PATH = Path("training/data/products_v2.csv")
MODEL_PATH = Path("training/models/lgbm_ranker.txt")
COMMERCES_PATH = Path("training/data/commerces.csv")

commerces = pl.read_csv(COMMERCES_PATH, separator=";")


INFERENCE_DATE = pl.datetime(2024, 1, 4)

TOP_K_CANDIDATES = 100
FINAL_N = 10
MIN_COOC = 3
MIN_LIFT = 2.0

FEATURE_COLS = [
    "cooc_count",
    "kiosk_product_cnt",
    #"pop_global",
    "pop_channel",
    "pop_region",
    "pop_store",
    "channel_Mayorista",
    "channel_Ruta",
    "channel_Foodservice",
    "channel_Distribuidores",
    "channel_Supermercados",
]


# ==========================
# Main
# ==========================
def main():
    print("[INFO] Loading trained LightGBM model")
    ranker = lgb.Booster(model_file=str(MODEL_PATH))

    print("[INFO] Loading orders")
    orders = pl.read_parquet(ORDERS_PATH).with_columns(
        pl.col("order_dt").cast(pl.Datetime)
    )

    # --------------------------------
    # Split history vs inference day
    # --------------------------------
    history_orders = orders.filter(pl.col("order_dt") < INFERENCE_DATE)
    inference_orders = orders.filter(pl.col("order_dt") >= INFERENCE_DATE)

    print(f"[INFO] History orders:   {history_orders.shape}")
    print(f"[INFO] Inference orders: {inference_orders.shape}")

    # --------------------------------
    # Build baskets
    # --------------------------------
    history_baskets = build_baskets(history_orders)
    inference_baskets = build_baskets(inference_orders)

    # --------------------------------
    # Queries = real anchor situations
    # --------------------------------
    queries = (
        inference_baskets
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .unique()
    )

    print(f"[INFO] Inference queries: {queries.shape}")

    # --------------------------------
    # Candidate generation (GLOBAL)
    # --------------------------------
    candidates = generate_candidates(history_baskets, min_cooc=MIN_COOC)

    topk_candidates = select_top_k_candidates(
        candidates,
        k=TOP_K_CANDIDATES,
        min_lift=MIN_LIFT,
    )

    # --------------------------------
    # Feature table
    # --------------------------------
    feature_table = build_feature_table(
        baskets=history_baskets,
        topk_candidates=topk_candidates,
        queries=queries,
    )

    products = pl.read_csv(PRODUCTS_PATH, separator=";")
    feature_table = add_product_features(feature_table, products)

    feature_table = add_kiosk_history_features(
        feature_table=feature_table,
        train_orders=history_orders,
    )
    feature_table = add_popularity_features(
        feature_table=feature_table,
        orders=history_orders,
        commerces=commerces,
    )

    feature_table = feature_table.with_columns(
        pl.col("pop_global").log1p().alias("pop_global")
    )

    feature_table = encode_channel_one_hot(feature_table)

    # --------------------------------
    # Scoring
    # --------------------------------
    for c in FEATURE_COLS:
        if c not in feature_table.columns:
            feature_table = feature_table.with_columns(pl.lit(0).alias(c))

    feature_table = feature_table.with_columns(
        [pl.col(c).fill_null(0) for c in FEATURE_COLS]
    )

    X = feature_table.select(FEATURE_COLS).to_pandas()
    scores = ranker.predict(X)

    scored = feature_table.with_columns(
        pl.Series("score", scores)
    )

    # --------------------------------
    # Final top-N per (kiosk, anchor)
    # --------------------------------
    final = (
        scored
        .sort(
            ["kiosk_id", "anchor_product_id", "score"],
            descending=[False, False, True],
        )
        .group_by(["kiosk_id", "anchor_product_id"], maintain_order=True)
        .head(FINAL_N)
        .with_columns(pl.col("score").round(6))
    )

    print(f"[INFO] Final inference result shape: {final.shape}")

    # --------------------------------
    # Save
    # --------------------------------
    out_path = Path("training/data/interim/predictions.parquet")
    final.write_parquet(out_path)
    print(f"[OK] Saved predictions to {out_path}")


if __name__ == "__main__":
    main()

