from __future__ import annotations

from pathlib import Path
from datetime import datetime

import polars as pl
import lightgbm as lgb

from training.src.io import load_commerces_csv, load_orders_parquet, load_products_csv
from training.src.paths import DATA_DIR, INTERIM_DIR, MODELS_DIR
from training.src.steps.build_baskets import build_baskets
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.build_feature_table import build_feature_table
from training.src.cli import parse_config_args
from training.src.config import FeatureConfig, load_yaml_config
from training.src.features import add_all_features



# ==========================
# Config
# ==========================
ORDERS_PATH = INTERIM_DIR / "orders_sample.parquet"
PRODUCTS_PATH = DATA_DIR / "products_v2.csv"
COMMERCES_PATH = DATA_DIR / "commerces.csv"
FEATURES_CONFIG_PATH = Path("training/configs/features.yaml")
SCRIPT_CONFIG_PATH = Path("training/configs/run_batch_scoring.yaml")
MODEL_PATH = MODELS_DIR / "lgbm_ranker.txt"

INFERENCE_DATE = pl.datetime(2024, 1, 4)

TOP_K_CANDIDATES = 100
FINAL_N = 10
MIN_COOC = 3
MIN_LIFT = 2.0

FEATURE_COLS_BASE = [
    #"cooc_count",
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


# ==========================
# Main
# ==========================
def main():
    config_path, features_path = parse_config_args(
        default_config=SCRIPT_CONFIG_PATH,
        default_features_config=FEATURES_CONFIG_PATH,
        description="Run batch scoring",
    )
    cfg = load_yaml_config(config_path) if config_path.exists() else {}
    global INFERENCE_DATE, MIN_COOC, MIN_LIFT, TOP_K_CANDIDATES, FINAL_N
    if "inference_date" in cfg:
        INFERENCE_DATE = datetime.fromisoformat(cfg["inference_date"])
    MIN_COOC = int(cfg.get("min_cooc", MIN_COOC))
    MIN_LIFT = float(cfg.get("min_lift", MIN_LIFT))
    TOP_K_CANDIDATES = int(cfg.get("top_k_candidates", TOP_K_CANDIDATES))
    FINAL_N = int(cfg.get("final_n", FINAL_N))
    print("[INFO] Loading trained LightGBM model")
    model_path = Path(cfg.get("model_path", MODEL_PATH))
    ranker = lgb.Booster(model_file=str(model_path))

    print("[INFO] Loading orders")
    orders_path = Path(cfg.get("orders_path", ORDERS_PATH))
    orders = load_orders_parquet(orders_path).with_columns(
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

    if queries.is_empty():
        print("[WARN] No inference queries – exiting.")
        return

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

    products_path = Path(cfg.get("products_path", PRODUCTS_PATH))
    commerces_path = Path(cfg.get("commerces_path", COMMERCES_PATH))
    products = load_products_csv(products_path)
    commerces = load_commerces_csv(commerces_path)
    feature_config = (
        FeatureConfig.from_yaml(features_path)
        if features_path and features_path.exists()
        else FeatureConfig()
    )
    feature_table = add_all_features(
        feature_table,
        orders=history_orders,
        products=products,
        commerces=commerces,
        config=feature_config,
    )

    # log scale for co-occurrence count
    feature_table = feature_table.with_columns(
        pl.col("cooc_count").log1p().alias("cooc_count")
    )

    # --------------------------------
    # FEATURE COLS (dynamic, like train)
    # --------------------------------
    region_cols = [c for c in feature_table.columns if c.startswith("region_")]
    FEATURE_COLS = FEATURE_COLS_BASE + region_cols

    print(f"[INFO] Using {len(FEATURE_COLS)} features ({len(region_cols)} region)")

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
    out_path = INTERIM_DIR / "predictions.parquet"
    final.write_parquet(out_path)
    print(f"[OK] Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
