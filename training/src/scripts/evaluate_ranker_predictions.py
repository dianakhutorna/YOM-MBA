from __future__ import annotations

import polars as pl

from training.src.io import load_orders_parquet, load_parquet
from training.src.paths import INTERIM_DIR
from training.src.steps.build_labels import build_labels
from training.src.steps.rank_eval_at_k import (
    hitrate_at_k_by_score,
    recall_at_k_by_score,
    ndcg_at_k_by_score,
    positives_at_k_by_score,
    quantity_captured_at_k_by_score,
    precision_at_k_by_score,
)

# ======================================================
# CONFIG
# ======================================================
PREDICTIONS_PATH = INTERIM_DIR / "predictions_xgb.parquet"
ORDERS_PATH = INTERIM_DIR / "orders_sample.parquet"

K_EVAL = 20
INFERENCE_DATE = pl.datetime(2024, 1, 4)

# ======================================================
# MAIN
# ======================================================
def main():
    print("[INFO] Loading predictions (scores only)")
    preds = load_parquet(PREDICTIONS_PATH, label="Predictions parquet")

    print("[INFO] Loading orders")
    orders = load_orders_parquet(ORDERS_PATH).with_columns(
        pl.col("order_dt").cast(pl.Datetime)
    )

    test_orders = orders.filter(pl.col("order_dt") >= INFERENCE_DATE)

    print(f"[INFO] Predictions shape: {preds.shape}")
    print(f"[INFO] Test orders shape: {test_orders.shape}")

    # --------------------------------------------------
    # 🔑 REBUILD LABELS (CRITICAL STEP)
    # --------------------------------------------------
    print("[INFO] Rebuilding labels for offline evaluation")

    scored_with_labels = build_labels(
        feature_table=preds,
        test_orders=test_orders,
    )

    # safety: fill nulls
    scored_with_labels = scored_with_labels.with_columns(
        pl.col("label").fill_null(0).cast(pl.Int8)
    )

    print(
        f"[INFO] Positives ratio: "
        f"{scored_with_labels.select(pl.col('label').mean()).item():.6f}"
    )

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    print("\n========== RANKING METRICS ==========")

    hr = hitrate_at_k_by_score(scored_with_labels, k=K_EVAL)
    recall = recall_at_k_by_score(scored_with_labels, k=K_EVAL)
    ndcg = ndcg_at_k_by_score(scored_with_labels, k=K_EVAL)
    pos = positives_at_k_by_score(scored_with_labels, k=K_EVAL)
    qty = quantity_captured_at_k_by_score(
        scored_with_labels,
        test_orders,
        k=K_EVAL,
    )
    prec = precision_at_k_by_score(scored_with_labels, k=K_EVAL)

    print(f"HitRate@{K_EVAL}:          {hr:.4f}")
    print(f"Recall@{K_EVAL}:           {recall:.4f}")
    print(f"NDCG@{K_EVAL}:             {ndcg:.4f}")
    print(f"Positives@{K_EVAL}:        {pos:.4f}")
    print(f"Precision@{K_EVAL}:        {prec:.4f}")
    print(f"QuantityCaptured@{K_EVAL}: {qty:.4f}")

    print("====================================\n")


if __name__ == "__main__":
    main()
