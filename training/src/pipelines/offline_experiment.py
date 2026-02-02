from __future__ import annotations

import polars as pl

from training.src.config import OfflineExperimentConfig
from training.src.io import load_orders_parquet, load_products_csv, save_parquet
from training.src.steps.add_product_features import add_product_features
from training.src.steps.build_baskets import build_baskets
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.build_labels import build_labels
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.recall_at_k import recall_at_k
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.split_orders import split_orders_by_time


def run(config: OfflineExperimentConfig) -> float:
    print("[INFO] Starting offline experiment")

    orders = load_orders_parquet(config.orders_path)
    products = load_products_csv(config.products_path)
    print(f"[INFO] Loaded orders: {orders.shape}")

    train_orders, val_orders, test_orders = split_orders_by_time(
        orders,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    print(f"[INFO] Train orders: {train_orders.shape}")
    print(f"[INFO] Val orders:   {val_orders.shape}")
    print(f"[INFO] Test orders:  {test_orders.shape}")

    baskets_train = build_baskets(train_orders)
    baskets_test = build_baskets(test_orders)

    candidates = generate_candidates(
        baskets_train,
        min_cooc=config.min_cooc,
    )

    topk_candidates = select_top_k_candidates(
        candidates,
        k=config.top_k,
        min_lift=config.min_lift,
    )

    feature_table = build_feature_table(
        baskets_train,
        topk_candidates,
    )

    feature_table = add_product_features(feature_table, products)

    labeled_train = build_labels(
        feature_table,
        val_orders,
        window_days=config.label_window_days,
    )
    labeled_test = build_labels(
        feature_table,
        test_orders,
        window_days=config.label_window_days,
    )

    save_parquet(labeled_train, config.labeled_train_out_path)
    save_parquet(labeled_test, config.labeled_test_out_path)
    print(f"[OK] Saved train labels to {config.labeled_train_out_path}")
    print(f"[OK] Saved test labels to {config.labeled_test_out_path}")

    recall = recall_at_k(
        labeled_test,
        k=config.top_k,
    )

    print(f"\n[FINAL RESULT] Recall@{config.top_k} = {recall:.4f}")
    return recall
