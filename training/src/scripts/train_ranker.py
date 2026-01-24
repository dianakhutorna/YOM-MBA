from __future__ import annotations

from pathlib import Path
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
from training.src.steps.rank_eval_at_k import recall_at_k_by_score
from training.src.steps.add_kiosk_features import add_kiosk_history_features



ORDERS_PATH = Path("training/data/interim/orders_sample.parquet")
PRODUCTS_PATH = Path("training/data/products_v2.csv")

K = 20
MIN_COOC = 3
MIN_LIFT = 2.0


def _make_group_sizes(df_pd: pd.DataFrame) -> np.ndarray:
    # group sizes for LightGBM ranker
    return df_pd.groupby(["kiosk_id", "anchor_product_id"], sort=False).size().to_numpy()


def main():
    print("[INFO] Loading orders")
    orders = pl.read_parquet(ORDERS_PATH).with_columns(
        pl.col("order_dt").cast(pl.Datetime)
    )

    # ---------- 80/20 split by time ----------
    SPLIT_DATE = pl.datetime(2024, 1, 4)
    # split_dt = orders.select(pl.col("order_dt").quantile(0.8)).item()
    # assert split_dt is not None, "split_dt is None; check order_dt dtype"

    train_orders = orders.filter(pl.col("order_dt") < SPLIT_DATE)
    test_orders = orders.filter(pl.col("order_dt") >= SPLIT_DATE)

    # print(f"[INFO] split_dt: {split_dt}")
    print(f"[INFO] Train orders: {train_orders.shape}")
    print(f"[INFO] Test orders:  {test_orders.shape}")

    # ---------- build train candidates ----------
    baskets_train = build_baskets(train_orders)
    candidates = generate_candidates(baskets_train, min_cooc=MIN_COOC)

    topk_candidates = select_top_k_candidates(
        candidates,
        k=K,
        min_lift=MIN_LIFT,
    )

    # ---------- feature table (based on train) ----------
    feature_table = build_feature_table(baskets_train, topk_candidates)

    products = pl.read_csv(PRODUCTS_PATH, separator=";")
    feature_table = add_product_features(feature_table, products)

    # ---------- ADD KIOSK-SPECIFIC FEATURES (NEW) ----------
    feature_table = add_kiosk_history_features(
        feature_table=feature_table,
        train_orders=train_orders,
    )

    # ---------- labels from test purchases ----------
    labeled = build_labels(feature_table, test_orders)


    # Save for debugging
    out_path = Path("training/data/interim/labeled_features_for_ranker.parquet")
    labeled.write_parquet(out_path)
    print(f"[OK] Saved labeled dataset to {out_path}")

    # ==========================================================
    # Train/test split FOR THE MODEL
    # ==========================================================
    # IMPORTANT:
    # - candidates/features built on train period already
    # - labels come from test period
    #
    # For the first ML prototype, we can do:
    # - model_train = a random subset of labeled rows
    # - model_valid = another subset
    #
    # BUT лучше: split по kiosk_id чтобы не было утечки по киоскам.
    # Сейчас делаем kiosk split 90/10 (быстро и честно).
    # ==========================================================

    kiosks = labeled.select("kiosk_id").unique()
    kiosks_pd = kiosks.to_pandas()
    rng = np.random.default_rng(42)
    mask = rng.random(len(kiosks_pd)) < 0.9
    train_kiosks = set(kiosks_pd.loc[mask, "kiosk_id"].tolist())

    model_train = labeled.filter(pl.col("kiosk_id").is_in(train_kiosks))
    model_valid = labeled.filter(~pl.col("kiosk_id").is_in(train_kiosks))

    print(f"[INFO] Model train rows: {model_train.shape}")
    print(f"[INFO] Model valid rows: {model_valid.shape}")

    # ---------- choose feature columns ----------
    feature_cols = [
        "cooc_count",
        "anchor_count",
        "candidate_count",
        "support",
        "confidence",
        "lift",
        "same_category",
        "kiosk_product_cnt",
        "kiosk_bought_candidate_before",
    ]

    # Ensure no nulls in features
    model_train = model_train.with_columns([pl.col(c).fill_null(0) for c in feature_cols])
    model_valid = model_valid.with_columns([pl.col(c).fill_null(0) for c in feature_cols])

    # Convert to pandas for LightGBM
    train_pd = model_train.select(
        ["kiosk_id", "anchor_product_id", "candidate_product_id", "label"] + feature_cols
    ).to_pandas()

    valid_pd = model_valid.select(
        ["kiosk_id", "anchor_product_id", "candidate_product_id", "label"] + feature_cols
    ).to_pandas()

    X_train = train_pd[feature_cols]
    y_train = train_pd["label"].astype(int)

    X_valid = valid_pd[feature_cols]
    y_valid = valid_pd["label"].astype(int)

    group_train = _make_group_sizes(train_pd)
    group_valid = _make_group_sizes(valid_pd)

    # ---------- LightGBM Ranker ----------
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

    print("[INFO] Training LightGBM ranker")

    print("[INFO] Group size stats (train):")
    print(pd.Series(group_train).describe())

    print("[INFO] Group size stats (valid):")
    print(pd.Series(group_valid).describe())

    ranker.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[K]
    )

    # ---------- Score validation set ----------
    valid_pd["score"] = ranker.predict(X_valid)

    valid_scored = pl.from_pandas(valid_pd)

    recall = recall_at_k_by_score(valid_scored, k=K, score_col="score")

    print(f"\n[FINAL RESULT] LightGBM Recall@{K} = {recall:.4f}")

    # ---------- Feature importance ----------
    feat_imp = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": ranker.feature_importances_,
        })
        .sort_values("importance", ascending=False)
    )

    print("\n[INFO] Feature importance:")
    print(feat_imp)

    MODEL_PATH = Path("training/models/lgbm_ranker.txt")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    ranker.booster_.save_model(str(MODEL_PATH))
    print(f"[OK] Model saved to {MODEL_PATH}")



if __name__ == "__main__":
    main()
