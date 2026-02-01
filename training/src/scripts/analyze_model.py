from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt

from training.src.paths import MODELS_DIR, INTERIM_DIR, LOGS_DIR, DATA_DIR
from training.src.io import load_orders_parquet, load_products_csv, load_commerces_csv
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.build_labels import build_labels
from training.src.steps.build_baskets import build_baskets
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.features import add_all_features
from training.src.config import FeatureConfig


# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = MODELS_DIR / "lgbm_ranker.txt"
ORDERS_PATH = INTERIM_DIR / "orders_sample.parquet"
FEATURES_CONFIG_PATH = Path("training/configs/features.yaml")

# PDP делаем ТОЛЬКО для этих фич
PDP_FEATURES = [
    #"cosine_sim",
    #"pop_store",
    #"kiosk_product_cnt",
    #"channel_Ruta",
    #"channel_Foodservice",
    #"anchor_kiosk_frequency",
    #"kiosk_bought_candidate_before",
    #"cand_is_new_for_kiosk",
    "region_Santiago",
    "channel_Supermercados",
    "channel_Mayorista",
    "channel_Distribuidores",
    #"region_Puerto Montt",
    "region_Temuco",
]

PDP_DIR = MODELS_DIR / "pdp"
PDP_SUBSAMPLE = 200_000
PDP_GRID_SIZE = 30


# ======================================================
# LOGGING
# ======================================================
def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


# ======================================================
# BUILD PDP DATA (FULL FEATURE SPACE!)
# ======================================================
def build_pdp_dataframe(
    model_features: list[str],
) -> pd.DataFrame:
    """
    ВАЖНО:
    - используем train-распределение
    - feature space = ТОЧНО как при обучении
    """

    orders = load_orders_parquet(ORDERS_PATH).with_columns(
        pl.col("order_dt").cast(pl.Datetime)
    )

    SPLIT_DATE = pl.datetime(2024, 1, 4)
    train_orders = orders.filter(pl.col("order_dt") < SPLIT_DATE)

    baskets = build_baskets(train_orders)

    candidates = generate_candidates(baskets, min_cooc=3)
    topk = select_top_k_candidates(candidates, k=100, min_lift=2.0)

    queries = (
        baskets
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .unique()
    )

    ft = build_feature_table(
        baskets=baskets,
        topk_candidates=topk,
        queries=queries,
    )

    products = load_products_csv(DATA_DIR / "products_v2.csv")
    commerces = load_commerces_csv(DATA_DIR / "commerces.csv")

    feature_cfg = FeatureConfig.from_yaml(FEATURES_CONFIG_PATH)
    ft = add_all_features(
        ft,
        orders=train_orders,
        products=products,
        commerces=commerces,
        config=feature_cfg,
    )

    ft = build_labels(ft, train_orders)

    pdf = ft.to_pandas()

    # --- КЛЮЧЕВОЕ МЕСТО ---
    # гарантируем тот же feature space, что был при обучении
    for col in model_features:
        if col not in pdf.columns:
            pdf[col] = 0.0

    pdf = pdf[model_features]

    pdf = pdf.sample(
        n=min(PDP_SUBSAMPLE, len(pdf)),
        random_state=42,
    )

    logging.info(f"PDP data shape: {pdf.shape}")
    return pdf


# ======================================================
# MANUAL PDP
# ======================================================
def compute_manual_pdp(
    booster: lgb.Booster,
    X: pd.DataFrame,
    feature: str,
    grid_size: int,
):
    values = np.quantile(
        X[feature],
        np.linspace(0.05, 0.95, grid_size),
    )

    pdp = []
    X_tmp = X.copy()

    for v in values:
        X_tmp[feature] = v
        preds = booster.predict(X_tmp)
        pdp.append(preds.mean())

    return values, np.array(pdp)


# ======================================================
# MAIN
# ======================================================
def main():
    setup_logging()
    logging.info("Starting MANUAL PDP analysis")

    # ---------- Load model ----------
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    model_features = booster.feature_name()

    logging.info(f"Loaded model from {MODEL_PATH}")
    logging.info(f"Model expects {len(model_features)} features")

    # ---------- Build PDP data ----------
    X_pdp = build_pdp_dataframe(model_features)

    # ---------- Plot PDP ----------
    PDP_DIR.mkdir(parents=True, exist_ok=True)

    n_feats = len(PDP_FEATURES)
    n_cols = 2
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(14, 4 * n_rows),
    )

    axes = axes.flatten()

    for i, feat in enumerate(PDP_FEATURES):
        xs, ys = compute_manual_pdp(
            booster=booster,
            X=X_pdp,
            feature=feat,
            grid_size=PDP_GRID_SIZE,
        )

        axes[i].plot(xs, ys)
        axes[i].set_title(f"PDP: {feat}")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Mean prediction")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    out_path = PDP_DIR / "pdp_ranker.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    logging.info(f"PDP saved to {out_path}")


if __name__ == "__main__":
    main()
