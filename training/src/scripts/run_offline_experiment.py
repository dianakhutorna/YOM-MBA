from pathlib import Path
import polars as pl

from training.src.steps.build_baskets import build_baskets
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.build_labels import build_labels
from training.src.steps.recall_at_k import recall_at_k
from training.src.steps.add_product_features import add_product_features


# --------------------------------------------------
# Experiment configuration (ВСЕ РЕШЕНИЯ ЗДЕСЬ)
# --------------------------------------------------

ORDERS_PATH = Path("training/data/interim/orders_sample.parquet")

PRODUCTS_PATH = "training/data/products_v2.csv"

SPLIT_DATE = pl.datetime(2024, 1, 4)

MIN_COOC = 3
MIN_LIFT = 2.0
TOP_K = 20


def main():
    print("[INFO] Starting offline experiment")

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    orders = pl.read_parquet(ORDERS_PATH)
    products = pl.read_csv(PRODUCTS_PATH, separator=";")
    print(f"[INFO] Loaded orders: {orders.shape}")

    # --------------------------------------------------
    # 2. Train / test split (ПО ВРЕМЕНИ)
    # --------------------------------------------------

    train_orders = orders.filter(pl.col("order_dt") < SPLIT_DATE)
    test_orders = orders.filter(pl.col("order_dt") >= SPLIT_DATE)

    print(f"[INFO] Train orders: {train_orders.shape}")
    print(f"[INFO] Test orders:  {test_orders.shape}")

    # --------------------------------------------------
    # 2. Train / test split (80/20 по времени)

        # находим 80-й перцентиль по времени

    # orders = orders.with_columns(
    # pl.col("order_dt").cast(pl.Datetime))   

    # split_dt = (orders.select(pl.col("order_dt").quantile(0.8)).item())

    # print(f"[INFO] Split datetime (80%): {split_dt}")

    # train_orders = orders.filter(pl.col("order_dt") <= split_dt)
    
    # test_orders  = orders.filter(pl.col("order_dt") > split_dt)

    # print(f"[INFO] Train orders: {train_orders.shape}")
    # print(f"[INFO] Test orders:  {test_orders.shape}")
    # --------------------------------------------------


    # --------------------------------------------------
    # 3. Build baskets
    # --------------------------------------------------

    baskets_train = build_baskets(train_orders)
    baskets_test = build_baskets(test_orders)

    # --------------------------------------------------
    # 4. Candidate generation (TRAIN ONLY)
    # --------------------------------------------------

    candidates = generate_candidates(
        baskets_train,
        min_cooc=MIN_COOC,
    )

    topk_candidates = select_top_k_candidates(
        candidates,
        k=TOP_K,
        min_lift=MIN_LIFT,
    )

    # --------------------------------------------------
    # 5. Feature table (TRAIN)
    # --------------------------------------------------

    feature_table = build_feature_table(
        baskets_train,
        topk_candidates,
    )

    feature_table = add_product_features(feature_table, products)
    # --------------------------------------------------
    # 6. Labels (FROM TEST)
    # --------------------------------------------------

    labeled_features = build_labels(
        feature_table,
        test_orders,
    )

    LABELED_PATH = "training/data/interim/labeled_features_sample.parquet"

    labeled_features.write_parquet(LABELED_PATH)
    print(f"[OK] Saved labeled features to {LABELED_PATH}")

    # --------------------------------------------------
    # 7. Offline evaluation
    # --------------------------------------------------

    recall = recall_at_k(
        labeled_features,
        k=TOP_K,
    )

    print(f"\n[FINAL RESULT] Recall@{TOP_K} = {recall:.4f}")


if __name__ == "__main__":
    main()
