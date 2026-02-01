import polars as pl

from training.src.io import load_orders_parquet
from training.src.paths import INTERIM_DIR
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.build_labels import build_labels
from training.src.steps.recall_at_k import recall_at_k

from training.src.steps.build_baskets import build_baskets
from training.src.steps.generate_candidates import generate_candidates


ORDERS_PATH = INTERIM_DIR / "orders_sample.parquet"

K_VALUES = [5, 10, 20, 50]
MIN_LIFT_VALUES = [1.5, 2.0, 3.0]


def main():
    orders = load_orders_parquet(ORDERS_PATH)

    orders = orders.with_columns(pl.col("order_dt").cast(pl.Datetime))

    SPLIT_DATE = pl.datetime(2024, 1, 4)

    train_orders = orders.filter(pl.col("order_dt") < SPLIT_DATE)
    test_orders = orders.filter(pl.col("order_dt") >= SPLIT_DATE)

    baskets_train = build_baskets(train_orders)

    candidates = generate_candidates(baskets_train, min_cooc=3)

    print("\n===== EVAL GRID =====")

    for min_lift in MIN_LIFT_VALUES:
        for k in K_VALUES:
            topk = select_top_k_candidates(
                candidates,
                k=k,
                min_lift=min_lift,
            )

            features = build_feature_table(baskets_train, topk)

            labeled = build_labels(features, test_orders)

            recall = recall_at_k(labeled, k=k)

            print(
                f"min_lift={min_lift:<4} | K={k:<2} | Recall@K={recall:.4f}"
            )


if __name__ == "__main__":
    main()
