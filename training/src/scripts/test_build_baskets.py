from pathlib import Path
import polars as pl

from training.src.steps.build_baskets import build_baskets


INTERIM_ORDERS = Path("training/data/interim/orders_sample.parquet")
INTERIM_BASKETS = Path("training/data/interim/baskets_sample.parquet")


def main():
    orders = pl.read_parquet(INTERIM_ORDERS)
    print(f"[INFO] Loaded orders: {orders.shape}")

    baskets = build_baskets(orders, min_items=2)

    print(baskets.head())
    print(baskets.select(pl.col("products").list.len().alias("basket_size")).describe()
    )

    baskets.write_parquet(INTERIM_BASKETS)
    print(f"[OK] Saved baskets to {INTERIM_BASKETS}")


if __name__ == "__main__":
    main()
