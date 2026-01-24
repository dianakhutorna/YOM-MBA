from __future__ import annotations

import polars as pl


def build_baskets(
    orders: pl.DataFrame,
    min_items: int = 2,
) -> pl.DataFrame:
    """
    Build baskets from cleaned orders data.

    Input schema:
    - order_id
    - kiosk_id
    - product_id
    - order_dt
    - quantity

    Output schema:
    - kiosk_id
    - order_id
    - products: list[str]
    """

    print(f"[INFO] Building baskets from orders: {orders.shape}")

    baskets = (
        orders
        # group by kiosk + order
        .group_by(["kiosk_id", "order_id"])
        .agg(
            pl.col("product_id")
            .unique()
            .alias("products")
        )
        # keep only baskets with >= min_items
        .filter(pl.col("products").list.len() >= min_items)
    )

    print(f"[INFO] Built baskets: {baskets.shape}")
    # print(f"[INFO] Avg basket size: "
    #      f"{baskets.select(pl.col('products').list.len().mean()).item():.2f}")

    avg_size = baskets.select(
        pl.col("products").list.len().mean()
    ).item()

    if avg_size is not None:
        print(f"[INFO] Avg basket size: {avg_size:.2f}")
    else:
        print("[INFO] Avg basket size: n/a (no baskets)")


    return baskets
