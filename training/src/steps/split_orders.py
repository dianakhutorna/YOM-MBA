from __future__ import annotations

import logging
from typing import Tuple

import polars as pl

LOGGER = logging.getLogger(__name__)


def split_orders_by_time(
    orders: pl.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    dt_col: str = "order_dt",
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if not 0.999 <= total <= 1.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    if dt_col not in orders.columns:
        raise ValueError(f"Missing datetime column: {dt_col}")

    orders = orders.with_columns(pl.col(dt_col).cast(pl.Datetime))
    orders = orders.sort(dt_col)

    n_total = orders.height
    if n_total == 0:
        return orders, orders, orders

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if n_test == 0 and n_total >= 3:
        n_test = 1
        if n_train > 0:
            n_train -= 1
        else:
            n_val = max(0, n_val - 1)

    if n_val == 0 and n_total >= 2:
        n_val = 1
        if n_train > 0:
            n_train -= 1
        else:
            n_test = max(0, n_test - 1)

    train_orders = orders.head(n_train)
    val_orders = orders.slice(n_train, n_val)
    test_orders = orders.tail(n_test)

    LOGGER.info("Split by time: train=%s val=%s test=%s", train_orders.shape, val_orders.shape, test_orders.shape)

    return train_orders, val_orders, test_orders
