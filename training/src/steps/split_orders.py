from __future__ import annotations

import logging
from typing import Tuple
from datetime import timedelta

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
    """
    Strict time-based split using datetime cutoffs.

    Splits by time intervals (not by number of rows).
    Ensures:
        train < val < test in time.
    """

    total = train_ratio + val_ratio + test_ratio
    if not 0.999 <= total <= 1.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    if dt_col not in orders.columns:
        raise ValueError(f"Missing datetime column: {dt_col}")

    if orders.is_empty():
        return orders, orders, orders

    # Ensure datetime
    orders = orders.with_columns(pl.col(dt_col).cast(pl.Datetime))

    # Sort strictly by time
    orders = orders.sort(dt_col)

    # Get time range
    dt_min, dt_max = orders.select(
        pl.col(dt_col).min().alias("min_dt"),
        pl.col(dt_col).max().alias("max_dt"),
    ).row(0)


    if dt_min == dt_max:
        # All orders same timestamp → fallback to row-based split
        LOGGER.warning("All orders have identical timestamp. Falling back to row-based split.")
        n_total = orders.height
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        return (
            orders.head(n_train),
            orders.slice(n_train, n_val),
            orders.tail(n_test),
        )

    total_seconds = (dt_max - dt_min).total_seconds()

    train_cutoff = dt_min + timedelta(seconds=int(total_seconds * train_ratio))
    val_cutoff = dt_min + timedelta(seconds=int(total_seconds * (train_ratio + val_ratio)))

    train_orders = orders.filter(pl.col(dt_col) < train_cutoff)
    val_orders = orders.filter(
        (pl.col(dt_col) >= train_cutoff) &
        (pl.col(dt_col) < val_cutoff)
    )
    test_orders = orders.filter(pl.col(dt_col) >= val_cutoff)

    # Safety fallback if any split is empty
    if train_orders.is_empty() or val_orders.is_empty() or test_orders.is_empty():
        LOGGER.warning(
            "One of the splits is empty after time-based split. "
            "Falling back to row-based split."
        )
        n_total = orders.height
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        train_orders = orders.head(n_train)
        val_orders = orders.slice(n_train, n_val)
        test_orders = orders.tail(n_test)

    LOGGER.info(
        "Time-based split: train=%s val=%s test=%s",
        train_orders.shape,
        val_orders.shape,
        test_orders.shape,
    )

    return train_orders, val_orders, test_orders

