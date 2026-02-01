# Из сырых orders получить каноническую таблицу, пригодную для:
# basket construction
# candidate generation
# ML

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

RAW_REQUIRED_COLS: tuple[str, ...] = (
    "documenttype",
    "orderid",
    "userid",
    "productid",
    "orderdt",
    "quantity",
)

CANONICAL_COLS: tuple[str, ...] = (
    "order_id",
    "kiosk_id",
    "product_id",
    "order_dt",
    "quantity",
)


# -----------------------------
# Assertions / data checks
# -----------------------------

def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def assert_not_null(df: pl.DataFrame, cols: Iterable[str]) -> None:
    """
    Assert that critical columns contain no nulls.
    Pipeline should fail if this is violated.
    """
    for c in cols:
        nulls = df.select(pl.col(c).null_count()).item()
        if nulls > 0:
            raise ValueError(f"Column '{c}' contains {nulls} null values")


def log_null_ratios(df: pl.DataFrame, cols: Iterable[str]) -> None:
    """
    Log (but do not fail) null ratios for non-critical columns.
    """
    for c in cols:
        ratio = df.select(pl.col(c).is_null().mean()).item()
        LOGGER.info("Null ratio in '%s': %.4f", c, ratio)


# -----------------------------
# Main preprocessing function
# -----------------------------

def preprocess_orders(
    df: pl.DataFrame,
    *,
    strict: bool = True,
    null_log_cols: Iterable[str] | None = None,
) -> pl.DataFrame:
    """
    Clean and normalize raw orders data.

    Output schema (canonical):
    - order_id
    - kiosk_id
    - product_id
    - order_dt
    - quantity
    """

    if strict:
        _ensure_columns(df, RAW_REQUIRED_COLS)

    LOGGER.info("Raw input shape: %s", df.shape)

    df_clean = (
        df
        # 1. Keep only real orders
        .filter(pl.col("documenttype") == "order")

        # 2. Rename + cast columns to canonical names
        .with_columns([
            pl.col("orderid").cast(pl.Utf8).alias("order_id"),
            pl.col("userid").cast(pl.Utf8).alias("kiosk_id"),
            pl.col("productid").cast(pl.Utf8).alias("product_id"),
            pl.col("orderdt")
              .str.strptime(pl.Datetime, strict=False)
              .alias("order_dt"),
            pl.col("quantity").cast(pl.Float64),
        ])

        # 3. Filter invalid rows
        .filter(pl.col("quantity") > 0)

        # 4. Select only columns we really need
        .select([
            "order_id",
            "kiosk_id",
            "product_id",
            "order_dt",
            "quantity",
        ])
    )

    LOGGER.info("After basic cleaning: %s", df_clean.shape)

    # -----------------------------
    # Hard assertions (must pass)
    # -----------------------------
    _ensure_columns(df_clean, CANONICAL_COLS)

    assert_not_null(
        df_clean,
        cols=[
            "order_id",
            "kiosk_id",
            "product_id",
            "order_dt",
        ],
    )

    if null_log_cols is not None:
        log_null_ratios(df_clean, null_log_cols)

    return df_clean
