# Из сырых orders получить каноническую таблицу, пригодную для:
# basket construction
# candidate generation
# ML

from __future__ import annotations

import polars as pl


# -----------------------------
# Assertions / data checks
# -----------------------------

def assert_not_null(df: pl.DataFrame, cols: list[str]) -> None:
    """
    Assert that critical columns contain no nulls.
    Pipeline should fail if this is violated.
    """
    for c in cols:
        nulls = df.select(pl.col(c).null_count()).item()
        if nulls > 0:
            raise ValueError(f"Column '{c}' contains {nulls} null values")


def log_null_ratios(df: pl.DataFrame, cols: list[str]) -> None:
    """
    Log (but do not fail) null ratios for non-critical columns.
    """
    for c in cols:
        ratio = df.select(pl.col(c).is_null().mean()).item()
        print(f"[INFO] Null ratio in '{c}': {ratio:.4f}")


# -----------------------------
# Main preprocessing function
# -----------------------------

def preprocess_orders(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean and normalize raw orders data.

    Output schema (canonical):
    - order_id
    - kiosk_id
    - product_id
    - order_dt
    - quantity
    """

    print(f"[INFO] Raw input shape: {df.shape}")

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

    print(f"[INFO] After basic cleaning: {df_clean.shape}")

    # -----------------------------
    # Hard assertions (must pass)
    # -----------------------------

    assert_not_null(
        df_clean,
        cols=[
            "order_id",
            "kiosk_id",
            "product_id",
            "order_dt",
        ],
    )

    return df_clean
