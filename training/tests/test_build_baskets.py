from __future__ import annotations

import polars as pl


def test_build_baskets_schema(baskets_df):
    expected = {"kiosk_id", "order_id", "products"}
    assert expected.issubset(set(baskets_df.columns))
    assert baskets_df.schema["products"] == pl.List(pl.Utf8)


def test_build_baskets_min_items(baskets_df):
    min_size = baskets_df.select(pl.col("products").list.len().min()).item()
    assert min_size is None or min_size >= 2
