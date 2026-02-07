from __future__ import annotations

import logging
import re

import polars as pl

LOGGER = logging.getLogger(__name__)


def encode_region_one_hot(
    df: pl.DataFrame,
    region_col: str = "region",
    prefix: str = "region",
) -> pl.DataFrame:
    """
    One-hot encode region column.
    """
    if region_col not in df.columns:
        return df

    LOGGER.info("One-hot encoding %s", region_col)

    regions = (
        df.select(region_col)
        .unique()
        .drop_nulls()
        .to_series()
        .to_list()
    )

    def _normalize_value(value: str) -> str:
        return re.sub(r"\s+", "_", str(value).strip())

    for r in regions:
        col_name = f"{prefix}_{_normalize_value(r)}"
        df = df.with_columns(
            (pl.col(region_col) == r).cast(pl.Int8).alias(col_name)
        )

    return df.drop(region_col)
