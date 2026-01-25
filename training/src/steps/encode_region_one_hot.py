from __future__ import annotations
import polars as pl


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

    regions = (
        df.select(region_col)
        .unique()
        .drop_nulls()
        .to_series()
        .to_list()
    )

    for r in regions:
        col_name = f"{prefix}_{r}"
        df = df.with_columns(
            (pl.col(region_col) == r).cast(pl.Int8).alias(col_name)
        )

    return df.drop(region_col)
