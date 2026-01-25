from __future__ import annotations
import polars as pl


def encode_channel_one_hot(
    df: pl.DataFrame,
    channel_col: str = "channel",
    prefix: str = "channel",
) -> pl.DataFrame:
    """
    One-hot encode channel column.

    Input:
        channel = "Mayorista"

    Output:
        channel_Mayorista = 1
        channel_Ruta = 0
        ...
    """

    if channel_col not in df.columns:
        return df

    # get unique channel values
    channels = (
        df.select(channel_col)
        .unique()
        .drop_nulls()
        .to_series()
        .to_list()
    )

    for ch in channels:
        col_name = f"{prefix}_{ch}"
        df = df.with_columns(
            (pl.col(channel_col) == ch)
            .cast(pl.Int8)
            .alias(col_name)
        )

    return df.drop(channel_col)
