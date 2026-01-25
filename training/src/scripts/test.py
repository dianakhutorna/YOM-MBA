import polars as pl

orders = pl.read_parquet("training/data/interim/orders_sample.parquet")

orders.select(
    pl.min("order_dt").alias("min_dt"),
    pl.max("order_dt").alias("max_dt"),
)
