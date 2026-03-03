from __future__ import annotations

from pathlib import Path
from typing import Mapping
import logging
import time

import polars as pl

LOGGER = logging.getLogger(__name__)


def _log_df_info(df: pl.DataFrame, *, label: str, path: Path, log_columns: bool = False) -> None:
    LOGGER.info("%s loaded: path=%s rows=%s cols=%s", label, path, df.height, df.width)
    if log_columns:
        LOGGER.debug("%s columns: %s", label, df.columns)


def load_parquet(path: Path, label: str = "parquet") -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    t0 = time.perf_counter()
    df = pl.read_parquet(path)
    LOGGER.info("%s loaded in %.2fs: %s", label, time.perf_counter() - t0, path)
    return df


def load_orders_csv_sample(
    raw_path: Path,
    n_rows: int = 1_000_000,
    schema_overrides: Mapping[str, pl.DataType] | None = None,
    sample_position: str = "head",
    infer_schema_length: int = 2000,   # вместо 0
    log_columns: bool = False,
) -> pl.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    if sample_position not in {"head", "tail"}:
        raise ValueError("sample_position must be 'head' or 'tail'")

    t0 = time.perf_counter()

    # Важно: tail для CSV может читать почти весь файл — логируем предупреждение.
    if sample_position == "tail":
        LOGGER.warning(
            "CSV tail sampling can be expensive for large files (may scan a big portion of the file): path=%s n_rows=%s",
            raw_path, n_rows
        )

    lf = pl.scan_csv(
        raw_path,
        has_header=True,
        separator=",",
        try_parse_dates=False,
        infer_schema_length=0,  # отключаем авто-инференс
        schema_overrides={
            "orderid": pl.Utf8,
            "productid": pl.Utf8,
            "userid": pl.Utf8,
            "documentcode": pl.Utf8,
            "documenttype": pl.Utf8,
            "currency": pl.Utf8,
            "origin": pl.Utf8,
            "sellerid": pl.Utf8,
            "sellerrouteid": pl.Utf8,
            "couponcode": pl.Utf8,
            "priceperunit": pl.Float64,
            "tax": pl.Float64,
            "discountperunit": pl.Float64,
            "discountedpriceperunit": pl.Float64,
            "quantity": pl.Float64,
        },
    )


    df = (lf.tail(n_rows) if sample_position == "tail" else lf.limit(n_rows)).collect()

    elapsed = time.perf_counter() - t0
    LOGGER.info(
        "Orders sample loaded in %.2fs: path=%s sample_position=%s rows=%s cols=%s",
        elapsed, raw_path, sample_position, df.height, df.width
    )
    if log_columns:
        LOGGER.debug("Orders columns: %s", df.columns)

    return df


def load_orders_parquet(path: Path) -> pl.DataFrame:
    return load_parquet(path, label="Orders parquet")


def load_products_csv(
    path: Path,
    separator: str = ";",
    filter_blocked: bool = True,
) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Products file not found: {path}")
    t0 = time.perf_counter()
    df = pl.read_csv(path, separator=separator)
    LOGGER.info("Products loaded in %.2fs: path=%s rows=%s cols=%s", time.perf_counter() - t0, path, df.height, df.width)

    if filter_blocked and "blocked" in df.columns:
        rows_before = df.height
        df = df.filter(
            ~pl.col("blocked").cast(pl.Utf8).str.to_lowercase().eq("true")
        )
        removed = rows_before - df.height
        LOGGER.info(
            "Blocked products filtered: %s removed, %s remaining",
            removed, df.height,
        )

    return df


def load_commerces_csv(path: Path, separator: str = ";") -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Commerces file not found: {path}")
    t0 = time.perf_counter()
    df = pl.read_csv(path, separator=separator)
    LOGGER.info("Commerces loaded in %.2fs: path=%s rows=%s cols=%s", time.perf_counter() - t0, path, df.height, df.width)
    return df


def save_parquet(df: pl.DataFrame, out_path: Path, compression: str = "zstd") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df.write_parquet(out_path, compression=compression)
    LOGGER.info(
        "Parquet saved in %.2fs: path=%s rows=%s cols=%s compression=%s",
        time.perf_counter() - t0, out_path, df.height, df.width, compression
    )

