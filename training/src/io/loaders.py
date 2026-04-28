from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Mapping

import polars as pl

LOGGER = logging.getLogger(__name__)


def _orders_schema_overrides() -> dict[str, pl.DataType]:
    return {
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
    }


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
        schema_overrides=schema_overrides or _orders_schema_overrides(),
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


def load_orders_csv_chunked(
    raw_paths: list[Path],
    *,
    preprocess_fn,
    chunk_size: int = 2_000_000,
    out_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Load one or more large CSV files in chunks, preprocess each chunk,
    and return a single concatenated DataFrame.
    """
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    schema_overrides = _orders_schema_overrides()
    chunk_paths: list[Path] = []
    parts: list[pl.DataFrame] = []
    total_raw = 0
    total_clean = 0

    for file_idx, raw_path in enumerate(raw_paths):
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")

        LOGGER.info("Chunked loading: file %d/%d — %s", file_idx + 1, len(raw_paths), raw_path)
        t0 = time.perf_counter()

        reader = pl.read_csv_batched(
            raw_path,
            has_header=True,
            separator=",",
            try_parse_dates=False,
            infer_schema_length=0,
            schema_overrides=schema_overrides,
            batch_size=chunk_size,
        )

        chunk_idx = 0
        while True:
            batches = reader.next_batches(1)
            if batches is None or len(batches) == 0:
                break

            raw_chunk = batches[0]
            total_raw += raw_chunk.height
            clean_chunk = preprocess_fn(raw_chunk)
            total_clean += clean_chunk.height

            if out_dir is not None:
                chunk_path = out_dir / f"chunk_{file_idx:02d}_{chunk_idx:04d}.parquet"
                clean_chunk.write_parquet(chunk_path, compression="zstd")
                chunk_paths.append(chunk_path)
            else:
                parts.append(clean_chunk)

            del raw_chunk, clean_chunk
            chunk_idx += 1

        elapsed = time.perf_counter() - t0
        LOGGER.info("  file done in %.1fs — %d chunks processed", elapsed, chunk_idx)

    if out_dir is not None:
        parts = [pl.read_parquet(path) for path in chunk_paths]

    if not parts:
        return pl.DataFrame()

    out = pl.concat(parts, how="vertical_relaxed")
    LOGGER.info(
        "Chunked orders loaded: raw_rows=%s clean_rows=%s parts=%s",
        f"{total_raw:,}", f"{total_clean:,}", len(parts),
    )
    return out


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

