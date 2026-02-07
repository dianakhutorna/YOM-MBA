from __future__ import annotations

from pathlib import Path
from typing import Mapping

import polars as pl


def load_parquet(path: Path, label: str = "parquet") -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return pl.read_parquet(path)


def load_orders_csv_sample(
    raw_path: Path,
    n_rows: int = 1_000_000,
    schema_overrides: Mapping[str, pl.DataType] | None = None,
    sample_position: str = "head",
) -> pl.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    lf = pl.scan_csv(
        raw_path,
        has_header=True,
        separator=",",
        try_parse_dates=False,
        infer_schema_length=0,
        schema_overrides=schema_overrides,
    )

    if sample_position not in {"head", "tail"}:
        raise ValueError("sample_position must be 'head' or 'tail'")

    if sample_position == "tail":
        df = lf.tail(n_rows).collect()
    else:
        df = lf.limit(n_rows).collect()

    print(f"Loaded orders sample from {raw_path}: rows={df.height}, cols={df.width}")
    print(f"Columns: {df.columns}")

    return df


def load_orders_parquet(path: Path) -> pl.DataFrame:
    return load_parquet(path, label="Orders parquet")


def load_products_csv(path: Path, separator: str = ";") -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Products file not found: {path}")
    return pl.read_csv(path, separator=separator)


def load_commerces_csv(path: Path, separator: str = ";") -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Commerces file not found: {path}")
    return pl.read_csv(path, separator=separator)


def save_parquet(df: pl.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"Saved parquet to {out_path}")
