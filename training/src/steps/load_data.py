from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "training" / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"


def load_orders_sample(
    raw_path: Path,
    n_rows: int = 1_000_000,
) -> pl.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    lf = pl.scan_csv(
        raw_path,
        has_header=True,
        separator=",",
        try_parse_dates=False,     # ВАЖНО
        infer_schema_length=0,     # читаем как строки
    )


    df = lf.limit(n_rows).collect()

    print(f"Loaded orders sample: rows={df.height}, cols={df.width}")
    print(f"Columns: {df.columns}")

    return df


def save_interim_orders(
    df: pl.DataFrame,
    out_path: Path,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"Saved interim orders to {out_path}")

