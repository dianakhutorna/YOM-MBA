from __future__ import annotations

import logging
import os
from typing import Sequence

import polars as pl

LOGGER = logging.getLogger(__name__)

REQUIRED_TEST_ORDER_COLS: tuple[str, ...] = (
    "kiosk_id",
    "order_id",
    "product_id",
)

REQUIRED_FEATURE_COLS: tuple[str, ...] = (
    "kiosk_id",
    "anchor_product_id",
    "candidate_product_id",
)


# ============================================================
# Utilities
# ============================================================

def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def _empty_pairs_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "kiosk_id": pl.Utf8,
            "anchor_product_id": pl.Utf8,
            "candidate_product_id": pl.Utf8,
            "cooc_count": pl.Int64,
        }
    )


# ============================================================
# Pair construction (core heavy logic)
# ============================================================

def _build_test_pairs(test_orders: pl.DataFrame) -> pl.DataFrame:
    test_baskets = (
        test_orders
        .group_by(["kiosk_id", "order_id"])
        .agg(pl.col("product_id").unique().alias("products"))
        .filter(pl.col("products").list.len() > 1)
    )

    return (
        test_baskets
        .select(["kiosk_id", "order_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .join(
            test_baskets
            .select(["kiosk_id", "order_id", "products"])
            .explode("products")
            .rename({"products": "candidate_product_id"}),
            on=["kiosk_id", "order_id"],
        )
        .filter(pl.col("anchor_product_id") != pl.col("candidate_product_id"))
        .group_by(["kiosk_id", "anchor_product_id", "candidate_product_id"])
        .agg(pl.len().alias("cooc_count"))
    )


def _build_test_pairs_window(
    test_orders: pl.DataFrame,
    *,
    window_days: int,
    dt_col: str,
) -> pl.DataFrame:
    window = pl.duration(days=window_days)

    anchor_events = (
        test_orders
        .select(["kiosk_id", "product_id", dt_col])
        .rename({"product_id": "anchor_product_id", dt_col: "anchor_dt"})
    )

    candidate_events = (
        test_orders
        .select(["kiosk_id", "product_id", dt_col])
        .rename({"product_id": "candidate_product_id", dt_col: "candidate_dt"})
    )

    return (
        anchor_events
        .join(candidate_events, on="kiosk_id", how="inner")
        .filter(pl.col("candidate_product_id") != pl.col("anchor_product_id"))
        .filter(
            (pl.col("candidate_dt") >= pl.col("anchor_dt")) &
            (pl.col("candidate_dt") <= pl.col("anchor_dt") + window)
        )
        .select(["kiosk_id", "anchor_product_id", "candidate_product_id"])
        .group_by(["kiosk_id", "anchor_product_id", "candidate_product_id"])
        .agg(pl.len().alias("cooc_count"))
    )


# ============================================================
# Batch resolution (memory-safe scaling)
# ============================================================

def _detect_total_ram_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int):
            return pages * page_size
    except Exception:
        pass
    return 8 * 1024**3  # conservative fallback


def _resolve_kiosk_batch_size(
    test_orders: pl.DataFrame,
    *,
    window_days: int | None,
    kiosk_batch_size: int | None,
) -> int:
    if kiosk_batch_size is not None and kiosk_batch_size > 0:
        return int(kiosk_batch_size)

    n_rows = int(test_orders.height)
    if n_rows == 0:
        return 1

    n_kiosks = int(
        test_orders
        .select(pl.col("kiosk_id").n_unique())
        .item()
    )

    if n_kiosks <= 1:
        return 1

    avg_rows_per_kiosk = max(1.0, n_rows / n_kiosks)
    total_ram_gb = _detect_total_ram_bytes() / float(1024**3)

    if window_days is None:
        target_rows_per_batch = max(20_000.0, min(600_000.0, total_ram_gb * 120_000.0))
    else:
        target_rows_per_batch = max(5_000.0, min(120_000.0, total_ram_gb * 25_000.0))

    auto_batch = int(target_rows_per_batch / avg_rows_per_kiosk)
    auto_batch = max(1, min(auto_batch, n_kiosks))

    LOGGER.info(
        "Auto-selected label kiosk batch size=%s (rows=%s kiosks=%s avg_rows_per_kiosk=%.1f ram_gb=%.1f window_days=%s)",
        auto_batch,
        n_rows,
        n_kiosks,
        avg_rows_per_kiosk,
        total_ram_gb,
        window_days,
    )

    return auto_batch


def _build_test_pairs_batched(
    test_orders: pl.DataFrame,
    *,
    window_days: int | None,
    dt_col: str,
    kiosk_batch_size: int | None,
) -> pl.DataFrame:
    if test_orders.is_empty():
        return _empty_pairs_df()

    kiosk_ids = (
        test_orders
        .select(pl.col("kiosk_id").cast(pl.Utf8))
        .unique()
        .to_series()
        .to_list()
    )

    if not kiosk_ids:
        return _empty_pairs_df()

    batch_size = _resolve_kiosk_batch_size(
        test_orders,
        window_days=window_days,
        kiosk_batch_size=kiosk_batch_size,
    )

    parts: list[pl.DataFrame] = []
    total_batches = (len(kiosk_ids) + batch_size - 1) // batch_size

    LOGGER.info(
        "Building label pairs in kiosk batches: kiosks=%s batch_size=%s batches=%s window_days=%s",
        len(kiosk_ids),
        batch_size,
        total_batches,
        window_days,
    )

    test_orders = test_orders.with_columns(pl.col("kiosk_id").cast(pl.Utf8))

    for start in range(0, len(kiosk_ids), batch_size):
        chunk = kiosk_ids[start:start + batch_size]
        chunk_orders = test_orders.filter(pl.col("kiosk_id").is_in(chunk))

        if chunk_orders.is_empty():
            continue

        if window_days is None:
            pairs = _build_test_pairs(chunk_orders)
        else:
            pairs = _build_test_pairs_window(
                chunk_orders,
                window_days=window_days,
                dt_col=dt_col,
            )

        if not pairs.is_empty():
            parts.append(pairs)

    if not parts:
        return _empty_pairs_df()

    return (
        pl.concat(parts, how="vertical_relaxed")
        .group_by(["kiosk_id", "anchor_product_id", "candidate_product_id"])
        .agg(pl.col("cooc_count").sum().cast(pl.Int64).alias("cooc_count"))
    )


# ============================================================
# Public API
# ============================================================

def build_label_pairs(
    test_orders: pl.DataFrame,
    *,
    window_days: int | None = None,
    min_cooc_label: int | None = None,
    dt_col: str = "order_dt",
    kiosk_batch_size: int | None = 0,
) -> pl.DataFrame:
    """
    Return positive (kiosk, anchor, candidate) pairs with cooc_count.
    """

    _ensure_columns(test_orders, REQUIRED_TEST_ORDER_COLS)

    if window_days is not None and dt_col not in test_orders.columns:
        raise ValueError(f"Missing datetime column for windowed labels: {dt_col}")

    if min_cooc_label is None or min_cooc_label < 1:
        min_cooc_label = 1

    test_pairs = _build_test_pairs_batched(
        test_orders,
        window_days=window_days,
        dt_col=dt_col,
        kiosk_batch_size=kiosk_batch_size,
    )

    if min_cooc_label > 1:
        test_pairs = test_pairs.filter(pl.col("cooc_count") >= min_cooc_label)

    return test_pairs


def build_labels(
    feature_table: pl.DataFrame,
    test_orders: pl.DataFrame,
    *,
    window_days: int | None = None,
    min_cooc_label: int | None = None,
    dt_col: str = "order_dt",
    kiosk_batch_size: int | None = 0,
) -> pl.DataFrame:
    """
    Build binary labels for ranking:
    1 if anchor & candidate co-occur in test period.
    """

    _ensure_columns(feature_table, REQUIRED_FEATURE_COLS)

    LOGGER.info("Building labels from feature table: %s", feature_table.shape)

    test_pairs = build_label_pairs(
        test_orders,
        window_days=window_days,
        min_cooc_label=min_cooc_label,
        dt_col=dt_col,
        kiosk_batch_size=kiosk_batch_size,
    )

    test_pairs = (
        test_pairs
        .select(["kiosk_id", "anchor_product_id", "candidate_product_id"])
        .with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    )

    labeled = (
        feature_table
        .join(
            test_pairs,
            on=["kiosk_id", "anchor_product_id", "candidate_product_id"],
            how="left",
        )
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )

    return labeled
