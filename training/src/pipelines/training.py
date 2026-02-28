"""
End-to-end training pipeline for the bundle recommendation system.

Flow:
  1. Load raw CSV orders  →  preprocess → save interim parquet
  2. Time-split into train / val / test-eval / test-label
  3. Build baskets from train orders
  4. Generate MBA candidates  →  select top-K
  5. Build feature table + labels for each split
  6. Train LightGBM LambdaRank
  7. Offline evaluation on held-out test set
  8. Save model + feature list
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import json

import lightgbm as lgb
import polars as pl
import pandas as pd
import numpy as np

from training.src.config import load_yaml_config
from training.src.features import add_all_features, lgbm_feature_exprs
from training.src.io import (
    load_orders_csv_sample,
    load_products_csv,
    load_commerces_csv,
    save_parquet,
)
from training.src.logging_utils import setup_logging
from training.src.paths import RAW_DIR, INTERIM_DIR, EXTERNAL_DIR, MODELS_DIR
from training.src.steps.build_baskets import build_baskets
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.build_labels import build_labels, build_label_pairs
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.preprocessing import preprocess_orders
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.split_orders import split_orders_by_time
from training.src.steps.rank_eval_at_k import (
    hitrate_at_k_by_score,
    recall_at_k_by_score,
    ndcg_at_k_by_score,
    positives_at_k_by_score,
    precision_at_k_by_score,
    mrr_at_k_by_score,
)


LOGGER = logging.getLogger(__name__)

KEY_COLS = ["kiosk_id", "anchor_product_id", "candidate_product_id"]


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class TrainingPipelineConfig:
    raw_paths: list[Path]
    n_rows: int
    sample_position: str
    interim_path: Path
    products_path: Path
    commerces_path: Path
    model_path: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    train_label_ratio: float
    min_cooc: int
    min_lift: float
    top_k: int
    top_k_train: int | None
    label_window_days: int | None
    min_cooc_label: int
    label_kiosk_batch_size: int
    max_neg_per_group: int
    max_eval_queries: int
    eval_ks: list[int]
    predict_batch_size: int
    lgbm_params: dict
    num_boost_round: int
    early_stopping_rounds: int
    eval_log_path: Path | None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingPipelineConfig":
        data = load_yaml_config(path) if path.exists() else {}
        raw_paths = [Path(p) for p in data.get("raw_paths", [])]
        if not raw_paths:
            raw_paths = [RAW_DIR / "2024-20250001_part_00-001.csv"]
        return cls(
            raw_paths=raw_paths,
            n_rows=int(data.get("n_rows", 500_000)),
            sample_position=str(data.get("sample_position", "head")),
            interim_path=Path(data.get("interim_path", INTERIM_DIR / "orders_sample.parquet")),
            products_path=Path(data.get("products_path", EXTERNAL_DIR / "products_v2.csv")),
            commerces_path=Path(data.get("commerces_path", EXTERNAL_DIR / "commerces.csv")),
            model_path=Path(data.get("model_path", MODELS_DIR / "lgbm_ranker.txt")),
            train_ratio=float(data.get("train_ratio", 0.8)),
            val_ratio=float(data.get("val_ratio", 0.1)),
            test_ratio=float(data.get("test_ratio", 0.1)),
            train_label_ratio=float(data.get("train_label_ratio", 0.3)),
            min_cooc=int(data.get("min_cooc", 3)),
            min_lift=float(data.get("min_lift", 2.0)),
            top_k=int(data.get("top_k", 100)),
            top_k_train=int(data["top_k_train"]) if data.get("top_k_train") is not None else None,
            label_window_days=data.get("label_window_days", 7),
            min_cooc_label=int(data.get("min_cooc_label", 1)),
            label_kiosk_batch_size=int(data.get("label_kiosk_batch_size", 0)),
            max_neg_per_group=int(data.get("max_neg_per_group", 60)),
            max_eval_queries=int(data.get("max_eval_queries", 50_000)),
            eval_ks=[int(k) for k in data.get("eval_ks", [20])],
            predict_batch_size=int(data.get("predict_batch_size", 200_000)),
            lgbm_params=dict(data.get("lgbm_params", {})),
            num_boost_round=int(data.get("num_boost_round", 2000)),
            early_stopping_rounds=int(data.get("early_stopping_rounds", 100)),
            eval_log_path=Path(data["eval_log_path"]) if data.get("eval_log_path") else None,
        )


# ============================================================
# Helpers (module-level for testability)
# ============================================================

def add_query_id(df: pl.DataFrame) -> pl.DataFrame:
    """Add a synthetic ``query_id`` column for LambdaRank grouping."""
    return df.with_columns(
        (pl.col("kiosk_id").cast(pl.Utf8) + pl.lit("::") + pl.col("anchor_product_id").cast(pl.Utf8))
        .alias("query_id")
    )


def filter_good_queries(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only queries that have both positive and negative examples."""
    df = add_query_id(df)
    stats = (
        df.group_by("query_id")
        .agg(pl.len().alias("q_size"), pl.sum("label").alias("q_pos"))
    )
    good = stats.filter((pl.col("q_size") > 1) & (pl.col("q_pos") > 0)).select("query_id")
    out = df.join(good, on="query_id", how="inner").drop("query_id")
    removed = stats.height - good.height
    LOGGER.debug("Queries total: %s, kept: %s, removed: %s", stats.height, good.height, removed)
    return out


def sample_negatives(df: pl.DataFrame, max_neg_per_group: int, seed: int = 42) -> pl.DataFrame:
    """Down-sample negatives to at most *max_neg_per_group* per query."""
    if max_neg_per_group <= 0:
        return df
    df = add_query_id(df)
    if df.height == 0:
        return df.drop("query_id")
    cols_with_query = list(df.columns)
    cols = [c for c in cols_with_query if c != "query_id"]
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)
    if neg.height == 0:
        return df.drop("query_id")
    neg = (
        neg
        .with_columns(pl.arange(0, pl.len()).shuffle(seed=seed).alias("_rand"))
        .sort(["query_id", "_rand"])
        .group_by("query_id")
        .head(max_neg_per_group)
        .drop("_rand")
    )
    combined = pl.concat(
        [pos.select(cols_with_query), neg.select(cols_with_query)],
        how="vertical",
    )
    return combined.select(cols)


def shuffle_within_query(df: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    """Shuffle row order within each query group."""
    if df.height == 0:
        return df
    df = add_query_id(df)
    df = df.with_columns(pl.arange(0, pl.len()).shuffle(seed=seed).alias("_rand"))
    return df.sort(["query_id", "_rand"]).drop(["_rand", "query_id"])


def log_label_stats(name: str, df: pl.DataFrame) -> None:
    """Log class balance statistics for a labeled DataFrame."""
    if df.height == 0:
        LOGGER.info("%s label stats: rows=0 positives=0 ratio=0.0000 queries=0", name)
        return
    stats = df.select(
        pl.len().alias("rows"),
        pl.col("label").sum().alias("positives"),
        pl.struct(["kiosk_id", "anchor_product_id"]).n_unique().alias("queries"),
    ).row(0)
    rows, positives, queries = stats
    negatives = int(rows) - int(positives)
    ratio = float(positives) / float(rows) if rows else 0.0
    LOGGER.info(
        "%s label stats: rows=%s positives=%s negatives=%s ratio=%.4f queries=%s",
        name, rows, positives, negatives, ratio, queries,
    )


def fill_missing_features(
    df: pl.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pl.DataFrame:
    """Fill nulls: 0 for numeric, ``__MISSING__`` for categorical."""
    exprs: list[pl.Expr] = [pl.col(c).fill_null(0) for c in numeric_cols if c in df.columns]
    exprs.extend(
        pl.col(c).cast(pl.Utf8).fill_null("__MISSING__")
        for c in categorical_cols if c in df.columns
    )
    return df.with_columns(exprs) if exprs else df


def ensure_feature_columns(
    df: pl.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> pl.DataFrame:
    """Add any missing feature columns with default values."""
    missing_exprs: list[pl.Expr] = []
    cat_set = set(categorical_cols)
    for c in feature_cols:
        if c not in df.columns:
            if c in cat_set:
                missing_exprs.append(pl.lit("__MISSING__").alias(c))
            else:
                missing_exprs.append(pl.lit(0).alias(c))
    if missing_exprs:
        df = df.with_columns(missing_exprs)
    return df


def to_lgbm_arrays(
    df: pl.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
):
    """Convert a labeled DataFrame to ``(X, y, group)`` arrays for LightGBM."""
    sorted_df = add_query_id(df).sort("query_id")
    group = (
        sorted_df
        .group_by("query_id", maintain_order=True)
        .agg(pl.len().alias("q_size"))
        .select("q_size")
        .to_series()
        .to_numpy()
    )
    X = sorted_df.select(lgbm_feature_exprs(feature_cols, categorical_cols)).to_numpy()
    y = sorted_df.select(pl.col("label").cast(pl.Int8)).to_series().to_numpy()
    return X, y, group


def predict_scores_batched(
    model: lgb.Booster,
    df: pl.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    batch_size: int,
) -> np.ndarray:
    """Predict LightGBM scores in batches to limit memory usage."""
    if df.height == 0:
        return np.array([], dtype=np.float64)
    batch_size = max(1, int(batch_size))
    out: list[np.ndarray] = []
    for start in range(0, df.height, batch_size):
        chunk = (
            df.slice(start, batch_size)
            .select(lgbm_feature_exprs(feature_cols, categorical_cols))
            .to_numpy()
        )
        out.append(np.asarray(model.predict(chunk)))
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


def log_candidate_recall(
    eval_queries: pl.DataFrame,
    eval_candidates: pl.DataFrame,
    label_orders: pl.DataFrame,
    window_days: int | None,
    min_cooc_label: int,
    kiosk_batch_size: int = 0,
) -> None:
    """Log how many true-positive pairs are covered by the candidate set."""
    test_pairs = build_label_pairs(
        label_orders,
        window_days=window_days,
        min_cooc_label=min_cooc_label,
        kiosk_batch_size=kiosk_batch_size,
    )
    test_pairs = (
        test_pairs
        .select(KEY_COLS)
        .join(
            eval_queries.select(["kiosk_id", "anchor_product_id"]),
            on=["kiosk_id", "anchor_product_id"],
            how="inner",
        )
    )
    total_pos = test_pairs.height
    if total_pos == 0:
        LOGGER.info("Candidate recall: no positives in label window for eval queries.")
        return
    hits = (
        test_pairs
        .join(eval_candidates.select(KEY_COLS), on=KEY_COLS, how="inner")
        .height
    )
    recall = hits / total_pos
    LOGGER.info("Candidate recall: %.4f (%s/%s)", recall, hits, total_pos)


def filter_active_kiosks(
    orders: pl.DataFrame,
    commerces: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Filter orders and commerces to active kiosks only."""
    if "active" not in commerces.columns:
        LOGGER.warning("Column 'active' not found in commerces; skipping active kiosk filter.")
        return orders, commerces

    active_kiosks = (
        commerces
        .filter(pl.col("active") == True)  # noqa: E712
        .select(pl.col("userid").cast(pl.Utf8).alias("kiosk_id"))
        .drop_nulls()
        .unique()
    )
    rows_before = orders.height
    kiosks_before = orders.select(pl.col("kiosk_id").n_unique()).item()
    orders = orders.join(active_kiosks, on="kiosk_id", how="inner")
    commerces = commerces.filter(pl.col("active") == True)  # noqa: E712
    rows_after = orders.height
    kiosks_after = orders.select(pl.col("kiosk_id").n_unique()).item() if rows_after > 0 else 0
    LOGGER.info(
        "Filtered to active kiosks: rows %s -> %s, kiosks %s -> %s",
        rows_before, rows_after, kiosks_before, kiosks_after,
    )
    return orders, commerces


# ============================================================
# Pipeline
# ============================================================

def run(config: TrainingPipelineConfig) -> None:
    setup_logging("training")

    def _banner(title: str) -> None:
        line = "=" * 64
        LOGGER.info(line)
        LOGGER.info(title)
        LOGGER.info(line)

    # ---- STEP 1: LOAD DATA ----
    _banner("STEP 1 — LOAD DATA")

    per_file = max(1, config.n_rows // len(config.raw_paths))
    LOGGER.info(
        "Loading raw paths: %s (per_file=%s, sample_position=%s)",
        [str(p) for p in config.raw_paths], per_file, config.sample_position,
    )
    raw_frames = [
        load_orders_csv_sample(path, n_rows=per_file, sample_position=config.sample_position)
        for path in config.raw_paths
    ]
    raw_orders = pl.concat(raw_frames, how="vertical")
    if "order_dt" in raw_orders.columns:
        dt_stats = raw_orders.select(
            pl.col("order_dt").min().alias("min_dt"),
            pl.col("order_dt").max().alias("max_dt"),
        ).row(0)
        LOGGER.info("Raw orders date range: %s to %s", dt_stats[0], dt_stats[1])

    clean_orders = preprocess_orders(raw_orders)
    save_parquet(clean_orders, config.interim_path)

    products = load_products_csv(config.products_path)
    commerces = load_commerces_csv(config.commerces_path)
    clean_orders, commerces = filter_active_kiosks(clean_orders, commerces)

    # ---- STEP 2: SPLIT DATA ----
    _banner("STEP 2 — SPLIT DATA")

    train_orders, val_orders, test_orders = split_orders_by_time(
        clean_orders,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    for name, df in (("Train", train_orders), ("Val", val_orders), ("Test", test_orders)):
        if df.height == 0:
            LOGGER.info("%s orders: rows=0", name)
            continue
        dt_stats = df.select(
            pl.col("order_dt").min().alias("min_dt"),
            pl.col("order_dt").max().alias("max_dt"),
        ).row(0)
        LOGGER.info("%s orders date range: %s to %s (rows=%s)", name, dt_stats[0], dt_stats[1], df.height)

    # Sub-split training orders: earlier portion for features, later for labels.
    # This prevents label leakage (e.g. pop_store computed from the same data
    # that defines the positive labels).
    train_sorted = train_orders.sort("order_dt")
    unique_train_dts = (
        train_sorted.select(pl.col("order_dt").cast(pl.Datetime))
        .unique()
        .sort("order_dt")
        .to_series()
        .to_list()
    )
    if len(unique_train_dts) < 2:
        train_feat_orders = train_orders
        train_label_orders = train_orders
        LOGGER.warning("Train has <2 unique dates; can't sub-split for leakage prevention.")
    else:
        feat_end_idx = max(1, int(len(unique_train_dts) * (1.0 - config.train_label_ratio)))
        feat_end_idx = min(feat_end_idx, len(unique_train_dts) - 1)
        label_start_dt = unique_train_dts[feat_end_idx]
        train_feat_orders = train_sorted.filter(pl.col("order_dt") < label_start_dt)
        train_label_orders = train_sorted.filter(pl.col("order_dt") >= label_start_dt)

    for name, df in (("TrainFeat", train_feat_orders), ("TrainLabel", train_label_orders)):
        if df.height == 0:
            LOGGER.info("  %s orders: rows=0", name)
            continue
        dt_stats = df.select(
            pl.col("order_dt").min().alias("min_dt"),
            pl.col("order_dt").max().alias("max_dt"),
        ).row(0)
        LOGGER.info("  %s orders date range: %s to %s (rows=%s)", name, dt_stats[0], dt_stats[1], df.height)

    # ---- STEP 3: BUILD BASKETS ----
    _banner("STEP 3 — BUILD BASKETS")

    baskets_train = build_baskets(train_orders)

    # ---- STEP 4: GENERATE CANDIDATES (MBA) ----
    _banner("STEP 4 — GENERATE CANDIDATES")

    def _generate_topk(top_k_value: int) -> pl.DataFrame:
        candidates = generate_candidates(baskets_train, min_cooc=config.min_cooc)
        return select_top_k_candidates(candidates, k=top_k_value, min_lift=config.min_lift)

    top_k_train = int(config.top_k_train) if config.top_k_train is not None else int(config.top_k)
    top_k_train = max(1, min(top_k_train, config.top_k))
    topk_candidates_train = _generate_topk(top_k_train)

    def _add_features(ft: pl.DataFrame, feat_orders: pl.DataFrame) -> pl.DataFrame:
        return add_all_features(
            ft, orders=feat_orders, products=products, commerces=commerces,
        )

    # ---- STEP 5: BUILD FEATURES + LABELS ----
    _banner("STEP 5 — BUILD FEATURES + LABELS")

    # Suppress sub-module loggers during Step 5 (details are noise here;
    # the final summary table shows everything we need).
    _step5_loggers = [
        logging.getLogger("training.src.steps.build_baskets"),
        logging.getLogger("training.src.steps.build_labels"),
        logging.getLogger("training.src.steps.build_feature_table"),
    ]
    _step5_saved_levels = [lg.level for lg in _step5_loggers]
    for lg in _step5_loggers:
        lg.setLevel(logging.WARNING)

    def _build_queries(orders: pl.DataFrame) -> pl.DataFrame:
        return (
            build_baskets(orders)
            .select(["kiosk_id", "products"])
            .explode("products")
            .rename({"products": "anchor_product_id"})
            .unique()
        )

    def _build_labeled_split(
        *,
        split_name: str,
        query_orders: pl.DataFrame,
        label_orders: pl.DataFrame,
        feat_orders: pl.DataFrame,
        topk_candidates: pl.DataFrame,
        filter_good: bool,
        do_sample_negatives: bool,
        shuffle_seed: int | None,
        max_queries: int = 0,
    ) -> pl.DataFrame:
        # 1) Queries from query_orders (kiosk, anchor)
        queries_all = _build_queries(query_orders)

        # 2) Positive pairs from label_orders
        pos_pairs = build_label_pairs(
            label_orders,
            window_days=config.label_window_days,
            min_cooc_label=config.min_cooc_label,
            dt_col="order_dt",
            kiosk_batch_size=config.label_kiosk_batch_size,
        ).select(KEY_COLS)

        # 3) Keep only queries that have at least one positive
        pos_queries = pos_pairs.select(["kiosk_id", "anchor_product_id"]).unique()
        queries = queries_all.join(pos_queries, on=["kiosk_id", "anchor_product_id"], how="inner")
        n_total_q = queries_all.height
        n_pos_q = queries.height

        # 3b) Sample queries if exceeds max_queries (for memory-safe evaluation)
        sampled = False
        if max_queries > 0 and queries.height > max_queries:
            sampled = True
            queries = queries.sample(n=max_queries, seed=42)
            # Also trim pos_pairs to sampled queries
            pos_pairs = pos_pairs.join(
                queries.select(["kiosk_id", "anchor_product_id"]),
                on=["kiosk_id", "anchor_product_id"],
                how="inner",
            )
            n_pos_q = queries.height

        # Log concise query funnel
        if sampled:
            LOGGER.info(
                "%s: %s queries → %s sampled",
                split_name, f"{n_total_q:,}", f"{n_pos_q:,}",
            )
        else:
            LOGGER.info(
                "%s: %s queries → %s with positives (dropped %s)",
                split_name, f"{n_total_q:,}", f"{n_pos_q:,}",
                f"{n_total_q - n_pos_q:,}",
            )
        if queries.is_empty():
            LOGGER.info("%s: empty after filtering.", split_name)
            return pl.DataFrame(schema={**{c: pl.Utf8 for c in KEY_COLS}, "label": pl.Int8})

        # 4) Build candidate feature table for remaining queries
        base_ft = build_feature_table(
            baskets=baskets_train, topk_candidates=topk_candidates, queries=queries,
        )

        # 5) Assign labels by joining to positives
        labeled_keys = (
            base_ft.select(KEY_COLS)
            .join(
                pos_pairs.with_columns(pl.lit(1).cast(pl.Int8).alias("label")),
                on=KEY_COLS,
                how="left",
            )
            .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
        )

        # 6) Query-quality filter
        if filter_good:
            labeled_keys = filter_good_queries(labeled_keys)

        # 7) Negative sampling
        if do_sample_negatives:
            labeled_keys = sample_negatives(labeled_keys, config.max_neg_per_group, seed=42)

        # 8) Shuffle within query
        if shuffle_seed is not None:
            labeled_keys = shuffle_within_query(labeled_keys, seed=shuffle_seed)

        if labeled_keys.is_empty():
            LOGGER.info("Split %s: empty after sampling/shuffling.", split_name)
            return labeled_keys

        # 9) Build features only for selected rows.
        #    For large datasets, process in query batches to avoid OOM.
        MAX_ROWS_PER_BATCH = 5_000_000

        # When filtering/sampling happened, only keep selected rows
        if filter_good or do_sample_negatives:
            selected_keys = labeled_keys.select(KEY_COLS).unique()
        else:
            selected_keys = None  # all rows survive — skip redundant join

        total_rows = base_ft.height if selected_keys is None else labeled_keys.height

        if total_rows <= MAX_ROWS_PER_BATCH:
            # Small enough — single pass
            slim_ft = base_ft if selected_keys is None else base_ft.join(selected_keys, on=KEY_COLS, how="inner")
            slim_ft = _add_features(slim_ft, feat_orders)
            out = (
                slim_ft
                .join(labeled_keys, on=KEY_COLS, how="left")
                .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
            )
            del slim_ft
        else:
            # Large dataset — process in query batches
            unique_queries = (
                labeled_keys.select(["kiosk_id", "anchor_product_id"]).unique()
            )
            n_queries = unique_queries.height
            avg_group = total_rows / n_queries if n_queries else 1
            batch_q = max(1000, int(MAX_ROWS_PER_BATCH / avg_group))
            n_batches = (n_queries + batch_q - 1) // batch_q
            LOGGER.info(
                "%s: %s rows — feature batches: %d × ~%d queries",
                split_name, f"{total_rows:,}", n_batches, batch_q,
            )

            parts: list[pl.DataFrame] = []
            for i in range(0, n_queries, batch_q):
                q_batch = unique_queries.slice(i, min(batch_q, n_queries - i))
                lk_batch = labeled_keys.join(q_batch, on=["kiosk_id", "anchor_product_id"], how="inner")
                ft_batch = base_ft.join(lk_batch.select(KEY_COLS), on=KEY_COLS, how="inner")
                ft_batch = _add_features(ft_batch, feat_orders)
                out_batch = (
                    ft_batch
                    .join(lk_batch, on=KEY_COLS, how="left")
                    .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
                )
                parts.append(out_batch)
                del ft_batch, lk_batch, out_batch

            out = pl.concat(parts, how="vertical_relaxed")
            del parts
        del base_ft, labeled_keys

        return out

    labeled_train = _build_labeled_split(
        split_name="Train",
        query_orders=train_feat_orders,
        label_orders=train_label_orders,
        feat_orders=train_feat_orders,
        topk_candidates=topk_candidates_train,
        filter_good=True,
        do_sample_negatives=True,
        shuffle_seed=42,
    )
    labeled_val = _build_labeled_split(
        split_name="Val",
        query_orders=val_orders,
        label_orders=val_orders,
        feat_orders=train_orders,
        topk_candidates=topk_candidates_train,
        filter_good=True,
        do_sample_negatives=True,
        shuffle_seed=43,
    )
    labeled_test = _build_labeled_split(
        split_name="Test",
        query_orders=test_orders,
        label_orders=test_orders,
        feat_orders=train_orders,
        topk_candidates=topk_candidates_train,
        filter_good=False,
        do_sample_negatives=False,
        shuffle_seed=None,
        max_queries=config.max_eval_queries,
    )

    # ---- Detect feature columns ----
    non_feature_cols = {"kiosk_id", "anchor_product_id", "candidate_product_id", "label"}
    categorical_candidates = ("channel", "region")
    numeric_dtypes = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Boolean,
    }
    numeric_feature_cols = [
        c for c, dtype in labeled_train.schema.items()
        if c not in non_feature_cols and dtype in numeric_dtypes
    ]
    categorical_feature_cols = [
        c for c in categorical_candidates
        if c in labeled_train.columns and c not in non_feature_cols
    ]
    feature_cols = sorted(numeric_feature_cols) + sorted(categorical_feature_cols)

    labeled_train = fill_missing_features(labeled_train, numeric_feature_cols, categorical_feature_cols)
    labeled_val = fill_missing_features(labeled_val, numeric_feature_cols, categorical_feature_cols)
    labeled_test = fill_missing_features(labeled_test, numeric_feature_cols, categorical_feature_cols)

    def _split_summary_row(name: str, df: pl.DataFrame) -> tuple:
        if df.height == 0:
            return (name, 0, 0, 0, 0.0, 0, 0.0)
        rows, pos, queries = df.select(
            pl.len(), pl.col("label").sum(),
            pl.struct(["kiosk_id", "anchor_product_id"]).n_unique(),
        ).row(0)
        neg = rows - pos
        ratio = 100.0 * pos / rows if rows else 0.0
        avg_g = rows / queries if queries else 0.0
        return (name, rows, pos, neg, ratio, queries, avg_g)

    summary_rows = [
        _split_summary_row("Train", labeled_train),
        _split_summary_row("Val", labeled_val),
        _split_summary_row("Test", labeled_test),
    ]
    hdr = f"{'Split':<7s}  {'Rows':>10s}  {'Pos':>8s}  {'Neg':>10s}  {'Pos%':>6s}  {'Queries':>8s}  {'AvgGrp':>7s}"
    sep = "-" * len(hdr)
    lines = [hdr, sep]
    for name, rows, pos, neg, ratio, queries, avg_g in summary_rows:
        lines.append(
            f"{name:<7s}  {rows:>10,d}  {pos:>8,d}  {neg:>10,d}  {ratio:>5.1f}%  {queries:>8,d}  {avg_g:>7.1f}"
        )
    LOGGER.info("\nStep 5 summary:\n%s", "\n".join(lines))
    LOGGER.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # Restore sub-module loggers
    for lg, lvl in zip(_step5_loggers, _step5_saved_levels):
        lg.setLevel(lvl)

    # ---- STEP 6: TRAIN LGBM ----
    _banner("STEP 6 — TRAIN LGBM")

    X_train, y_train, g_train = to_lgbm_arrays(labeled_train, feature_cols, categorical_feature_cols)
    X_val, y_val, g_val = to_lgbm_arrays(labeled_val, feature_cols, categorical_feature_cols)
    if len(g_train) == 0 or len(g_val) == 0:
        raise ValueError("Empty train/val groups; adjust split ratios or data volume.")

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train, feature_name=feature_cols)
    valid_set = lgb.Dataset(X_val, label=y_val, group=g_val, feature_name=feature_cols, reference=train_set)

    eval_ks = sorted({k for k in config.eval_ks if k > 0}) or [20]
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": eval_ks,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 8,
        "min_data_in_leaf": 200,
        "min_gain_to_split": 0.1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "seed": 42,
        "verbosity": -1,
    }
    if config.lgbm_params:
        params.update(config.lgbm_params)

    evals_result: dict = {}
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ],
    )

    # Free training arrays — no longer needed
    del X_train, y_train, g_train, X_val, y_val, g_val
    del train_set, valid_set, labeled_train, labeled_val

    if config.eval_log_path:
        config.eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for dataset, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                for idx, val in enumerate(values, start=1):
                    rows.append({"iteration": idx, "dataset": dataset, "metric": metric_name, "value": val})
        if rows:
            pl.DataFrame(rows).write_csv(config.eval_log_path)
            LOGGER.info("Eval curves saved to %s", config.eval_log_path)

    if booster.best_iteration:
        best_iter = booster.best_iteration
        best = {}
        for dataset, metrics in evals_result.items():
            best[dataset] = {m: metrics[m][best_iter - 1] for m in metrics if len(metrics[m]) >= best_iter}
        LOGGER.info("Best iteration: %s", best_iter)
        LOGGER.info("Best metrics: %s", best)

    # Feature importance
    imp_df = pd.DataFrame(
        {"feature": feature_cols, "importance": booster.feature_importance(importance_type="gain")}
    ).sort_values("importance", ascending=False)
    LOGGER.info("Feature importance (gain):\n%s", imp_df.to_string(index=False))

    # ---- STEP 7: OFFLINE EVALUATION ----
    _banner("STEP 7 — OFFLINE EVALUATION")

    # Use the pre-built labeled_test from Step 5 (same features, same
    # candidate set, already labeled).  No need to rebuild anything.
    eval_labeled = labeled_test
    eval_labeled = ensure_feature_columns(eval_labeled, feature_cols, categorical_feature_cols)
    eval_labeled = fill_missing_features(eval_labeled, numeric_feature_cols, categorical_feature_cols)
    eval_labeled = eval_labeled.with_columns(pl.col("label").fill_null(0).cast(pl.Int8))

    # Candidate recall: how many true positives are in the candidate set
    eval_queries = eval_labeled.select(["kiosk_id", "anchor_product_id"]).unique()
    log_candidate_recall(
        eval_queries=eval_queries,
        eval_candidates=eval_labeled,
        label_orders=test_orders,
        window_days=config.label_window_days,
        min_cooc_label=config.min_cooc_label,
        kiosk_batch_size=config.label_kiosk_batch_size,
    )

    eval_scores = predict_scores_batched(
        booster, eval_labeled, feature_cols, categorical_feature_cols,
        batch_size=config.predict_batch_size,
    )
    # Keep only columns needed for metrics + score to free memory
    eval_scored = (
        eval_labeled.select(["kiosk_id", "anchor_product_id", "candidate_product_id", "label"])
        .with_columns(pl.Series("score", eval_scores))
    )
    del eval_labeled, eval_scores
    eval_scored = shuffle_within_query(eval_scored, seed=99)

    n_eval_queries = eval_scored.select(pl.struct(["kiosk_id", "anchor_product_id"]).n_unique()).item()
    n_eval_queries_with_pos = (
        eval_scored.filter(pl.col("label") == 1)
        .select(pl.struct(["kiosk_id", "anchor_product_id"]).n_unique()).item()
    )
    log_label_stats("Test", eval_scored)
    LOGGER.info(
        "Test queries total: %s, with >=1 positive: %s (%.1f%%)",
        n_eval_queries, n_eval_queries_with_pos,
        100.0 * n_eval_queries_with_pos / n_eval_queries if n_eval_queries else 0.0,
    )

    # Compute all metrics and display as a compact table
    metric_funcs = [
        ("HitRate",   hitrate_at_k_by_score),
        ("Recall",    recall_at_k_by_score),
        ("NDCG",      ndcg_at_k_by_score),
        ("MRR",       mrr_at_k_by_score),
        ("Precision", precision_at_k_by_score),
        ("Positives", positives_at_k_by_score),
    ]
    metric_results: dict[str, dict[int, float]] = {}
    for metric_name, metric_fn in metric_funcs:
        metric_results[metric_name] = {}
        for k in eval_ks:
            metric_results[metric_name][k] = metric_fn(eval_scored, k=k)

    # Build summary table
    k_headers = "  ".join(f"{'@' + str(k):>8s}" for k in eval_ks)
    table_lines = [f"{'Metric':<12s}  {k_headers}"]
    table_lines.append("-" * len(table_lines[0]))
    for metric_name in metric_results:
        vals = "  ".join(f"{metric_results[metric_name][k]:>8.4f}" for k in eval_ks)
        table_lines.append(f"{metric_name:<12s}  {vals}")
    LOGGER.info("\n[TEST] Offline evaluation results:\n%s", "\n".join(table_lines))

    # ---- STEP 8: SAVE ARTIFACTS ----
    _banner("STEP 8 — SAVE ARTIFACTS")

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(config.model_path))
    LOGGER.info("Model saved to %s", config.model_path)

    feature_path = config.model_path.with_suffix(".features.json")
    feature_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Feature list saved to %s", feature_path)
