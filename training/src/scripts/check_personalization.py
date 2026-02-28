"""
Analyze how personalized the recommendations are across kiosks.

Answers the question: "Are different kiosks getting different recommendations,
or is the model just recommending the same popular products to everyone?"

Usage:
    python -m training.src.scripts.check_personalization \
        --config training/configs/training_pipeline.yaml \
        --top-k 5 \
        --sample-kiosks 500

Metrics reported:
  - Unique recommendation sets: how many distinct top-K lists exist
  - Coverage: how many unique products appear across all top-K lists
  - Overlap with global top-K: % of a kiosk's recs that match the non-personalized baseline
  - Pairwise Jaccard: average overlap between random pairs of kiosks
  - Per-kiosk examples: side-by-side comparison of different kiosk recommendations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from training.src.config import load_yaml_config
from training.src.features import add_all_features, lgbm_feature_exprs
from training.src.io import load_orders_csv_sample, load_products_csv, load_commerces_csv
from training.src.logging_utils import setup_logging
from training.src.paths import EXTERNAL_DIR, MODELS_DIR
from training.src.steps.build_baskets import build_baskets
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.preprocessing import preprocess_orders
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.split_orders import split_orders_by_time

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _load_feature_list(model_path: Path) -> list[str]:
    feature_path = model_path.with_suffix(".features.json")
    if feature_path.exists():
        return json.loads(feature_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Feature list not found: {feature_path}")


def _predict_batched(
    ranker: lgb.Booster,
    df: pl.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    batch_size: int = 200_000,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, df.height, batch_size):
        chunk = (
            df.slice(start, batch_size)
            .select(lgbm_feature_exprs(feature_cols, categorical_cols))
            .to_numpy()
        )
        parts.append(np.asarray(ranker.predict(chunk)))
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


def _get_top_k_per_query(scored: pl.DataFrame, k: int) -> pl.DataFrame:
    """For each (kiosk, anchor), keep the top-K candidates by score."""
    return (
        scored
        .sort(["kiosk_id", "anchor_product_id", "score"], descending=[False, False, True])
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(k)
    )


# ------------------------------------------------------------------
# personalization metrics
# ------------------------------------------------------------------

def compute_personalization_report(
    top_k_df: pl.DataFrame,
    k: int,
    products: pl.DataFrame | None = None,
) -> None:
    """Compute and log all personalization metrics."""

    n_kiosks = top_k_df.select(pl.col("kiosk_id").n_unique()).item()
    n_queries = top_k_df.select(
        pl.struct(["kiosk_id", "anchor_product_id"]).n_unique()
    ).item()

    LOGGER.info("Personalization analysis (top-%d)", k)
    LOGGER.info("  Kiosks: %s, Queries (kiosk×anchor): %s", f"{n_kiosks:,}", f"{n_queries:,}")

    # ---- 1. Per-kiosk recommended product sets ----
    kiosk_rec_sets = (
        top_k_df
        .group_by("kiosk_id")
        .agg(pl.col("candidate_product_id").alias("rec_products"))
    )

    # ---- 2. Coverage: unique products recommended across all kiosks ----
    all_products = top_k_df.select("candidate_product_id").unique().height
    LOGGER.info("  Product coverage: %d unique products recommended", all_products)

    # ---- 3. Unique recommendation sets (per anchor) ----
    # For each anchor, how many distinct top-K lists exist?
    anchor_diversity = (
        top_k_df
        .sort(["anchor_product_id", "kiosk_id", "score"], descending=[False, False, True])
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(
            pl.col("candidate_product_id")
            .sort_by("score", descending=True)
            .head(k)
            .alias("rec_list")
        )
        .with_columns(
            pl.col("rec_list").list.sort().list.join(",").alias("rec_key")
        )
    )

    per_anchor_stats = (
        anchor_diversity
        .group_by("anchor_product_id")
        .agg(
            pl.col("kiosk_id").n_unique().alias("n_kiosks"),
            pl.col("rec_key").n_unique().alias("n_unique_lists"),
        )
        .with_columns(
            (pl.col("n_unique_lists") / pl.col("n_kiosks") * 100).alias("pct_unique")
        )
        .sort("n_kiosks", descending=True)
    )

    # Top-10 anchors by kiosk coverage
    top_anchors = per_anchor_stats.head(10)
    LOGGER.info("\n  Top-10 anchors by kiosk count:")
    LOGGER.info("  %-15s  %8s  %8s  %8s", "Anchor", "Kiosks", "Unique", "Unique%")
    LOGGER.info("  %s", "-" * 48)
    for row in top_anchors.iter_rows(named=True):
        LOGGER.info(
            "  %-15s  %8d  %8d  %7.1f%%",
            row["anchor_product_id"], row["n_kiosks"],
            row["n_unique_lists"], row["pct_unique"],
        )

    # Aggregate
    avg_unique_pct = per_anchor_stats.select(pl.col("pct_unique").mean()).item()
    median_unique_pct = per_anchor_stats.select(pl.col("pct_unique").median()).item()
    LOGGER.info("\n  Across all anchors: avg %.1f%% unique lists, median %.1f%%",
                avg_unique_pct, median_unique_pct)

    # ---- 4. Overlap with "global popularity" baseline ----
    # Build non-personalized baseline: for each anchor, what are the top-K candidates
    # by average score across all kiosks?
    global_top_k = (
        top_k_df
        .group_by(["anchor_product_id", "candidate_product_id"])
        .agg(pl.col("score").mean().alias("avg_score"))
        .sort(["anchor_product_id", "avg_score"], descending=[False, True])
        .group_by("anchor_product_id")
        .head(k)
        .select(["anchor_product_id", "candidate_product_id"])
        .with_columns(pl.lit(1).alias("in_global"))
    )

    overlap = (
        top_k_df
        .join(global_top_k, on=["anchor_product_id", "candidate_product_id"], how="left")
        .with_columns(pl.col("in_global").fill_null(0))
        .group_by(["kiosk_id", "anchor_product_id"])
        .agg(
            pl.len().alias("n_recs"),
            pl.col("in_global").sum().alias("n_overlap"),
        )
        .with_columns(
            (pl.col("n_overlap") / pl.col("n_recs") * 100).alias("overlap_pct")
        )
    )

    avg_overlap = overlap.select(pl.col("overlap_pct").mean()).item()
    p25_overlap = overlap.select(pl.col("overlap_pct").quantile(0.25)).item()
    p50_overlap = overlap.select(pl.col("overlap_pct").median()).item()
    p75_overlap = overlap.select(pl.col("overlap_pct").quantile(0.75)).item()
    LOGGER.info(
        "  Overlap with global baseline: mean=%.1f%%, p25=%.1f%%, median=%.1f%%, p75=%.1f%%",
        avg_overlap, p25_overlap, p50_overlap, p75_overlap,
    )

    # ---- 5. Pairwise Jaccard (sampled) ----
    # Pick a popular anchor, sample kiosk pairs, compute Jaccard
    most_popular_anchor = per_anchor_stats.row(0, named=True)["anchor_product_id"]
    anchor_recs = (
        anchor_diversity
        .filter(pl.col("anchor_product_id") == most_popular_anchor)
    )
    if anchor_recs.height >= 2:
        n_sample = min(200, anchor_recs.height)
        sampled = anchor_recs.sample(n=n_sample, seed=42)
        rec_lists = sampled.select("rec_list").to_series().to_list()

        jaccards = []
        n_pairs = min(1000, n_sample * (n_sample - 1) // 2)
        rng = np.random.RandomState(42)
        for _ in range(n_pairs):
            i, j = rng.choice(len(rec_lists), size=2, replace=False)
            set_i = set(rec_lists[i])
            set_j = set(rec_lists[j])
            if set_i or set_j:
                jaccards.append(len(set_i & set_j) / len(set_i | set_j))

        if jaccards:
            avg_j = np.mean(jaccards)
            LOGGER.info(
                "  Pairwise Jaccard (anchor=%s, %d pairs): mean=%.3f (0=fully unique, 1=identical)",
                most_popular_anchor, len(jaccards), avg_j,
            )

    # ---- 6. Example: show 3 kiosks side-by-side for the most popular anchor ----
    example_anchor = most_popular_anchor
    example_kiosks = (
        top_k_df
        .filter(pl.col("anchor_product_id") == example_anchor)
        .select("kiosk_id")
        .unique()
        .sample(n=min(3, top_k_df.select(pl.col("kiosk_id").n_unique()).item()), seed=123)
        .to_series()
        .to_list()
    )

    # Resolve product names if available
    prod_names: dict[str, str] = {}
    if products is not None and "productid" in products.columns and "name" in products.columns:
        prod_names = dict(
            products.select(
                pl.col("productid").cast(pl.Utf8),
                pl.col("name").cast(pl.Utf8),
            ).iter_rows()
        )

    def _fmt_product(pid: str) -> str:
        name = prod_names.get(pid, "")
        return f"{pid} ({name})" if name else pid

    LOGGER.info("\n  Example: anchor=%s, top-%d for 3 kiosks:", _fmt_product(example_anchor), k)
    for kiosk_id in example_kiosks:
        recs = (
            top_k_df
            .filter(
                (pl.col("kiosk_id") == kiosk_id)
                & (pl.col("anchor_product_id") == example_anchor)
            )
            .sort("score", descending=True)
            .head(k)
            .select("candidate_product_id")
            .to_series()
            .to_list()
        )
        LOGGER.info("    Kiosk %s:", kiosk_id[:12])
        for rank, pid in enumerate(recs, 1):
            LOGGER.info("      %d. %s", rank, _fmt_product(pid))


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Check recommendation personalization")
    parser.add_argument("--config", default="training/configs/training_pipeline.yaml")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K recommendations to analyze")
    parser.add_argument("--sample-kiosks", type=int, default=500,
                        help="Max kiosks to score (0 = all)")
    args = parser.parse_args()

    setup_logging("check_personalization")

    cfg_data = load_yaml_config(Path(args.config))

    # ---- Load data ----
    raw_paths = [Path(p) for p in cfg_data.get("raw_paths", [])]
    n_rows = int(cfg_data.get("n_rows", 500_000))
    sample_position = str(cfg_data.get("sample_position", "head"))
    products_path = Path(cfg_data.get("products_path", EXTERNAL_DIR / "products_v2.csv"))
    commerces_path = Path(cfg_data.get("commerces_path", EXTERNAL_DIR / "commerces.csv"))
    model_path = Path(cfg_data.get("model_path", MODELS_DIR / "lgbm_ranker.txt"))

    LOGGER.info("Loading data...")
    parts = []
    for rp in raw_paths:
        raw = load_orders_csv_sample(rp, n_rows=n_rows, sample_position=sample_position)
        parts.append(preprocess_orders(raw))
    orders = pl.concat(parts) if parts else pl.DataFrame()

    products = load_products_csv(products_path)
    commerces = load_commerces_csv(commerces_path)

    # Filter active kiosks
    if "active" in commerces.columns:
        active_kiosks = (
            commerces.filter(pl.col("active") == True)  # noqa: E712
            .select(pl.col("userid").cast(pl.Utf8).alias("kiosk_id"))
            .unique()
        )
        orders = orders.join(active_kiosks, on="kiosk_id", how="inner")

    # Use last portion of data (like test period)
    train_ratio = float(cfg_data.get("train_ratio", 0.6))
    val_ratio = float(cfg_data.get("val_ratio", 0.2))
    test_ratio = float(cfg_data.get("test_ratio", 0.2))
    train_orders, _, test_orders = split_orders_by_time(
        orders, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
    )

    LOGGER.info("Train orders: %s rows, Test orders: %s rows", f"{train_orders.height:,}", f"{test_orders.height:,}")

    # ---- Build candidates + features ----
    baskets = build_baskets(train_orders)

    min_cooc = int(cfg_data.get("min_cooc", 2))
    min_lift = float(cfg_data.get("min_lift", 1.2))
    top_k_cand = int(cfg_data.get("top_k", 100))

    candidates = generate_candidates(baskets, min_cooc=min_cooc)
    topk = select_top_k_candidates(candidates, k=top_k_cand, min_lift=min_lift)

    # Build queries from test orders (kiosk, anchor)
    queries = (
        build_baskets(test_orders)
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .unique()
    )

    # Sample kiosks
    if args.sample_kiosks > 0:
        unique_kiosks = queries.select("kiosk_id").unique()
        n_sample = min(args.sample_kiosks, unique_kiosks.height)
        sampled_kiosks = unique_kiosks.sample(n=n_sample, seed=42)
        queries = queries.join(sampled_kiosks, on="kiosk_id", how="inner")
        LOGGER.info("Sampled %d kiosks (%d queries)", n_sample, queries.height)

    # Feature table
    ft = build_feature_table(baskets=baskets, topk_candidates=topk, queries=queries)
    ft = add_all_features(ft, orders=train_orders, products=products, commerces=commerces)

    LOGGER.info("Feature table: %s rows", f"{ft.height:,}")

    # ---- Score ----
    ranker = lgb.Booster(model_file=str(model_path))
    feature_cols = _load_feature_list(model_path)
    categorical_cols = [c for c in ("channel", "region") if c in feature_cols]

    from training.src.pipelines.training import fill_missing_features
    numeric_cols = [c for c in feature_cols if c not in set(categorical_cols)]
    ft = fill_missing_features(ft, numeric_cols, categorical_cols)

    scores = _predict_batched(ranker, ft, feature_cols, categorical_cols)
    scored = ft.select(["kiosk_id", "anchor_product_id", "candidate_product_id"]).with_columns(
        pl.Series("score", scores)
    )

    # ---- Get top-K ----
    top_k_recs = _get_top_k_per_query(scored, k=args.top_k)
    LOGGER.info("Top-%d recommendations: %s rows", args.top_k, f"{top_k_recs.height:,}")

    # ---- Analyze ----
    compute_personalization_report(top_k_recs, k=args.top_k, products=products)


if __name__ == "__main__":
    main()
