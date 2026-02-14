from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from training.src.steps.generate_candidates import generate_candidates

LOGGER = logging.getLogger(__name__)

REQUIRED_BASKET_COLS: tuple[str, ...] = (
    "order_id",
    "products",
)


def _ensure_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def _empty_schema_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "anchor_product_id": pl.Utf8,
            "candidate_product_id": pl.Utf8,
            "cooc_count": pl.Int64,
            "anchor_count": pl.Int64,
            "candidate_count": pl.Int64,
            "support": pl.Float64,
            "confidence": pl.Float64,
            "lift": pl.Float64,
            "cooc_cosine_sim": pl.Float64,
            "embedding_cosine_sim": pl.Float64,
        }
    )


def generate_candidates_item2vec(
    baskets: pl.DataFrame,
    *,
    min_cooc: int = 1,
    top_k: int = 50,
    embedding_dim: int = 64,
    svd_n_iter: int = 10,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Item2Vec-like candidate generation via latent item embeddings.

    Steps:
    1. Build item-item co-occurrence matrix from baskets.
    2. Learn dense item embeddings with TruncatedSVD.
    3. Retrieve nearest neighbors by cosine in embedding space.
    4. Attach co-occurrence/MBA metrics when available.
    """

    _ensure_columns(baskets, REQUIRED_BASKET_COLS)
    if baskets.is_empty() or top_k <= 0:
        return _empty_schema_df()

    exploded = (
        baskets
        .select(["order_id", "products"])
        .explode("products")
        .rename({"products": "product_id"})
        .with_columns(pl.col("product_id").cast(pl.Utf8))
        .unique(subset=["order_id", "product_id"])
    )
    if exploded.is_empty():
        return _empty_schema_df()

    product_ids = exploded.select("product_id").unique().to_series().to_list()
    n_items = len(product_ids)
    if n_items < 2:
        return _empty_schema_df()

    id_to_idx = {pid: i for i, pid in enumerate(product_ids)}

    pairs = (
        exploded
        .join(exploded, on="order_id", how="inner")
        .rename({
            "product_id": "anchor_product_id",
            "product_id_right": "candidate_product_id",
        })
        .filter(pl.col("anchor_product_id") != pl.col("candidate_product_id"))
        .group_by(["anchor_product_id", "candidate_product_id"])
        .agg(pl.len().alias("cooc_count"))
        .filter(pl.col("cooc_count") >= min_cooc)
    )
    if pairs.is_empty():
        return _empty_schema_df()

    rows = np.array([id_to_idx[x] for x in pairs["anchor_product_id"].to_list()], dtype=np.int32)
    cols = np.array([id_to_idx[x] for x in pairs["candidate_product_id"].to_list()], dtype=np.int32)
    vals = np.array(pairs["cooc_count"].to_list(), dtype=np.float32)

    # Dense matrix is acceptable here because training runs on a sampled dataset.
    cooc = np.zeros((n_items, n_items), dtype=np.float32)
    cooc[rows, cols] = vals

    max_components = min(embedding_dim, n_items - 1)
    if max_components < 1:
        return _empty_schema_df()

    svd = TruncatedSVD(
        n_components=max_components,
        n_iter=svd_n_iter,
        random_state=random_state,
    )
    embeddings = svd.fit_transform(cooc)
    if embeddings.ndim != 2 or embeddings.shape[0] != n_items:
        return _empty_schema_df()

    n_neighbors = min(top_k + 1, n_items)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    out_rows: list[dict[str, float | str]] = []
    for anchor_idx, neigh_idx_list in enumerate(indices):
        anchor_id = product_ids[anchor_idx]
        for rank_pos, cand_idx in enumerate(neigh_idx_list.tolist()):
            if cand_idx == anchor_idx:
                continue
            candidate_id = product_ids[cand_idx]
            cosine = float(1.0 - distances[anchor_idx, rank_pos])
            out_rows.append(
                {
                    "anchor_product_id": anchor_id,
                    "candidate_product_id": candidate_id,
                    "embedding_cosine_sim": cosine,
                }
            )
            if len(out_rows) >= (anchor_idx + 1) * top_k:
                break

    if not out_rows:
        return _empty_schema_df()

    item2vec_top = pl.DataFrame(out_rows)

    mba_metrics = generate_candidates(baskets, min_cooc=min_cooc).select(
        [
            "anchor_product_id",
            "candidate_product_id",
            "cooc_count",
            "anchor_count",
            "candidate_count",
            "support",
            "confidence",
            "lift",
            "cooc_cosine_sim",
        ]
    )

    result = (
        item2vec_top
        .join(
            mba_metrics,
            on=["anchor_product_id", "candidate_product_id"],
            how="left",
        )
        .with_columns(
            [
                pl.col("cooc_count").fill_null(0).cast(pl.Int64),
                pl.col("anchor_count").fill_null(0).cast(pl.Int64),
                pl.col("candidate_count").fill_null(0).cast(pl.Int64),
                pl.col("support").fill_null(0.0).cast(pl.Float64),
                pl.col("confidence").fill_null(0.0).cast(pl.Float64),
                pl.col("lift").fill_null(0.0).cast(pl.Float64),
                pl.col("cooc_cosine_sim").fill_null(0.0).cast(pl.Float64),
                pl.col("embedding_cosine_sim").cast(pl.Float64),
            ]
        )
        .sort(["anchor_product_id", "embedding_cosine_sim"], descending=[False, True])
        .group_by("anchor_product_id")
        .head(top_k)
    )

    LOGGER.info("Item2Vec candidates shape: %s", result.shape)
    return result
