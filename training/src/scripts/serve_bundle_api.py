"""FastAPI service for bundle recommendations.

Start:
    BUNDLE_CONFIG=training/configs/serve_bundle.yaml \
    uvicorn training.src.scripts.serve_bundle_api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET /health          — readiness probe
    GET /bundle          — build a recommendation bundle
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import polars as pl
from fastapi import FastAPI, Query

from training.src.config import load_yaml_config
from training.src.io import load_parquet, load_products_csv
from training.src.paths import EXTERNAL_DIR, INTERIM_DIR
from training.src.scripts.serve_bundle import build_bundle, _parse_list

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory stores (populated at startup)
# ---------------------------------------------------------------------------
_preds_index: dict[tuple[str, str], pl.DataFrame] | None = None
_fallback: pl.DataFrame | None = None
_cat_fallback: pl.DataFrame | None = None
_global_fallback: pl.DataFrame | None = None
_products: pl.DataFrame | None = None
_product_names: dict[str, str] = {}


def _load_assets() -> None:
    global _preds_index, _fallback, _cat_fallback, _global_fallback
    global _products, _product_names

    config_path = Path(os.environ.get(
        "BUNDLE_CONFIG", "training/configs/serve_bundle.yaml",
    ))
    cfg = load_yaml_config(config_path) if config_path.exists() else {}

    predictions_path = Path(cfg.get("predictions_path", INTERIM_DIR / "predictions.parquet"))
    popularity_path = Path(cfg.get("popularity_path", INTERIM_DIR / "popularity_fallback.parquet"))
    category_fallback_path = Path(cfg.get("category_fallback_path", INTERIM_DIR / "category_fallback.parquet"))
    global_fallback_path = Path(cfg.get("global_fallback_path", INTERIM_DIR / "global_fallback.parquet"))
    products_path = Path(cfg.get("products_path", EXTERNAL_DIR / "products_v2.csv"))

    preds = load_parquet(predictions_path, label="Predictions parquet")
    _fallback = load_parquet(popularity_path, label="Anchor fallback")
    _cat_fallback = (
        load_parquet(category_fallback_path, label="Category fallback")
        if category_fallback_path.exists() else None
    )
    _global_fallback = (
        load_parquet(global_fallback_path, label="Global fallback")
        if global_fallback_path.exists() else None
    )
    _products = load_products_csv(products_path)  # blocked products are filtered out

    # Exclude blocked products from pre-computed predictions and fallback
    _active_product_ids = set(
        _products.select(pl.col("productid").cast(pl.Utf8)).to_series().to_list()
    )
    if _active_product_ids:
        for col in ("anchor_product_id", "candidate_product_id"):
            if col in preds.columns:
                preds = preds.filter(pl.col(col).is_in(_active_product_ids))
        if "candidate_product_id" in _fallback.columns:
            _fallback = _fallback.filter(
                pl.col("candidate_product_id").is_in(_active_product_ids)
            )
        LOGGER.info(
            "After blocked-product filter: preds=%s rows, fallback=%s rows",
            preds.height, _fallback.height,
        )

    # Build product-name lookup for enriching responses
    if _products is not None and "name" in _products.columns:
        _product_names = dict(zip(
            _products.select(pl.col("productid").cast(pl.Utf8)).to_series().to_list(),
            _products["name"].to_list(),
        ))

    # Build dict-index: (kiosk_id, anchor) → pre-filtered DataFrame
    # Trades ~2× RAM for O(1) lookup instead of O(N) scan on 26.9M rows
    _preds_index = defaultdict(lambda: pl.DataFrame(schema={
        "kiosk_id": pl.Utf8, "anchor_product_id": pl.Utf8,
        "candidate_product_id": pl.Utf8, "score": pl.Float64,
    }))
    LOGGER.info("Building prediction index …")
    t0 = time.perf_counter()
    for (kid, aid), group_df in preds.group_by(["kiosk_id", "anchor_product_id"]):
        _preds_index[(kid, aid)] = group_df
    elapsed = time.perf_counter() - t0
    LOGGER.info(
        "Index built: %s keys in %.1f s  (%.0f MB in RAM)",
        len(_preds_index), elapsed,
        preds.estimated_size("mb"),
    )


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_assets()
    yield


app = FastAPI(title="Bundle Service", version="1.1", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Readiness / liveness probe."""
    ready = _preds_index is not None and _fallback is not None
    return {"status": "ok" if ready else "loading"}


@app.get("/bundle")
def get_bundle(
    kiosk_id: str = Query(..., description="Kiosk ID"),
    anchor_product_id: str = Query(..., description="Anchor product ID"),
    included_products: Optional[str] = Query(None, description="CSV list of product IDs to force-include"),
    excluded_products: Optional[str] = Query(None, description="CSV list of product IDs to exclude"),
    allowed_categories: Optional[str] = Query(None, description="CSV list of allowed categories"),
    n_group_key: Optional[int] = Query(None, description="Max items per category"),
    n_min: int = Query(10, description="Min bundle size"),
    n_max: int = Query(20, description="Max bundle size"),
) -> dict:
    if _preds_index is None or _fallback is None or _products is None:
        raise RuntimeError("Assets not loaded — check /health")

    t0 = time.perf_counter()

    # O(1) lookup instead of scanning 26.9M rows
    preds_slice = _preds_index[(kiosk_id, anchor_product_id)]

    final = build_bundle(
        preds_slice,
        _fallback,
        _products,
        kiosk_id=kiosk_id,
        anchor_product_id=anchor_product_id,
        included_products=_parse_list(included_products),
        excluded_products=_parse_list(excluded_products),
        allowed_categories=_parse_list(allowed_categories),
        n_group_key=n_group_key,
        n_min=max(1, n_min),
        n_max=max(n_min, n_max),
        category_fallback=_cat_fallback,
        global_fallback=_global_fallback,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Enrich with product names
    items = final.to_dicts()
    for item in items:
        pid = item.get("candidate_product_id", "")
        item["candidate_name"] = _product_names.get(pid, "")

    return {
        "kiosk_id": kiosk_id,
        "anchor_product_id": anchor_product_id,
        "n_items": len(items),
        "latency_ms": round(elapsed_ms, 1),
        "items": items,
    }
