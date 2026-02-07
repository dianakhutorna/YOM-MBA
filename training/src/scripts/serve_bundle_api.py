from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import polars as pl
from fastapi import FastAPI, Query

from training.src.config import load_yaml_config
from training.src.io import load_parquet, load_products_csv
from training.src.paths import EXTERNAL_DIR, INTERIM_DIR
from training.src.scripts.serve_bundle import build_bundle, _parse_list


app = FastAPI(title="Bundle Service", version="1.0")

_preds: pl.DataFrame | None = None
_fallback: pl.DataFrame | None = None
_products: pl.DataFrame | None = None


def _load_assets(config_path: Path) -> None:
    global _preds, _fallback, _products
    cfg = load_yaml_config(config_path) if config_path else {}

    predictions_path = Path(cfg.get("predictions_path", INTERIM_DIR / "predictions.parquet"))
    popularity_path = Path(cfg.get("popularity_path", INTERIM_DIR / "popularity_fallback.parquet"))
    products_path = Path(cfg.get("products_path", EXTERNAL_DIR / "products_v2.csv"))

    _preds = load_parquet(predictions_path, label="Predictions parquet")
    _fallback = load_parquet(popularity_path, label="Popularity fallback")
    if "category" not in _preds.columns:
        _products = load_products_csv(products_path)
    else:
        _products = None


@app.on_event("startup")
def _startup() -> None:
    config_env = os.environ.get("BUNDLE_CONFIG", "training/configs/serve_bundle.yaml")
    _load_assets(Path(config_env))


@app.get("/bundle")
def get_bundle(
    kiosk_id: str = Query(..., description="Kiosk ID"),
    anchor_product_id: str = Query(..., description="Anchor product ID"),
    included_products: Optional[str] = Query(None, description="CSV list"),
    excluded_products: Optional[str] = Query(None, description="CSV list"),
    allowed_categories: Optional[str] = Query(None, description="CSV list"),
    n_group_key: Optional[int] = Query(None, description="Max items per category"),
    n_min: int = Query(10, description="Min bundle size"),
    n_max: int = Query(20, description="Max bundle size"),
) -> dict:
    if _preds is None or _fallback is None:
        raise RuntimeError("Assets not loaded")

    final = build_bundle(
        _preds,
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
    )

    return {
        "kiosk_id": kiosk_id,
        "anchor_product_id": anchor_product_id,
        "items": final.to_dicts(),
    }
