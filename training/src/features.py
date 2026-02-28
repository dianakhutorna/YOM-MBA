"""
Feature utilities for the bundle recommendation LightGBM model.

This module provides:
  - ``lgbm_feature_exprs``  — shared encoding (training + inference)
  - ``add_all_features``    — single entry-point for feature augmentation
"""

from __future__ import annotations

import polars as pl

from training.src.steps.add_features import add_features


# ============================================================
# Shared encoding — MUST be used by both training and inference
# to guarantee identical feature representation.
# ============================================================

def lgbm_feature_exprs(
    feature_cols: list[str],
    categorical_cols: list[str],
) -> list[pl.Expr]:
    """
    Build Polars expressions that convert feature columns to the
    numeric representation expected by LightGBM.

    Categorical columns are hashed to UInt64 then cast to Float64.
    Numeric columns are cast to Float64 with null → 0.
    """
    exprs: list[pl.Expr] = []
    cat_set = set(categorical_cols)
    for c in feature_cols:
        if c in cat_set:
            exprs.append(
                pl.col(c)
                .cast(pl.Utf8)
                .fill_null("__MISSING__")
                .hash(seed=42, seed_1=43, seed_2=44, seed_3=45)
                .cast(pl.Float64)
            )
        else:
            exprs.append(pl.col(c).fill_null(0).cast(pl.Float64))
    return exprs


# ============================================================
# Feature orchestration
# ============================================================

def add_all_features(
    feature_table: pl.DataFrame,
    *,
    orders: pl.DataFrame,
    products: pl.DataFrame | None = None,
    commerces: pl.DataFrame | None = None,
    config: object | None = None,      # kept for backward-compat; ignored
) -> pl.DataFrame:
    """Add all features to the base feature table."""
    return add_features(
        feature_table,
        orders=orders,
        products=products,
        commerces=commerces,
    )
