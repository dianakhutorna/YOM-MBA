from __future__ import annotations

from pathlib import Path

import pytest

from training.src.io import load_orders_csv_sample
from training.src.paths import DATA_DIR
from training.src.steps.preprocessing import preprocess_orders
from training.src.steps.build_baskets import build_baskets
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.select_top_k_candidates import select_top_k_candidates

RAW_CSV = DATA_DIR / "2024-20250001_part_00-001.csv"


@pytest.fixture(scope="session")
def raw_orders_df():
    if not RAW_CSV.exists():
        pytest.skip(f"Missing raw CSV: {RAW_CSV}")
    return load_orders_csv_sample(RAW_CSV, n_rows=20_000)


@pytest.fixture(scope="session")
def cleaned_orders_df(raw_orders_df):
    return preprocess_orders(raw_orders_df)


@pytest.fixture(scope="session")
def baskets_df(cleaned_orders_df):
    baskets = build_baskets(cleaned_orders_df, min_items=2)
    if baskets.is_empty():
        pytest.skip("No baskets generated from sample data")
    return baskets


@pytest.fixture(scope="session")
def candidates_df(baskets_df):
    candidates = generate_candidates(baskets_df, min_cooc=2)
    if candidates.is_empty():
        pytest.skip("No candidates generated from sample baskets")
    return candidates


@pytest.fixture(scope="session")
def topk_df(candidates_df):
    topk = select_top_k_candidates(candidates_df, k=10, min_lift=1.5)
    if topk.is_empty():
        pytest.skip("No top-k candidates generated from sample candidates")
    return topk
