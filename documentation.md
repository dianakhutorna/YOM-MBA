# YOM Bundle Recommender — Technical Documentation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Flow](#2-data-flow)
3. [Training Pipeline](#3-training-pipeline)
4. [Batch Scoring (Inference)](#4-batch-scoring-inference)
5. [Serving Layer](#5-serving-layer)
6. [Features](#6-features)
7. [Business Rules](#7-business-rules)
8. [Fallback System](#8-fallback-system)
9. [File Reference](#9-file-reference)
10. [Configuration Reference](#10-configuration-reference)
11. [Testing](#11-testing)
12. [Deployment](#12-deployment)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. System Overview

### What the system does

Given a **kiosk** (point-of-sale) and an **anchor product** (the product a customer is currently viewing), the system returns a ranked list of **candidate products** to recommend as a bundle. For example: customer picks "Leche Frutil" → system recommends "Leche Blanca", "Nectar UHT", "Leche Chocolate", etc.

### Three-stage architecture

The system is split into three independent stages that run at different frequencies:

```
┌────────────────────────┐
│  STAGE 1: TRAINING     │  Run: monthly or when data/model changes
│  training.py           │
│                        │
│  Input:  raw order CSVs, product catalog, kiosk metadata
│  Output: lgbm_ranker.txt (model) + lgbm_ranker.features.json
└────────────┬───────────┘
             │ model file
             ▼
┌────────────────────────┐
│  STAGE 2: SCORING      │  Run: daily or weekly
│  generate_predictions  │
│                        │
│  Input:  model + recent orders (90-day window)
│  Output: 4 parquet files (predictions + 3 fallbacks)
└────────────┬───────────┘
             │ parquet files
             ▼
┌────────────────────────┐
│  STAGE 3: SERVING      │  Run: 24/7 (FastAPI)
│  serve_bundle_api.py   │
│                        │
│  Input:  4 parquet files (loaded at startup)
│  Output: JSON recommendations per request (~2 ms)
└────────────────────────┘
```

### Why this architecture?

- **No online inference** — LightGBM is never called at request time. All scoring is done ahead of time. Serving is a dictionary lookup.
- **Decoupled stages** — Training can be done on a powerful machine; serving runs on minimal hardware.
- **Never-empty results** — Four-level fallback guarantees a recommendation for any (kiosk, anchor) pair, even completely unknown ones.

### Technology stack

| Component | Technology |
|-----------|-----------|
| Data processing | Polars (columnar DataFrame) |
| Model | LightGBM LambdaRank (learning-to-rank) |
| Candidate generation | Market Basket Analysis (MBA) — co-occurrence, lift, cosine similarity |
| Serving | FastAPI + Uvicorn |
| Storage | Parquet files (no database) |
| Config | YAML |
| Testing | pytest + custom smoke tests |

---

## 2. Data Flow

### Input data

| File | Location | Description |
|------|----------|-------------|
| Order CSVs | `training/data/raw/*.csv` | Raw transaction data: `order_id`, `kiosk_id`, `product_id`, `date`, `quantity` |
| Products | `training/data/external/products_v2.csv` | Product catalog: `productid`, `name`, `category` |
| Commerces | `training/data/external/commerces.csv` | Kiosk metadata: `kiosk_id`, `channel`, `region`, active flag |

### Intermediate artifacts

| File | Created by | Description |
|------|-----------|-------------|
| `orders_sample.parquet` | Training pipeline | Preprocessed and deduplicated orders |
| `train/val/test batches` | Training pipeline | Time-split order subsets (temporary) |

### Output artifacts

| File | Created by | Size | Description |
|------|-----------|------|-------------|
| `lgbm_ranker.txt` | Training | ~200 KB | Trained LightGBM model |
| `lgbm_ranker.features.json` | Training | ~1 KB | Ordered list of feature columns the model expects |
| `predictions.parquet` | Batch scoring | ~256 MB | Scored (kiosk, anchor, candidate, score) — top-20 per query. ~26.9M rows |
| `popularity_fallback.parquet` | Batch scoring | ~63 KB | Per-anchor MBA co-purchase rankings. ~6K rows, ~315 anchors |
| `category_fallback.parquet` | Batch scoring | ~4 KB | Per-category popular products. ~149 items across ~14 categories |
| `global_fallback.parquet` | Batch scoring | ~2 KB | Top-20 most purchased products overall |

### Current coverage (90-day window)

- **3.07M orders** processed
- **41,337 active kiosks** with predictions
- **315 anchor products** with MBA candidates
- **83.6% kiosk coverage** (40,427 / 48,342 active kiosks)

---

## 3. Training Pipeline

**File:** `training/src/pipelines/training.py` (~872 lines)
**CLI:** `training/src/scripts/run_training_pipeline.py`
**Config:** `training/configs/training_pipeline.yaml`

### Pipeline steps

```
Step 1: Load & Preprocess
    Raw CSV(s) → clean columns, cast types, deduplicate → orders_sample.parquet
    Filter to active kiosks (from commerces.csv)

Step 2: Time-Based Split
    Sort by date → split into train (80%) / val (10%) / test (10%)
    Sub-split train into:
      - feature_orders (first 70% of train) — used for MBA + features
      - label_orders   (last 30% of train)  — used for positive labels
    This prevents label leakage: labels come from different time period

Step 3: Build Baskets
    Group orders by (kiosk_id, date) → list of product_ids per basket
    A basket = one kiosk's purchases on one day

Step 4: Generate Candidates (MBA)
    Explode baskets into (anchor, candidate) product pairs
    Compute for each pair:
      - cooc_count:  how many baskets contain both products
      - support:     cooc_count / total_baskets
      - lift:        support(A,B) / (support(A) * support(B))
      - cosine_sim:  cooc_count / sqrt(count(A) * count(B))
    Filter: min_cooc >= 2, min_lift >= 1.2
    Keep top-100 candidates per anchor (by lift, then cosine_sim)

Step 5: Build Feature Table + Labels
    For each split (train, val, test):
      - Cross-join kiosks × anchors × top-K candidates → (kiosk, anchor, candidate) rows
      - Add 8 features (see Features section)
      - Add binary labels from co-purchase pairs
      - Sample negatives (max 20 per query group)
      - Filter to queries with both positive and negative examples
      - Assign synthetic query_id for LambdaRank grouping
      - Shuffle within query groups (LightGBM requirement)

Step 6: Train LightGBM
    Objective: lambdarank (NDCG optimization)
    Train with validation-based early stopping (patience=100)
    Log metrics: NDCG@5, NDCG@10, NDCG@20, NDCG@50

Step 7: Offline Evaluation
    On test set, compute:
      - HitRate@K — fraction of queries with at least 1 positive in top-K
      - NDCG@K — normalized discounted cumulative gain
      - Recall@K — fraction of positives captured in top-K
      - MRR@K — mean reciprocal rank of first positive
      - Precision@K — fraction of top-K that are positive

Step 8: Save Artifacts
    Save lgbm_ranker.txt and lgbm_ranker.features.json
```

### Label leakage prevention

The `train_label_ratio` parameter (default 0.3) splits the training period into two non-overlapping windows:

```
|------ train (80%) ------|-- val (10%) --|-- test (10%) --|
|-- features (70%) --|-- labels (30%) --|
```

Features are computed from the first 70% of the training period. Labels come from the last 30%. This ensures the model doesn't memorize features from the same orders used for labels.

### LightGBM hyperparameters

```yaml
lgbm_params:
  learning_rate: 0.02
  num_leaves: 15
  max_depth: 5
  min_data_in_leaf: 1000
  min_gain_to_split: 0.5
  lambda_l1: 2.0
  lambda_l2: 10.0
  feature_fraction: 0.55
  bagging_fraction: 0.6
  bagging_freq: 1
```

Conservative regularization (high L1/L2, low num_leaves/depth) to prevent overfitting on sparse co-purchase data.

---

## 4. Batch Scoring (Inference)

**File:** `training/src/scripts/generate_predictions.py` (~310 lines)
**Config:** `training/configs/generate_predictions.yaml`

### Process

```
1. Load trained model (lgbm_ranker.txt) + feature list (features.json)
2. Load orders_sample.parquet + products + commerces
3. Filter to active kiosks (commerces table)
4. Select time window: last 90 days of orders
5. Build baskets → MBA candidates (top-50 per anchor)
6. Build feature table: all (kiosk, anchor, candidate) triples
7. Add 8 features (same as training)
8. Align columns to model's feature list:
   - Drop extra columns
   - Add missing columns as zeros (with warning)
9. Batch-predict with LightGBM (batch_size=200,000)
10. Keep top-20 candidates per (kiosk, anchor) query
11. Save predictions.parquet with product names/categories

12. Build per-anchor fallback:
    - From MBA topk_candidates, sorted by cooc_cosine_sim
    - Top-20 per anchor
    - Scores normalized to model score range [min, max]
    → popularity_fallback.parquet

13. Build category fallback:
    - Count product purchases per category
    - Top-N products per category
    - Scores normalized to model range
    → category_fallback.parquet

14. Build global fallback:
    - Top-20 most purchased products overall
    - Scores normalized to model range
    → global_fallback.parquet
```

### Key configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `inference_last_n_days` | 90 | How far back to look for orders. 90 days gives 83.6% kiosk coverage. |
| `top_k_candidates` | 50 | MBA candidates per anchor at inference. Lower than training (100) to keep catalog size manageable. |
| `catalog_top_k` | 20 | Final number of recommendations stored per (kiosk, anchor) pair. |
| `predict_batch_size` | 200,000 | Rows per LightGBM predict call (memory safety). |

### Runtime

~16 minutes on MacBook Air M2. Most time is spent on feature table construction and LightGBM prediction (66.4M feature rows before top-K filtering).

---

## 5. Serving Layer

### serve_bundle.py — Core logic

**File:** `training/src/scripts/serve_bundle.py` (~411 lines)

The core function is `build_bundle()`:

```python
def build_bundle(
    preds, fallback, products, *,
    kiosk_id, anchor_product_id,
    included_products, excluded_products, allowed_categories,
    n_group_key, n_min, n_max,
    category_fallback=None, global_fallback=None,
) -> pl.DataFrame
```

It implements the 4-level fallback chain (see Fallback System section) and delegates business rule application to `apply_bundle_rules()`.

Can be used standalone via CLI:
```bash
./venv/bin/python -m training.src.scripts.serve_bundle \
  --kiosk-id abc123 --anchor-product-id 000056-002
```

### serve_bundle_api.py — FastAPI service

**File:** `training/src/scripts/serve_bundle_api.py` (~155 lines)

**Startup process:**
1. Read config from `BUNDLE_CONFIG` env var (default: `training/configs/serve_bundle.yaml`)
2. Load 4 parquet files into memory
3. Load products CSV (for name/category enrichment)
4. Build dict-index: `{(kiosk_id, anchor_product_id): DataFrame}` — pre-groups all 26.9M prediction rows for O(1) lookup
5. Build product name lookup dict

**Startup time:** ~60–90 seconds (dominated by parquet loading and dict construction)

**Memory usage:** ~2 GB (256 MB parquet + index overhead)

**Request processing:**
1. O(1) dict lookup → pre-filtered DataFrame for this (kiosk, anchor)
2. `build_bundle()` applies business rules + fallback
3. Enrich items with product names
4. Return JSON with `n_items`, `latency_ms`, and item list

**Measured latency:** ~2 ms per request (after startup)

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Returns `{"status": "ok"}` when assets are loaded. Use for k8s readiness probes. |
| `GET /bundle` | Main recommendation endpoint. All business rules are query parameters. |
| `GET /docs` | Auto-generated Swagger UI (FastAPI built-in) |

### Starting the server

```bash
BUNDLE_CONFIG=training/configs/serve_bundle.yaml \
uvicorn training.src.scripts.serve_bundle_api:app \
  --host 0.0.0.0 --port 8000

# With multiple workers (production)
BUNDLE_CONFIG=training/configs/serve_bundle.yaml \
uvicorn training.src.scripts.serve_bundle_api:app \
  --host 0.0.0.0 --port 8000 --workers 4
```

Note: Each worker loads its own copy of the data (~2 GB each). With 4 workers, expect ~8 GB RAM.

---

## 6. Features

**File:** `training/src/steps/add_features.py`
**Orchestrator:** `training/src/features.py`

The model uses 8 features:

| # | Feature | Type | Description | Source |
|---|---------|------|-------------|--------|
| 1 | `cooc_cosine_sim` | float | Cosine similarity between anchor and candidate from MBA co-occurrence matrix | MBA candidates table (already present) |
| 2 | `pop_store` | int | Number of times this specific kiosk ordered the candidate product | Join orders × candidates on (kiosk, product) |
| 3 | `pop_global` | int | Total orders for the candidate product across all kiosks | Global aggregation on product_id |
| 4 | `kiosk_product_cnt` | int | Total number of order rows for this kiosk (proxy for kiosk size/activity) | Aggregation on kiosk_id |
| 5 | `cand_is_new` | binary (0/1) | 1 if the kiosk has never ordered the candidate product before | pop_store == 0 |
| 6 | `same_category` | binary (0/1) | 1 if anchor and candidate share the same product category | Product catalog join |
| 7 | `channel` | categorical (hashed) | Kiosk sales channel (e.g., retail, wholesale) | Commerces table join |
| 8 | `region` | categorical (hashed) | Kiosk geographic region | Commerces table join |

### Feature processing

- Categorical features (`channel`, `region`) are hash-encoded: `hash(string) → UInt64 → Float64`
- All numeric features are cast to `Float64` with nulls filled as 0
- Feature columns are aligned to the model's `features.json` list before prediction

### Feature importance

The model relies most heavily on:
1. `cooc_cosine_sim` — MBA signal (strongest)
2. `pop_store` — kiosk-level personalization
3. `pop_global` — global popularity prior
4. `kiosk_product_cnt` — kiosk activity level

---

## 7. Business Rules

**Function:** `apply_bundle_rules()` in `serve_bundle.py`

Business rules are applied **after** scoring and **before** returning results. They are specified per-request via API query parameters or CLI flags.

| Rule | Parameter | Behavior |
|------|-----------|----------|
| **Exclude products** | `excluded_products` | Remove specific product IDs from recommendations |
| **Category filter** | `allowed_categories` | Only keep products from specified categories |
| **Category diversity** | `n_group_key` | Max N items per category (enforced across all fallback levels) |
| **Force include** | `included_products` | Inject products with highest score (appear first). If product isn't in candidates, it's added. |
| **Bundle size min** | `n_min` | Log a warning if fewer items than this (but still return what's available) |
| **Bundle size max** | `n_max` | Hard cap on number of items returned |

### Processing order

1. Exclude products → 2. Filter categories → 3. Sort by score → 4. Enforce n_group_key → 5. Force-include products → 6. Clip to n_max

### Priority when rules conflict

- `included_products` **wins** over `excluded_products` — if a product appears in both lists, it will be included
- `n_group_key` is enforced **after** all fallback levels fill — so fallback items also respect category diversity
- `allowed_categories` is **strictly enforced** in fallback — fallback items from forbidden categories are excluded

---

## 8. Fallback System

The system guarantees **non-empty recommendations** for any input through a 4-level fallback chain.

### Fallback levels

```
Level 1: LightGBM Predictions (personalized)
    ├── Lookup (kiosk_id, anchor_product_id) in predictions.parquet
    ├── Returns scored candidates specific to this kiosk
    └── If found: typically 20 items, scores in range [-8, +4]

Level 2: Per-Anchor MBA Co-Purchase
    ├── Lookup anchor_product_id in popularity_fallback.parquet
    ├── Returns co-purchased products regardless of kiosk
    ├── Scores normalized to model range
    └── Used when: kiosk is unknown but anchor is known

Level 3: Per-Category Popularity
    ├── Look up anchor's category from product catalog
    ├── Fetch popular products in that category from category_fallback.parquet
    ├── Scores below Level 2
    └── Used when: anchor is unknown but its category exists

Level 4: Global Popularity
    ├── Top-20 most purchased products from global_fallback.parquet
    ├── Scores below Level 3
    └── Used when: everything is unknown (last resort)
```

### Score normalization

All fallback levels have their scores normalized to the model's actual score range to maintain sensible relative ordering when mixed:
- Level 2 scores are in `[model_min, model_max]`
- Level 3 scores start below Level 2 minimum
- Level 4 scores start below Level 3 minimum

### Coverage analysis

| Scenario | Fallback level | Coverage |
|----------|---------------|----------|
| Known kiosk + known anchor | Level 1 | ~83.6% of requests |
| Unknown kiosk + known anchor | Level 2 | ~99% of anchors covered |
| Unknown kiosk + unknown anchor | Level 3-4 | 100% (global fallback) |

---

## 9. File Reference

### Source code

| File | Lines | Purpose |
|------|-------|---------|
| `src/pipelines/training.py` | ~872 | End-to-end training pipeline orchestration |
| `src/scripts/run_training_pipeline.py` | ~20 | CLI wrapper for training pipeline |
| `src/scripts/generate_predictions.py` | ~310 | Batch scoring and fallback generation |
| `src/scripts/serve_bundle.py` | ~411 | Bundle building with business rules and 4-level fallback |
| `src/scripts/serve_bundle_api.py` | ~155 | FastAPI service with dict-index for O(1) lookup |
| `src/scripts/test_serve_bundle.py` | ~510 | Comprehensive smoke test (scenarios A–G) |
| `src/scripts/check_personalization.py` | — | Analysis: how different are recommendations across kiosks |
| `src/scripts/check_new_vs_repeat.py` | — | Analysis: new vs. repeat product recommendations |
| `src/features.py` | — | Feature orchestrator, hash encoding, column alignment |
| `src/config.py` | — | YAML config loader |
| `src/paths.py` | — | Path constants (RAW_DIR, EXTERNAL_DIR, INTERIM_DIR, etc.) |
| `src/logging_utils.py` | — | Logging setup with file + console handlers |

### Pipeline steps (`src/steps/`)

| File | Function | Purpose |
|------|----------|---------|
| `preprocessing.py` | `preprocess_orders()` | Clean raw CSVs: validate columns, cast types, deduplicate, normalize |
| `split_orders.py` | `split_orders_by_time()` | Time-sorted train/val/test split |
| `build_baskets.py` | `build_baskets()` | Group orders → (kiosk, date) baskets with product lists |
| `generate_candidates.py` | `generate_candidates()` | MBA co-occurrence: cooc_count, support, lift, cosine_sim |
| `select_top_k_candidates.py` | `select_top_k_candidates()` | Keep top-K candidates per anchor by lift/cosine |
| `build_feature_table.py` | `build_feature_table()` | Cross-join kiosks × anchors × candidates |
| `add_features.py` | `add_features()` | Compute all 8 features |
| `build_labels.py` | `build_labels()` | Binary labels from co-purchase pairs |
| `rank_eval_at_k.py` | `rank_eval_at_k()` | HitRate, NDCG, Recall, MRR, Precision @K |

### I/O helpers (`src/io/`)

| Function | Purpose |
|----------|---------|
| `load_orders_csv_sample()` | Load N rows from raw CSV(s) |
| `load_orders_parquet()` | Load preprocessed orders |
| `load_products_csv()` | Load product catalog |
| `load_commerces_csv()` | Load kiosk metadata |
| `load_parquet()` | Generic parquet loader |
| `save_parquet()` | Save DataFrame to parquet |

### Tests (`tests/`)

| File | What it tests |
|------|--------------|
| `test_preprocessing.py` | Order cleaning, type casting, deduplication |
| `test_build_baskets.py` | Basket construction from orders |
| `test_generate_candidates.py` | MBA co-occurrence computation |
| `test_select_top_k.py` | Top-K candidate selection |
| `test_build_feature_table.py` | Feature table cross-join |
| `test_features_orchestrator.py` | Feature computation pipeline |
| `test_build_labels.py` | Label generation from co-purchase |
| `test_pipeline_smoke.py` | End-to-end pipeline with tiny data |
| `conftest.py` | Shared fixtures |

---

## 10. Configuration Reference

### training_pipeline.yaml

```yaml
# Data
raw_paths:                                    # List of raw CSV file paths
  - training/data/raw/2024-20250000_part_00-003.csv
n_rows: 4000000                               # Number of rows to sample
sample_position: tail                         # Sample from end (most recent)
interim_path: training/data/interim/orders_sample.parquet
products_path: training/data/external/products_v2.csv
commerces_path: training/data/external/commerces.csv
model_path: training/models/lgbm_ranker.txt

# Split
train_ratio: 0.8                              # 80% for training
val_ratio: 0.1                                # 10% for validation
test_ratio: 0.1                               # 10% for testing
train_label_ratio: 0.3                        # 30% of train for labels (leakage prevention)

# MBA Candidates
min_cooc: 2                                   # Minimum co-occurrence count
min_lift: 1.2                                 # Minimum lift threshold
top_k: 100                                    # Candidates per anchor at training
top_k_train: 100                              # Same as top_k (used in some code paths)

# Labels
label_window_days: 7                          # Co-purchase window for positive labels
min_cooc_label: 1                             # Minimum co-occurrence for label
label_kiosk_batch_size: 0                     # 0 = no batching

# Training
max_neg_per_group: 20                         # Max negatives per query group
max_eval_queries: 50000                       # Max queries for eval speed
eval_ks: [5, 10, 20, 50]                      # @K values for metrics
predict_batch_size: 200000                    # Batch size for prediction
lgbm_params:                                  # LightGBM hyperparameters
  learning_rate: 0.02
  num_leaves: 15
  max_depth: 5
  min_data_in_leaf: 1000
  min_gain_to_split: 0.5
  lambda_l1: 2.0
  lambda_l2: 10.0
  feature_fraction: 0.55
  bagging_fraction: 0.6
  bagging_freq: 1
  seed: 42
num_boost_round: 2000                         # Max boosting iterations
early_stopping_rounds: 100                    # Early stopping patience
eval_log_path: logs/training_eval_curve.csv   # Training curve log
```

### generate_predictions.yaml

```yaml
# Paths
orders_path: training/data/interim/orders_sample.parquet
products_path: training/data/external/products_v2.csv
commerces_path: training/data/external/commerces.csv
model_path: training/models/lgbm_ranker.txt
predictions_path: training/data/interim/predictions.parquet
popularity_path: training/data/interim/popularity_fallback.parquet
category_fallback_path: training/data/interim/category_fallback.parquet
global_fallback_path: training/data/interim/global_fallback.parquet

# Inference
inference_last_n_days: 90                     # Use orders from last N days
inference_max_rows: 0                         # 0 = no limit
min_cooc: 2                                   # MBA filter
min_lift: 1.2                                 # MBA filter
top_k_candidates: 50                          # MBA candidates per anchor
catalog_top_k: 20                             # Final top-K per (kiosk, anchor)
predict_batch_size: 200000                    # Memory safety
query_sample_n: 0                             # 0 = no sampling (all queries)
```

### serve_bundle.yaml

```yaml
# Default query (for CLI testing)
kiosk_id: "fe7ef5cd7c273ec75600d6c710216f69"
anchor_product_id: "000056-002"
included_products: ""
excluded_products: "004747-001"
allowed_categories: ""
agg_key: ""
n_group_key: 2                                # Max 2 items per category
n_min: 4                                      # Warn if < 4 items
n_max: 10                                     # Return max 10 items

# Paths
predictions_path: training/data/interim/predictions.parquet
popularity_path: training/data/interim/popularity_fallback.parquet
category_fallback_path: training/data/interim/category_fallback.parquet
global_fallback_path: training/data/interim/global_fallback.parquet
products_path: training/data/external/products_v2.csv
```

### features.yaml

```yaml
# Legacy feature flags — most features are now always computed in add_features.py
include_product_features: false
include_kiosk_features: true
include_behavioral_features: true
include_personalization_features: false
include_popularity_features: false
encode_channel: false
encode_region: false
```

---

## 11. Testing

### Unit tests

```bash
./venv/bin/python -m pytest training/tests/ -q
```

18 tests covering all pipeline steps: preprocessing, baskets, candidates, features, labels, top-K selection, and a full pipeline smoke test with synthetic data.

### Smoke test (serve_bundle)

```bash
./venv/bin/python -m training.src.scripts.test_serve_bundle
```

Requires generated parquet files. Tests 7 scenario groups:

| Group | Tests | What it verifies |
|-------|-------|-----------------|
| A | 1 | Known kiosk + known anchor → gets personalized results |
| B | 1 | Unknown kiosk + known anchor → falls back to per-anchor MBA |
| C | 1 | Known kiosk + unknown anchor → falls back to category/global |
| D | 1 | Unknown kiosk + unknown anchor → falls back to global only |
| E | 250 | Random queries: 0 empty bundles, 100% full, p50 ~18ms, p95 ~21ms |
| F | 100 | Relevance: ~19% item overlap, ~82% category overlap with kiosk history |
| G | 12 | Business rules: exclusions, inclusions, category filters, n_group_key, n_max, combined rules, edge cases |

### Quality checks

```bash
# How personalized are recommendations across kiosks?
./venv/bin/python -m training.src.scripts.check_personalization \
  --config training/configs/training_pipeline.yaml --top-k 5 --sample-kiosks 300

# What fraction of recommendations are new vs. repeat products?
./venv/bin/python -m training.src.scripts.check_new_vs_repeat --top-k 5 --sample-kiosks 200
```

---

## 12. Deployment

### Production deployment steps

```
1. Train model (run once or monthly):
   ./venv/bin/python -m training.src.scripts.run_training_pipeline \
     --config training/configs/training_pipeline.yaml

2. Generate predictions (run daily/weekly):
   ./venv/bin/python -m training.src.scripts.generate_predictions \
     --config training/configs/generate_predictions.yaml

3. Start API server:
   BUNDLE_CONFIG=training/configs/serve_bundle.yaml \
   uvicorn training.src.scripts.serve_bundle_api:app \
     --host 0.0.0.0 --port 8000

4. Verify:
   curl http://localhost:8000/health
   # {"status": "ok"}
```

### Refresh cycle

| What | Frequency | Time | Impact |
|------|-----------|------|--------|
| Retrain model | Monthly | ~30 min | New model file. Requires rerunning step 2. |
| Regenerate predictions | Daily/Weekly | ~16 min | New parquet files. Requires server restart or hot-reload. |
| Server restart | After new parquets | ~90 sec | Downtime during loading. Use blue-green deployment. |

### Hardware requirements

| Stage | CPU | RAM | Disk |
|-------|-----|-----|------|
| Training | 4+ cores | 4 GB | 2 GB |
| Batch scoring | 4+ cores | 8 GB | 1 GB |
| Serving (per worker) | 1 core | 2 GB | 300 MB |

### Environment variable

| Variable | Default | Purpose |
|----------|---------|---------|
| `BUNDLE_CONFIG` | `training/configs/serve_bundle.yaml` | Path to serving config (used by API) |

---

## 13. Troubleshooting

### Common issues

**"Missing features" warnings during inference**
- Compare `training/models/lgbm_ranker.features.json` with columns produced by feature computation
- Missing features are filled with zeros, which degrades predictions
- Fix: retrain the model or ensure same feature flags in training and inference

**"Zero-only features" warnings**
- Indicates a feature is computed but always zero
- Usually caused by mismatch between training and inference data
- Check that `channel`/`region` encoding is consistent

**Empty predictions for a kiosk**
- Kiosk may not be in `commerces.csv` (not "active")
- Or kiosk has no orders in the 90-day window
- The 4-level fallback still provides recommendations even without predictions

**Server startup takes too long**
- Loading 256 MB parquet + building dict-index takes ~60–90 seconds
- This is a one-time cost; subsequent requests are ~2 ms
- For production, use health check endpoint to gate traffic

**Candidate alignment between training and inference**
- `top_k` (training, config) = 100, `top_k_candidates` (inference) = 50
- Lower inference value is intentional (keeps catalog smaller)
- If needed, align them to get identical candidate pools

### Logs location

All logs are written to `logs/` with timestamps:
```
logs/
├── training_20260301_120000.log
├── generate_predictions_20260301_130000.log
├── serve_bundle_20260301_140000.log
└── test_serve_bundle_20260301_150000.log
```

### Useful debugging commands

```bash
# Check model feature list
cat training/models/lgbm_ranker.features.json

# Check predictions parquet schema
./venv/bin/python -c "import polars as pl; df=pl.read_parquet('training/data/interim/predictions.parquet'); print(df.schema); print(f'Rows: {df.height}, Unique kiosks: {df[\"kiosk_id\"].n_unique()}, Anchors: {df[\"anchor_product_id\"].n_unique()}')"

# Check fallback sizes
./venv/bin/python -c "
import polars as pl
for f in ['predictions','popularity_fallback','category_fallback','global_fallback']:
    p = f'training/data/interim/{f}.parquet'
    try:
        df = pl.read_parquet(p)
        print(f'{f}: {df.height} rows, {df.estimated_size(\"mb\"):.1f} MB')
    except: print(f'{f}: not found')
"

# Quick API test
curl -s "http://localhost:8000/bundle?kiosk_id=TEST&anchor_product_id=TEST" | python3 -m json.tool
```