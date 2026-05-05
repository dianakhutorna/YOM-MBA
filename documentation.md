# YOM Bundle Recommender — Technical Documentation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Flow](#2-data-flow)
3. [Training Pipeline](#3-training-pipeline)
4. [Batch Scoring (Inference)](#4-batch-scoring-inference)
5. [Serving Layer](#5-serving-layer)
6. [Features](#6-features)
7. [Fallback System](#7-fallback-system)
8. [File Reference](#8-file-reference)
9. [Configuration Reference](#9-configuration-reference)
10. [Testing & Analysis](#10-testing--analysis)
11. [Deployment](#11-deployment)
12. [Troubleshooting](#12-troubleshooting)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. System Overview

### What the system does

Given a **kiosk** (point-of-sale) and an **anchor product** (the product a customer is currently viewing), the system returns a ranked list of **candidate products** to recommend as a bundle. For example: customer picks "Leche Frutil" → system recommends "Leche Blanca", "Nectar UHT", "Leche Chocolate", etc.

### Three-stage architecture

The system is split into three independent stages that run at different frequencies:

```
┌────────────────────────┐
│  STAGE 1: TRAINING     │  Frequency: monthly or on-demand
│  training.py           │
│                        │
│  Input:  raw order CSVs, product catalog, kiosk metadata
│  Output: lgbm_ranker.txt (model) + lgbm_ranker.features.json
│  Time:   ~30 minutes
└────────────┬───────────┘
             │ model file
             ▼
┌────────────────────────┐
│  STAGE 2: SCORING      │  Frequency: daily or weekly
│  generate_predictions  │
│                        │
│  Input:  model + recent orders (90-day window)
│  Output: 4 parquet files (predictions + 3 fallbacks)
│  Time:   ~16 minutes
└────────────┬───────────┘
             │ parquet files (total ~300 MB)
             ▼
┌────────────────────────┐
│  STAGE 3: SERVING      │  Frequency: 24/7 (AWS Lambda)
│  serve_recommendations │
│  _api + lambda_handler │
│                        │
│  Input:  4 parquet files (loaded at startup)
│  Output: JSON recommendations per request (~2 ms)
│  Startup: ~90 seconds
└────────────────────────┘
```

### Why this architecture?

- **No online inference** — LightGBM is never called at request time. All scoring done in batch Stage 2. Serving is dictionary lookup.
- **Decoupled stages** — Training runs on powerful machine; batch scoring on compute servers; Lambda runs on minimal hardware (~2 GB).
- **Never-empty results** — Four-level fallback guarantees recommendations for any (kiosk, anchor) pair, even completely unknown.
- **Memory safe** — Chunked CSV loading (Stage 1), batched candidate generation (Stage 1), strategic garbage collection throughout.

### Technology stack

| Component | Technology |
|-----------|-----------|
| Data processing | Polars 0.20+ (columnar DataFrame, memory-efficient) |
| Model | LightGBM LambdaRank (learning-to-rank, 869 iterations) |
| Candidate generation | Market Basket Analysis (MBA) — co-occurrence counts, lift, cosine similarity |
| Serving | FastAPI (ASGI) + Mangum (Lambda adapter) |
| Deployment | AWS Lambda + Docker (ECR) |
| Storage | Parquet files (no database) |
| Config | YAML files |
| Testing | pytest + analysis scripts |
| CI/CD | GitHub Actions (auto-deploy on push)

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

### Production: AWS Lambda

**Architecture:**
```
Request → FastAPI (ASGI) → Mangum (Lambda adapter) → handler event
         ↓
    Dict lookup (kiosk, anchor)
         ↓
    4-level fallback
         ↓
    Apply business rules (exclusions, category filters, etc.)
         ↓
    JSON response
```

**Files:**
- `training/src/scripts/serve_recommendations_api.py` — FastAPI application (REST API definition)
- `training/src/scripts/lambda_handler.py` — AWS Lambda entry point (Mangum adapter)
- `training/src/services/recommendation_service.py` — Core lookup + fallback logic

**Startup Process:**
1. Lambda initializes with `lambda_handler.handler`
2. Mangum converts Lambda event to ASGI request
3. FastAPI app loads (first-time startup):
   - Load 4 parquet files (256 MB total) from `training/data/interim/`
   - Build dict-index: `{(kiosk_id, anchor_product_id): [recs]}` for O(1) lookup
   - Load product lookup dict for fallback mapping
4. Request routed to endpoint handler
5. Return JSON response

**Startup Time:** ~90 seconds (dominated by parquet loading)

**Memory Usage:** ~2 GB per Lambda instance (parquets + dict-index)

**Request Latency:** ~2 ms (after startup)

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Readiness check: `{"status": "ok"}` |
| `/recommendations` | GET | Get recommendations for (kiosk, anchor) |
| `/recommendations/multi` | POST | Batch recommendations for multiple queries |
| `/docs` | GET | Swagger UI (FastAPI auto-generated) |

**Example request:**
```
GET /recommendations?kioskId=fe7ef5cd7c27&anchorId=000056-002&limit=30
```

**Example response:**
```json
[
  {
    "anchor_id": "000056-002",
    "kiosk_id": "fe7ef5cd7c27...",
    "product_id": "002030-001",
    "model_id": "lgbm_ranker",
    "recommendation_date": "2026-05-05T10:30:45Z"
  },
  ...
]
```

### Local Development

For local testing during development (NOT production, use Lambda for production):

```bash
# Install full dependencies
pip install -r requirements.txt

# Start server
./venv/bin/python -m training.src.scripts.serve_recommendations_api

# Access API
open http://localhost:8000/docs
```

This loads parquets from local `training/data/interim/` directory (~2 GB RAM).

### 4-Level Fallback

The serving layer guarantees **non-empty recommendations** through intelligent fallback:

| Level | Source | Score Range | When used |
|-------|--------|------------|-----------|
| 1 | LightGBM predictions | [-8, +4] | Known kiosk + known anchor (~83% of queries) |
| 2 | Per-anchor MBA | [scaled] | Unknown kiosk, known anchor (~99% with fallback) |
| 3 | Per-category popularity | [scaled] | Unknown anchor (uses anchor's category) |
| 4 | Global popularity | [scaled] | Everything unknown (last resort, always succeeds) |

All fallback scores are normalized to maintain consistent ranking when mixed with Level 1 predictions.

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

## 7. Fallback System

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

## 8. File Reference

### Source code structure

#### Core pipeline files

| File | Lines | Purpose |
|------|-------|---------|
| `src/pipelines/training.py` | ~931 | End-to-end training: preprocessing → split → baskets → candidates → features → train → eval |
| `src/pipelines/training_experiment.py` | ~684 | Variant: train/test already pre-split in separate CSV files |
| `src/scripts/run_training_pipeline.py` | ~20 | CLI wrapper for training.py |
| `src/scripts/generate_predictions.py` | ~410 | Batch inference: score predictions + generate 4 fallback parquets |

#### Serving files

| File | Purpose |
|------|---------|
| `src/scripts/serve_recommendations_api.py` | FastAPI application (REST endpoints: /health, /recommendations, /recommendations/multi) |
| `src/scripts/lambda_handler.py` | AWS Lambda entry point (Mangum ASGI adapter) |
| `src/services/recommendation_service.py` | Core service: load parquets, build dict-index, O(1) lookup, 4-level fallback |

#### Analysis tools

| File | Purpose |
|------|---------|
| `src/scripts/check_personalization.py` | Analyze kiosk diversity in recommendations |
| `src/scripts/check_new_vs_repeat.py` | Analyze new vs. repeat product coverage |

#### Pipeline step modules (src/steps/)

| Module | Function | Purpose |
|--------|----------|---------|
| `preprocessing.py` | `preprocess_orders()` | Clean CSVs: validate columns, deduplicate, normalize |
| `split_orders.py` | `split_orders_by_time()` | Time-based train/val/test split (80% / 10% / 10%) |
| `build_baskets.py` | `build_baskets()` | Group orders into (kiosk, date) baskets |
| `generate_candidates.py` | `generate_candidates()` | MBA co-occurrence: cooc_count, lift, cosine_sim (batched generation for memory safety) |
| `select_top_k_candidates.py` | `select_top_k_candidates()` | Filter to top-K candidates per anchor |
| `build_feature_table.py` | `build_feature_table()` | Cross-join: kiosks × anchors × candidates |
| `add_features.py` | `add_features()` | Compute 8 features: cooc_cosine_sim, pop_store, pop_global, etc. |
| `build_labels.py` | `build_labels()` | Generate binary labels from co-purchase pairs |
| `rank_eval_at_k.py` | `rank_eval_at_k()` | Compute HitRate, NDCG, Recall, MRR, Precision @K |

#### I/O and utilities (src/io/, src/)

| File | Purpose |
|------|---------|
| `io/loaders.py` | Load orders, products, commerces; includes `load_orders_csv_chunked()` for memory-safe batch loading |
| `io/__init__.py` | Export all loader functions |
| `features.py` | Feature orchestrator, hash encoding, column alignment |
| `config.py` | YAML config loader |
| `paths.py` | Path constants (RAW_DIR, EXTERNAL_DIR, INTERIM_DIR, MODELS_DIR, LOGS_DIR) |
| `logging_utils.py` | Logging setup (file + console, timestamp-based log filenames) |

#### Tests (tests/)

| File | Purpose |
|------|---------|
| `test_preprocessing.py` | Unit tests: order cleaning, deduplication |
| `test_build_baskets.py` | Unit tests: basket construction |
| `test_generate_candidates.py` | Unit tests: MBA pair generation |
| `test_build_feature_table.py` | Unit tests: feature table cross-join |
| `test_select_top_k.py` | Unit tests: top-K selection |
| `test_features_orchestrator.py` | Unit tests: feature computation |
| `test_build_labels.py` | Unit tests: label generation |
| `test_pipeline_smoke.py` | Integration test: full pipeline on toy data |
| `conftest.py` | Shared pytest fixtures |

---

## 9. Configuration Reference

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
inference_last_n_days: 90                     # Use orders from last 90 days (83.6% coverage)
inference_max_rows: 0                         # 0 = no limit
min_cooc: 2                                   # MBA filter: minimum co-occurrence
min_lift: 1.2                                 # MBA filter: minimum lift
top_k_candidates: 50                          # MBA candidates per anchor (note: < training top_k=100)
catalog_top_k: 20                             # Final top-K stored per (kiosk, anchor)
predict_batch_size: 200000                    # Memory-safe batch size for prediction
query_sample_n: 0                             # 0 = use all queries (no sampling)
```

### features.yaml

```yaml
# Legacy feature flags — all features always computed in add_features.py
# These flags are kept for backward compatibility but have minimal effect
include_product_features: false
include_kiosk_features: true
include_behavioral_features: true
include_personalization_features: false
include_popularity_features: false
encode_channel: false
encode_region: false
```

## 10. Testing & Analysis

### Unit tests

```bash
./venv/bin/python -m pytest training/tests/ -q
```

18 tests covering all pipeline steps: preprocessing, baskets, candidates, features, labels, top-K selection, and a full end-to-end smoke test on tiny synthetic data.

| Test file | Coverage |
|-----------|----------|
| `test_preprocessing.py` | CSV cleaning, type casting, deduplication |
| `test_build_baskets.py` | Basket construction from orders |
| `test_generate_candidates.py` | MBA co-occurrence computation and batched generation |
| `test_build_feature_table.py` | Cross-join and feature alignment |
| `test_select_top_k.py` | Top-K candidate filtering |
| `test_features_orchestrator.py` | Feature computation (8 features) |
| `test_build_labels.py` | Label generation from co-purchase pairs |
| `test_pipeline_smoke.py` | **Integration test:** full pipeline on 100 rows of data |

### Quality analysis tools

```bash
# Check how personalized recommendations are across kiosks
./venv/bin/python -m training.src.scripts.check_personalization \
  --config training/configs/training_pipeline.yaml \
  --top-k 5 \
  --sample-kiosks 300

# Output: Unique recommendation sets, coverage metrics, Jaccard similarity, per-kiosk examples
```

```bash
# Analyze new vs. repeat product coverage
./venv/bin/python -m training.src.scripts.check_new_vs_repeat \
  --top-k 5 \
  --sample-kiosks 200

# Output: Fraction of new products in recommendations, overlap with repeat products
```

### Local API testing (development only)

```bash
# Start FastAPI server locally (NOT used in production)
./venv/bin/python -m training.src.scripts.serve_recommendations_api

# Test endpoints
curl http://localhost:8000/health
# {"status": "ok"}

curl "http://localhost:8000/recommendations?kioskId=TEST&anchorId=TEST&limit=10"

# View auto-generated API docs
open http://localhost:8000/docs
```

---

## 11. Deployment

### Architecture overview

```
GitHub          ECR              Lambda          CloudFront
  ↓              ↓                  ↓                ↓
Push to main ← Docker push ←  Auto-update  ← Edge cache
   |                            Instance
   └── Triggers .github/workflows/deploy.yml
```

### Deployment workflow

The system uses a **3-stage deployment architecture** with automatic CI/CD:

```
Stage 1: Training (as needed, ~30 min)
│
├─ Run ./venv/bin/python -m training.src.scripts.run_training_pipeline
├─ Output: training/models/lgbm_ranker.txt + .features.json
└─ Stores model in repository

Stage 2: Batch Scoring (daily/weekly, ~16 min)
│
├─ Run ./venv/bin/python -m training.src.scripts.generate_predictions
├─ Output: 4 parquet files (predictions, popularity, category, global)
└─ Stores predictions in training/data/interim/

Stage 3: Deploy to Lambda (automatic on git push)
│
├─ GitHub Actions triggers .github/workflows/deploy.yml
├─ Build Docker image (Linux AMD64 for Lambda)
├─ Push to ECR (Amazon Elastic Container Registry)
├─ Update Lambda function with new image
└─ Lambda auto-restarts (~90 sec downtime, background)
```

### AWS Infrastructure

| Resource | Details |
|----------|---------|
| **AWS Account** | 124661688886 (diana production) |
| **Region** | eu-central-1 |
| **ECR Repository** | diana-backend |
| **Lambda Function** | diana-backend |
| **Lambda RAM** | 2 GB |
| **Lambda Timeout** | 30 seconds |
| **Lambda Role** | Uses OpenID Connect (OIDC) for secure GitHub Actions auth |
| **Entry Point** | `training.src.scripts.lambda_handler.handler` |
| **Startup Time** | ~90 sec (parquet loading + index building) |
| **Request Latency** | ~2 ms (per recommendation) |

### GitHub Actions Workflow

**Location:** `.github/workflows/deploy.yml`

**Trigger:** Push to `main` or `add-backend` branch

**Steps:**
1. Checkout code
2. Build Docker image (multistage build to minimize size)
3. Authenticate with AWS via OIDC (no hardcoded credentials)
4. Push image to ECR
5. Update Lambda function URI
6. Lambda container re-initializes with new code

### Docker Image

```dockerfile
FROM public.ecr.aws/lambda/python:3.12

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY training /var/task/training

CMD ["training.src.scripts.lambda_handler.handler"]
```

**Image size:** ~500 MB (compressed)
**Runtime:** ~300 MB (uncompressed in Lambda)

### Requirements files

**requirements-backend.txt** (used by Lambda):
- **Purpose:** Minimal dependencies for runtime (no dev tools)
- **Packages:** FastAPI, uvicorn, Mangum, Polars, boto3, pydantic, pyyaml
- Used to keep Lambda startup time and image size small

**requirements.txt** (used for local development):
- **Purpose:** Full development environment (training, testing, analysis)
- **Packages:** fastapi, uvicorn, mangum, polars, lightgbm, boto3, pytest, numpy, pydantic, pyyaml

Both files have been cleaned of unused dependencies.

### Hardware specs & timing

| Component | CPU | RAM | Disk | Duration |
|-----------|-----|-----|------|----------|
| Training | 4+ cores | 4 GB | 2 GB | ~30 min |
| Batch Scoring | 4+ cores | 8 GB | 1 GB | ~16 min |
| Lambda (per request) | 0.5 vCPU* | 2 GB | 512 MB | ~2 ms |

*Lambda allocates CPU proportional to RAM. 2 GB RAM = ~0.5 vCPU @ 5% utilization.

### Monitoring & Refresh cycle

| What | When | Impact | Downtime |
|------|------|--------|----------|
| Model retraining | Monthly (on-demand) | New model file. Triggers batch scoring. | None (async) |
| Batch scoring | Daily/Weekly | New parquets pushed to Git. Auto-deploys to Lambda. | ~90 sec (Lambda restart) |
| Code update | After git push | New image in ECR. Lambda auto-updates. | ~2 min (container handoff) |

### Troubleshooting deployment

**Lambda fails to start:**
- Check CloudWatch logs: AWS Console > Lambda > diana-backend > Logs
- Common causes: missing parquet files, corrupted index, OOM (increase RAM)

**Predictions outdated:**
- Verify latest parquets in `training/data/interim/` match local copy
- Re-run batch scoring and git push to re-deploy

**Slow requests:**
- First request after Lambda restart is slow (~90 sec) due to cold start
- Subsequent requests: ~2 ms
- Monitor with CloudWatch Metrics

---

## 12. Troubleshooting

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
- Or kiosk has no orders in the 90-day historical window
- The 4-level fallback system still provides recommendations even without personalized predictions

**Server startup takes too long**
- Lambda cold start (first request after deploy): ~90 seconds
- This is normal and expected; subsequent requests are ~2 ms
- Use health check endpoint to gate traffic during startup
- See Section 11: Deployment for Lambda performance details

**Candidate alignment between training and inference**
- `top_k` (training, config) = 100, `top_k_candidates` (inference) = 50
- Lower inference value is intentional (keeps catalog smaller)
- If needed, align them to get identical candidate pools

### Logs location

All logs are written to `logs/` directory with timestamps:
```
logs/
├── training_20260301_120000.log      (full training pipeline)
├── generate_predictions_20260301_130000.log  (batch inference)
└── experiment_1m_eval_curve.csv     (evaluation metrics)
```

### Debugging: local development

```bash
# Start FastAPI server locally for testing (NOT production)
./venv/bin/python -m training.src.scripts.serve_recommendations_api

# Test endpoints
curl http://localhost:8000/health
# {"status": "ok"}

curl "http://localhost:8000/recommendations?kioskId=TEST&anchorId=TEST&limit=10"

# View auto-generated API docs
open http://localhost:8000/docs
```

### Useful debugging commands

```bash
# Check model feature list
cat training/models/lgbm_ranker.features.json | python3 -m json.tool | head -20

# Check predictions parquet schema and size
./venv/bin/python3 << 'EOF'
import polars as pl
df = pl.read_parquet('training/data/interim/predictions.parquet')
print(f"Schema: {df.schema}")
print(f"Rows: {df.height}")
print(f"Unique kiosks: {df['kiosk_id'].n_unique()}")
print(f"Unique anchors: {df['anchor_product_id'].n_unique()}")
EOF

# Check all fallback files
./venv/bin/python3 << 'EOF'
import polars as pl
import os
for fname in ['predictions', 'popularity_fallback', 'category_fallback', 'global_fallback']:
    path = f'training/data/interim/{fname}.parquet'
    if os.path.exists(path):
        df = pl.read_parquet(path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f'{fname}: {df.height} rows, {size_mb:.1f} MB')
    else:
        print(f'{fname}: not found')
EOF

# Quick test of the recommendation logic
./venv/bin/python3 << 'EOF'
from training.src.services.recommendation_service import RecommendationService
from training.src.config import CONFIG

# Initialize service with current config
service = RecommendationService.from_config(CONFIG)
# Test with arbitrary kiosk/anchor IDs
recs = service.get_recommendations(kiosk_id='TEST', anchor_id='TEST', limit=10)
print(f'Got {len(recs)} recommendations')
EOF
```
```

### Lambda-specific debugging

**View Lambda logs:**
```bash
# Via AWS CLI
aws logs tail /aws/lambda/diana-backend --follow

# Via AWS Console:
# Lambda > diana-backend > Monitor > Logs > View logs in CloudWatch
```

**Check Lambda memory usage:**
```bash
aws lambda get-function-concurrency --function-name diana-backend
```

**Manual Lambda test:**
```bash
aws lambda invoke \
  --function-name diana-backend \
  --payload '{"path":"/health"}' \
  response.json
cat response.json
```

---