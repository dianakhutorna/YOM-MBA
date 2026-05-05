# YOM Bundle Recommender System

Offline ML pipeline that **trains** a ranking model, **generates** bundle predictions in batch, and **serves** them via AWS Lambda with business rules and multi-level fallback.

**Key principle:** No online ML inference — the model scores all (kiosk, anchor, candidate) triples ahead of time in batch. Serving retrieves pre-computed recommendations via dict lookup (~2 ms/request).

---

## Architecture

```
┌──────────────────────────┐     ┌──────────────────────────────┐     ┌─────────────────────────────┐
│   1. TRAINING            │     │ 2. BATCH SCORING             │     │  3. SERVING (AWS Lambda)    │
│   (monthly)              │     │   (daily/weekly)             │     │   (24/7)                    │
│   training.py            │     │ generate_predictions.py      │     │ serve_recommendations_api   │
│                          │     │                              │     │ + lambda_handler.py         │
├──────────────────────────┤     ├──────────────────────────────┤     ├─────────────────────────────┤
│ Raw CSV → preprocess     │     │ Load model + recent orders   │     │ Load 4 parquets at startup  │
│ Time split train/val/test│     │ MBA candidates (90-day)      │     │ Dict-index for O(1) lookup  │
│ MBA candidates + features│ →   │ Feature table → LightGBM     │ →   │ 4-level fallback:           │
│ LightGBM LambdaRank      │     │ Top-20 per (kiosk, anchor)   │     │   1. Model predictions      │
│ Save model + features    │     │ Save 4 parquet artifacts     │     │   2. Per-anchor MBA         │
│ features optimizations:  │     │                              │     │   3. Per-category popular   │
│ - chunked CSV loading    │     │                              │     │   4. Global popular         │
│ - batched pair generation│     │                              │     │ Business rules + JSON       │
│ - garbage collection     │     │                              │     │ Mangum adapter (ASGI→Lambda)
└──────────────────────────┘     └──────────────────────────────┘     └─────────────────────────────┘
```

---

## Project Structure

```
training/
├── configs/
│   ├── training_pipeline.yaml        # Training hyperparameters
│   ├── generate_predictions.yaml     # Batch inference settings
│   └── features.yaml                 # Feature flags (legacy)
├── data/
│   ├── raw/                          # Raw order CSVs
│   ├── external/                     # products_v2.csv, commerces.csv
│   └── interim/                      # Generated parquets
├── models/
│   ├── lgbm_ranker.txt               # Trained LightGBM model
│   └── lgbm_ranker.features.json     # Feature column list
├── src/
│   ├── pipelines/
│   │   └── training.py               # End-to-end training pipeline (8 steps)
│   ├── scripts/
│   │   ├── run_training_pipeline.py          # CLI: run training
│   │   ├── generate_predictions.py           # CLI: batch scoring → 4 parquets
│   │   ├── serve_recommendations_api.py      # FastAPI recommendation service
│   │   ├── lambda_handler.py                 # AWS Lambda entry point
│   │   ├── check_personalization.py          # Personalization analysis
│   │   └── check_new_vs_repeat.py            # New vs repeat item analysis
│   ├── services/
│   │   └── recommendation_service.py         # Core recommendation logic (4-level fallback)
│   ├── steps/                         # Modular pipeline steps
│   ├── io/                            # I/O helpers (load/save, with chunked CSV)
│   ├── features.py                    # Feature orchestrator
│   ├── config.py                      # YAML config loader
│   ├── paths.py                       # Canonical path constants
│   └── logging_utils.py              # Logging setup
├── tests/                             # Unit tests (pytest)
└── logs/                              # Runtime logs
```

---

##Memory:
  - Training: 4 GB
  - Batch scoring: 8 GB  
  - Lambda (per worker): 2 GB (includes 256 MB parquets + dict-index)
- Disk: ~500 MB for models, ~1 GB for predictions parquet

## Dependencies

Two requirements files:
- `requirements.txt` — full data science stack (training, analysis, batch scoring)
- `requirements-backend.txt` — minimal Lambda runtime (FastAPI, Polars, Boto3 only

- Python **3.11+**
- ~500 MB RAM for training, ~1 GB for inference, ~2 GB for API (dict-index)

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Mac / Linux
pip install -r requirements.txt
```

---

## Data

Place input files before running the pipeline:

```
training/data/external/
├── commerces.csv         # Kiosk metadata (channel, region, active flag)
└── products_v2.csv       # Product catalog (productid, name, category)

training/data/raw/
└── *.csv                 # Raw order data (order_id, kiosk_id, product_id, date, qty)
```

---

## 1. Training

```bash
./venv/bin/python -m training.src.scripts.run_training_pipeline \
  --config training/configs/training_pipeline.yaml
```

**Outputs:**
- `training/models/lgbm_ranker.txt` — trained model
- `training/models/lgbm_ranker.features.json` — feature column list
- `training/data/interim/orders_sample.parquet` — preprocessed orders
- `logs/training_*.log` — metrics log

**Key config** (`training_pipeline.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_rows` | 4000000 | Number of order rows to sample |
| `train_ratio` | 0.8 | Time-based train split |
| `train_label_ratio` | 0.3 | Fraction of train for label generation (prevents leakage) |
| `top_k` | 100 | MBA candidates per anchor |
| `label_window_days` | 7 | Co-purchase time window for labels |
| `num_boost_round` | 2000 | Max LightGBM iterations |
| `early_stopping_rounds` | 100 | Early stopping patience |

---

## 2. Batch Scoring (Inference)

```bash
./venv/bin/python -m training.src.scripts.generate_predictions \
  --config training/configs/generate_predictions.yaml
```

**Outputs (4 files):**

| File | Size | Description |
|------|------|-------------|
| `predictions.parquet` | ~256 MB | Scored top-20 per (kiosk, anchor). ~26.9M rows. |
| `popularity_fallback.parquet` | ~63 KB | Per-anchor MBA co-purchase (for unknown kiosks). |
| `category_fallback.parquet` | ~4 KB | Per-category popular products (for unknown anchors). |
| `global_fallback.parquet` | ~2 KB | Top-20 most purchased overall (last resort). |

**Key config** (`generate_predictions.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_last_n_days` | 90 | Time window for recent orders |
| `top_k_candidates` | 50 | MBA candidates per anchor |
| `catalog_top_k` | 20 | Final top-K per (kiosk, anchor) |

---

## 3. Serving

### Production: AWS Lambda

Deployment via Docker + AWS Lambda:

```bash
# Build and push to ECR (auto-triggered by GitHub Actions on push to main/add-backend)
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -t diana-backend:latest \
  --push .

# Deployed via GitHub Actions workflow (.github/workflows/deploy.yml)
# Entry point: training.src.scripts.lambda_handler.handler
# ASGI adapter: Mangum (FastAPI ↔ Lambda event/response)
```

**Startup:** ~90 seconds (parquet loading + dict-index construction)
**Latency:** ~2 ms per request (after startup)
**Memory:** 2 GB per Lambda instance

### Local Testing (Development Only)

For local development, you can test the API with FastAPI's built-in server:

```bash
# This is NOT used in production — only for local testing
pip install -r requirements.txt
./venv/bin/python -m training.src.scripts.serve_recommendations_api
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Readiness: `{"status": "ok"}` |
| `/recommendations` | GET | Get personalized bundle |
| `/docs` | GET | Swagger UI (FastAPI auto-generated) |

**Query parameters** (URL-encoded):

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `anchorId` | string | yes | — | Anchor product ID |
| `kioskId` | string | yes | — | Kiosk identifier |
| `limit` | int | no | 20 | Max items to return (1–100) |

**Example:**
```
GET /recommendations?kioskId=fe7ef5cd7c27&anchorId=000056-002&limit=30
```

**Response:**
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

---

## 4. Tests & Verification

### Unit Tests

```bash
# Run all pipeline tests (18 tests, ~1s)
./venv/bin/python -m pytest training/tests/ -q
```

Covers: preprocessing, baskets, candidates, features, labels, top-K selection, end-to-end pipeline.

### Analysis Tools

```bash
# Analyze how personalized recommendations are across kiosks
./venv/bin/python -m training.src.scripts.check_personalization \
  --config training/configs/training_pipeline.yaml --top-k 5 --sample-kiosks 300

# Check fraction of new vs. repeat product recommendations
./venv/bin/python -m training.src.scripts.check_new_vs_repeat --top-k 5 --sample-kiosks 200
```

### Local API Test

```bash
# Start server locally (development only)
./venv/bin/python -m training.src.scripts.serve_recommendations_api &

# Test health endpoint
curl http://localhost:8000/health

# Test recommendations endpoint
curl "http://localhost:8000/recommendations?kioskId=TEST&anchorId=TEST&limit=10"

# View API docs (Swagger UI)
open http://localhost:8000/docs
```

---

## 5. Configs

| Config | Used by | Purpose |
|--------|---------|---------|
| `training_pipeline.yaml` | `run_training_pipeline.py` | Training hyperparams, data paths, split ratios, LightGBM settings |
| `generate_predictions.yaml` | `generate_predictions.py` | Inference window (90d), candidate settings, output paths |
| `features.yaml` | `features.py` | Feature group flags (legacy, most features always computed) |

---

## Logs

All logs are written to `logs/` with timestamps:
- `training_*.log` — training pipeline
- `generate_predictions_*.log` — batch scoring
- `training_eval_curve.csv` — training metrics (NDCG@K, MAP@K)