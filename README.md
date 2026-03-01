# YOM Bundle Recommender System

Offline ML pipeline that **trains** a ranking model, **generates** bundle predictions in batch, and **serves** them via a FastAPI endpoint with business rules and multi-level fallback.

No online ML inference — the model scores all (kiosk, anchor, candidate) triples ahead of time; serving is a simple dict lookup (~2 ms/request).

---

## Architecture

```
┌──────────────────────────┐     ┌──────────────────────────────┐     ┌─────────────────────────────┐
│   1. TRAINING (rare)     │     │ 2. BATCH SCORING (daily)     │     │  3. SERVING (24/7)          │
│   training.py            │     │ generate_predictions.py      │     │  serve_bundle_api.py        │
├──────────────────────────┤     ├──────────────────────────────┤     ├─────────────────────────────┤
│ Raw CSV → preprocess     │     │ Load model + recent orders   │     │ Load 4 parquets at startup  │
│ Time split train/val/test│     │ MBA candidates (90-day)      │     │ Dict-index for O(1) lookup  │
│ MBA candidates + features│ →   │ Feature table → LightGBM     │ →   │ 4-level fallback:           │
│ LightGBM LambdaRank      │     │ Top-20 per (kiosk, anchor)   │     │   1. Model predictions      │
│ Save model + features.json     │ Save 4 parquet artifacts     │     │   2. Per-anchor MBA         │
└──────────────────────────┘     └──────────────────────────────┘     │   3. Per-category popular   │
                                                                      │   4. Global popular         │
                                                                      │ Business rules + JSON resp  │
                                                                      └─────────────────────────────┘
```

---

## Project Structure

```
training/
├── configs/
│   ├── training_pipeline.yaml        # Training hyperparameters
│   ├── generate_predictions.yaml     # Batch inference settings
│   ├── serve_bundle.yaml             # Serving defaults + paths
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
│   │   └── training.py               # End-to-end training pipeline
│   ├── scripts/
│   │   ├── run_training_pipeline.py   # CLI: run training
│   │   ├── generate_predictions.py    # CLI: batch scoring → 4 parquets
│   │   ├── serve_bundle.py            # Bundle logic + business rules
│   │   ├── serve_bundle_api.py        # FastAPI service
│   │   ├── test_serve_bundle.py       # Smoke + business rules tests
│   │   ├── check_personalization.py   # Personalization analysis
│   │   └── check_new_vs_repeat.py     # New vs repeat item analysis
│   ├── steps/                         # Modular pipeline steps
│   ├── io/                            # I/O helpers (load/save)
│   ├── features.py                    # Feature orchestrator
│   ├── config.py                      # YAML config loader
│   ├── paths.py                       # Canonical path constants
│   └── logging_utils.py              # Logging setup
├── tests/                             # Unit tests (pytest)
└── logs/                              # Runtime logs
```

---

## Requirements

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

### Option A: FastAPI (production)

```bash
BUNDLE_CONFIG=training/configs/serve_bundle.yaml \
uvicorn training.src.scripts.serve_bundle_api:app \
  --host 0.0.0.0 --port 8000
```

Startup takes ~60–90 seconds (loading 256 MB parquet + building dict-index). After that, requests are ~2 ms.

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Readiness probe: `{"status": "ok"}` |
| `/bundle` | GET | Get personalized bundle |

**`/bundle` query parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `kiosk_id` | string | yes | — | Kiosk identifier |
| `anchor_product_id` | string | yes | — | Trigger product |
| `included_products` | string | no | — | CSV list of product IDs to force-include |
| `excluded_products` | string | no | — | CSV list of product IDs to exclude |
| `allowed_categories` | string | no | — | CSV list of allowed categories |
| `n_group_key` | int | no | — | Max items per category |
| `n_min` | int | no | 10 | Min bundle size |
| `n_max` | int | no | 20 | Max bundle size |

**Example request:**
```
GET /bundle?kiosk_id=fe7ef5cd7c27...&anchor_product_id=000056-002&n_max=5&n_min=1
```

**Example response:**
```json
{
  "kiosk_id": "fe7ef5cd7c27...",
  "anchor_product_id": "000056-002",
  "n_items": 5,
  "latency_ms": 2.1,
  "items": [
    {
      "candidate_product_id": "002030-001",
      "candidate_name": "SOPROLE LECHE BLANCA",
      "category": "Leche Liquida",
      "score": 2.81
    }
  ]
}
```

### Option B: CLI (testing)

```bash
./venv/bin/python -m training.src.scripts.serve_bundle \
  --kiosk-id fe7ef5cd7c273ec75600d6c710216f69 \
  --anchor-product-id 000056-002 \
  --excluded-products 004747-001 \
  --n-group-key 3 --n-min 4 --n-max 10
```

Or use defaults from `serve_bundle.yaml`:
```bash
./venv/bin/python -m training.src.scripts.serve_bundle
```

---

## 4. Tests

```bash
# Unit tests (18 tests, ~1s)
./venv/bin/python -m pytest training/tests/ -q

# Smoke test — serve_bundle end-to-end (requires generated parquets)
./venv/bin/python -m training.src.scripts.test_serve_bundle
```

The smoke test covers 7 scenario groups (A–G):
- **A:** Known kiosk + known anchor (catalog hit)
- **B:** Unknown kiosk + known anchor (per-anchor fallback)
- **C:** Known kiosk + unknown anchor (category/global fallback)
- **D:** Unknown kiosk + unknown anchor (global fallback only)
- **E:** 250 random queries (latency + completeness)
- **F:** Relevance check (category overlap with purchase history)
- **G:** Business rules — 12 sub-tests: exclusions, inclusions, category filters, `n_group_key`, `n_max` variations, combined rules

---

## Fallback Logic

Bundles are **never empty**. If a level produces insufficient items, the next level fills the gap:

| Level | Source | When used |
|-------|--------|-----------|
| 1 | LightGBM predictions | Known kiosk + known anchor |
| 2 | Per-anchor MBA co-purchase | Unknown kiosk, known anchor |
| 3 | Per-category popularity | Unknown anchor (uses anchor's category) |
| 4 | Global popularity | Everything unknown (last resort) |

After filling, `n_group_key` is enforced across all sources to guarantee category diversity.

---

## Configs

| Config | Used by | Purpose |
|--------|---------|---------|
| `training_pipeline.yaml` | `run_training_pipeline.py` | Hyperparams, paths, split ratios |
| `generate_predictions.yaml` | `generate_predictions.py` | Inference window, candidate settings |
| `serve_bundle.yaml` | `serve_bundle.py`, API | Bundle defaults, file paths |
| `features.yaml` | `features.py` | Feature group flags (legacy) |

---

## Logs

All logs are written to `logs/` with timestamps:
- `training_*.log` — training pipeline
- `generate_predictions_*.log` — batch scoring
- `serve_bundle_*.log` — CLI bundle serving
- `test_serve_bundle_*.log` — smoke test