# Bundle Recommendations

Offline ML pipeline for generating **bundle recommendations** and serving them from **precomputed files**.

The system avoids online ML inference by training models offline, generating predictions in advance, and applying lightweight business rules at serving time.

---

## Key Idea

**Train offline → generate predictions → serve bundles**

Pipeline:
1. Train an LTR model offline
2. Generate `predictions.parquet`
3. Serve bundles using rules and fallbacks (no online inference)

---

## Project Structure

```
training/
├── src/
│   ├── pipelines/
│   │   └── training.py              # Training pipeline (LTR)
│   └── scripts/
│       ├── run_training_pipeline.py
│       ├── generate_predictions.py
│       └── serve_bundle.py           # Bundle serving with rules & fallback
├── configs/
│   ├── training_pipeline.yaml
│   ├── generate_predictions.yaml
│   ├── serve_bundle.yaml
│   └── features.yaml
├── data/
│   ├── raw/
│   ├── external/
│   └── interim/
└── models/
```

---

## Requirements

- Python **3.11**
- pip **25.3**

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Data

Place input data in the following directories:

```
training/data/external/
├── commerces.csv
└── products_v2.csv

training/data/raw/
├── 2022-20230000_part_00-002.csv
├── 2022-20230001_part_00-004.csv
├── 2024-20250000_part_00-003.csv
└── 2024-20250001_part_00-001.csv
```

---

## 1️⃣ Training

Run the training pipeline:

```bash
# Mac / Linux
./venv/bin/python -m training.src.scripts.run_training_pipeline --config training/configs/training_pipeline.yaml

# Windows
python -m training.src.scripts.run_training_pipeline --config training/configs/training_pipeline.yaml

```

**Outputs:**
- `training/models/lgbm_ranker.txt`
- `training/models/lgbm_ranker.features.json`
- logs: `logs/training_*.log`

---

## 2️⃣ Predictions Generation

Generate bundle predictions:

```bash
# Mac / Linux
./venv/bin/python -m training.src.scripts.generate_predictions --config training/configs/generate_predictions.yaml

# Windows
python -m training.src.scripts.generate_predictions --config training/configs/generate_predictions.yaml

```

**Outputs:**
- `training/data/interim/predictions.parquet`
- `training/data/interim/popularity_fallback.parquet`
- logs: `logs/generate_predictions_*.log`

---

## 3️⃣ Serve Bundles

Bundles can be served via **CLI** or **YAML config**.

### CLI example

```bash
# Mac / Linux
python -m training.src.scripts.serve_bundle --kiosk-id 30037f531441414d92ac845f7f3e1357 --anchor-product-id 004752-001 --excluded-products 004747-001 --n-group-key 3 --n-min 4 --n-max 10

python -m training.src.scripts.serve_bundle --kiosk-id c6fd182599091ddb67ebb5d972d92685 --anchor-product-id 002395-002 --excluded-products 004747-001 --n-group-key 3 --n-min 4 --n-max 10

python -m training.src.scripts.serve_bundle --kiosk-id 6c052c61e2246ede3ee7324faa41da28 --anchor-product-id 002360-002 --n-group-key 3 --n-min 4 --n-max 10


# Windows
python -m training.src.scripts.serve_bundle --kiosk-id 30037f531441414d92ac845f7f3e1357 --anchor-product-id 004752-001 --excluded-products 004747-001 --n-group-key 3 --n-min 4 --n-max 10

```

### Using `serve_bundle.yaml`

```yaml
kiosk_id: "30037f531441414d92ac845f7f3e1357"
anchor_product_id: "004752-001"
excluded_products: "004747-001"
n_group_key: 3
n_min: 4
n_max: 10

predictions_path: training/data/interim/predictions.parquet
popularity_path: training/data/interim/popularity_fallback.parquet
products_path: training/data/external/products_v2.csv
```

Run without parameters:
```bash
python -m training.src.scripts.serve_bundle
```

---

## Configs

Main configuration files:
- `training/configs/training_pipeline.yaml`
- `training/configs/generate_predictions.yaml`
- `training/configs/serve_bundle.yaml`
- `training/configs/features.yaml`

---

## Logs

All logs are written to `logs/` with timestamps:
- `training_*.log`
- `generate_predictions_*.log`
- `serve_bundle_*.log`
 
  