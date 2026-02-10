# Bundle Recommendations
   Offline ML pipeline for generating bundle recommendations and serving them from precomputed files.
   **Key idea**
  - Train offline → generate `predictions.parquet` → serve bundles with rules (no online ML inference).
   ---
   **Project Layout**
  - `training/src/pipelines/training.py` — training pipeline (LTR).
  - `training/src/scripts/generate_predictions.py` — catalog generation.
  - `training/src/scripts/serve_bundle.py` — bundle serving with rules and fallback.
  - `training/configs/*.yaml` — configs.
   ---
   **Requirements**
  - Python 3.11
  - pip-25.3
  - `pip install -r requirements.txt`
   ---
   **Setup**
  1. Create and activate a virtual environment:
  ```bash
  python -m venv venv
  Windows: venv\Scripts\activate
  Mac/Linux: source venv/bin/activate
  ```
   2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
   ---
   **Data**
  Put the data in the following orders:
  - `training/data/external/`
    - `commerces.csv`
    - `products_v2.csv`
  - `training/data/raw/`
    - `2022-20230000_part_00-002.csv`
    - `2022-20230001_part_00-004.csv`
    - `2024-20250000_part_00-003.csv`
    - `2024-20250001_part_00-001.csv`
   ---
   **1) Training**
  ```bash
  ./venv/bin/python -m training.src.scripts.run_training_pipeline \
    --config training/configs/training_pipeline.yaml
  ```
   Result:
  - `training/models/lgbm_ranker.txt`
  - `training/models/lgbm_ranker.features.json`
  - logs: `logs/training_*.log`
   ---
   **2) Predictions generation**
  ```bash
  ./venv/bin/python -m training.src.scripts.generate_predictions \
    --config training/configs/generate_predictions.yaml
  ```
   Result:
  - `training/data/interim/predictions.parquet`
  - `training/data/interim/popularity_fallback.parquet`
  - logs: `logs/generate_predictions_*.log`
   ---
   **3) Serve bundles**
  You can run through CLI **or** through config `training/configs/serve_bundle.yaml`.
   Example CLI:
  ```bash
  python -m training.src.scripts.serve_bundle \
    --kiosk-id 30037f531441414d92ac845f7f3e1357 \
    --anchor-product-id 004752-001 \
    --excluded-products 004747-001 \
    --n-group-key 3 \
    --n-min 4 \
    --n-max 10
  ```
   Example using `serve_bundle.yaml`:
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
   **Configs**
  Main:
  - `training/configs/training_pipeline.yaml`
  - `training/configs/generate_predictions.yaml`
  - `training/configs/serve_bundle.yaml`
  - `training/configs/features.yaml`
   ---
   **Logs**
  Logs are written in `logs/` with timestamps:
  - `training_*.log`
  - `generate_predictions_*.log`
  - `serve_bundle_*.log`
 
  