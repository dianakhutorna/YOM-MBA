# Bundle Recommendations

This service provides bundle recommendations based on offline batch inference.

ML pipeline:
- training.py (offline)
- generate_predictions.py (offline)

Serving:
- uses predictions.parquet
- no online ML inference
- supports filtering and business rules


Used Python 3.11.9
pip-25.3

First you need following VSC extensions:

Container Tools
Docker
GitHub Pull Requests
Jupyter
Jupyter Cell Tags
Jupyter Keymap
Jupyter Notebook Renderers
Jupyter Slide Show
Python
Python Debugger
Python Environments

1. Create virtual environment
python -m venv venv
& "C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv

2. Activate virtual environment
Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate

3. Install the libraries:
pip install -r requirements.txt

python -m ipykernel install --user --name=YOM-venv --display-name "Python 3.11 (YOM-venv)"

For Notebooks use (venv) Python 3.11 !!!

Put in the data folder following files:

external:
commerces.csv
products_v2.csv

internal:
(empty)

raw:
2022-20230000_part_00-002.csv
2022-20230001_part_00-004.csv
2024-20250000_part_00-003.csv
2024-20250001_part_00-001.csv



./venv/bin/python -m training.src.scripts.run_training_pipeline --config training/configs/training_pipeline.yaml

./venv/bin/python -m training.src.scripts.generate_predictions --config training/configs/generate_predictions.yaml

./venv/bin/python -m training.src.scripts.serve_bundle \            
  --kiosk-id 30037f531441414d92ac845f7f3e1357 \           
  --anchor-product-id 004752-001 \
  --excluded-products 004747-001 \
  --n-group-key 3 \
  --n-min 4 \
  --n-max 10

./venv/bin/python -m training.src.scripts.serve_bundle \
  --kiosk-id 30037f531441414d92ac845f7f3e1357 \
  --anchor-product-id 004752-001 \
  --excluded-products 004747-001 \
  --n-group-key 3 \
  --n-min 4 \
  --n-max 10

