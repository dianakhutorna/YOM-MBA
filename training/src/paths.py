from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "training"

DATA_DIR = TRAINING_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
EXTERNAL_DIR = DATA_DIR / "external"

MODELS_DIR = TRAINING_DIR / "models"
LOGS_DIR = TRAINING_DIR / "logs"
RESULTS_DIR = TRAINING_DIR / "results"
NOTEBOOKS_DIR = TRAINING_DIR / "notebooks"
