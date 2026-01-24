from pathlib import Path

from training.src.steps.load_data import load_orders_sample
from training.src.steps.preprocessing import preprocess_orders


RAW_FILE = Path("training/data/2024-20250001_part_00-001.csv")
INTERIM_FILE = Path("training/data/interim/orders_sample.parquet")


def main():
    df_raw = load_orders_sample(
        raw_path=RAW_FILE,
        n_rows=100_000,   # маленький тест
    )

    df_clean = preprocess_orders(df_raw)

    print(df_clean.head())
    print(df_clean.describe())

    df_clean.write_parquet(INTERIM_FILE)
    print(f"[OK] Saved cleaned data to {INTERIM_FILE}")


if __name__ == "__main__":
    main()
