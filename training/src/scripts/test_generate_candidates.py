from pathlib import Path
import polars as pl

from training.src.steps.generate_candidates import generate_candidates


BASKETS_PATH = Path("training/data/interim/baskets_sample.parquet")
CANDIDATES_PATH = Path("training/data/interim/candidates_sample.parquet")


def main():
    baskets = pl.read_parquet(BASKETS_PATH)
    print(f"[INFO] Loaded baskets: {baskets.shape}")

    candidates = generate_candidates(baskets, min_cooc=3)

    print(candidates.head(10))
    print(candidates.select("cooc_count").describe())

    candidates.write_parquet(CANDIDATES_PATH)
    print(f"[OK] Saved candidates to {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
