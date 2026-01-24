from pathlib import Path
import polars as pl

from training.src.steps.build_feature_table import build_feature_table


BASKETS_PATH = Path("training/data/interim/baskets_sample.parquet")
TOPK_PATH = Path("training/data/interim/topk_candidates_sample.parquet")
FEATURES_PATH = Path("training/data/interim/features_sample.parquet")


def main():
    baskets = pl.read_parquet(BASKETS_PATH)
    topk = pl.read_parquet(TOPK_PATH)

    features = build_feature_table(baskets, topk)

    print(features.head(10))
    print(features.select(pl.col("lift")).describe())

    features.write_parquet(FEATURES_PATH)
    print(f"[OK] Saved features to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
