from pathlib import Path
import polars as pl

from training.src.steps.select_top_k_candidates import select_top_k_candidates


CANDIDATES_PATH = Path("training/data/interim/candidates_sample.parquet")
TOPK_PATH = Path("training/data/interim/topk_candidates_sample.parquet")


def main():
    candidates = pl.read_parquet(CANDIDATES_PATH)
    print(f"[INFO] Loaded candidates: {candidates.shape}")

    topk = select_top_k_candidates(
        candidates,
        k=20,
        min_lift=2.0,
    )

    print(topk.head(10))

    topk.write_parquet(TOPK_PATH)
    print(f"[OK] Saved top-K candidates to {TOPK_PATH}")


if __name__ == "__main__":
    main()
