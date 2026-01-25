from __future__ import annotations
import polars as pl
import itertools
import numpy as np


# ===============================
# CONFIG
# ===============================
PREDICTIONS_PATH = "training/data/interim/predictions.parquet"  # ← поменяй если нужно
ANCHOR_PRODUCT_ID = "000295-003"   # ← выбери популярный anchor
N_KIOSKS = 50
TOP_K = 10
RANDOM_SEED = 42


# ===============================
# UTILS
# ===============================
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ===============================
# MAIN
# ===============================
def main():
    print("[INFO] Loading predictions...")
    df = pl.read_parquet(PREDICTIONS_PATH)

    print(f"[INFO] Total rows: {df.shape}")

    # -------------------------------
    # 1. Filter by anchor
    # -------------------------------
    df_anchor = df.filter(pl.col("anchor_product_id") == ANCHOR_PRODUCT_ID)

    if df_anchor.is_empty():
        raise ValueError(f"No predictions found for anchor {ANCHOR_PRODUCT_ID}")

    print(f"[INFO] Rows for anchor {ANCHOR_PRODUCT_ID}: {df_anchor.shape}")

    # -------------------------------
    # 2. Select kiosks
    # -------------------------------
    kiosks = (
        df_anchor
        .select("kiosk_id")
        .unique()
        .to_series()
        .to_list()
    )

    if len(kiosks) < 2:
        raise ValueError("Not enough kiosks to compute overlap")

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(kiosks)
    kiosks = kiosks[:N_KIOSKS]

    print(f"[INFO] Using {len(kiosks)} kiosks")

    # -------------------------------
    # 3. Build top-K lists per kiosk
    # -------------------------------
    topk_per_kiosk = {}

    for k in kiosks:
        topk = (
            df_anchor
            .filter(pl.col("kiosk_id") == k)
            .sort("score", descending=True)
            .select("candidate_product_id")
            .head(TOP_K)
            .to_series()
            .to_list()
        )
        topk_per_kiosk[k] = set(topk)

    # -------------------------------
    # 4. Pairwise Jaccard
    # -------------------------------
    overlaps = []

    for k1, k2 in itertools.combinations(topk_per_kiosk.keys(), 2):
        s1 = topk_per_kiosk[k1]
        s2 = topk_per_kiosk[k2]
        overlaps.append(jaccard(s1, s2))

    mean_overlap = float(np.mean(overlaps))
    std_overlap = float(np.std(overlaps))

    # -------------------------------
    # 5. Report
    # -------------------------------
    print("\n========== PERSONALIZATION CHECK ==========")
    print(f"Anchor product: {ANCHOR_PRODUCT_ID}")
    print(f"Kiosks sampled: {len(kiosks)}")
    print(f"Top-K: {TOP_K}")
    print("------------------------------------------")
    print(f"Mean Jaccard overlap: {mean_overlap:.3f}")
    print(f"Std  Jaccard overlap: {std_overlap:.3f}")

    print("\nInterpretation:")
    if mean_overlap > 0.9:
        print("Recommendations are almost identical → MBA-like")
    elif mean_overlap > 0.6:
        print("Weak personalization")
    elif mean_overlap > 0.4:
        print("Moderate personalization")
    else:
        print("Strong personalization")

    print("===========================================\n")


if __name__ == "__main__":
    main()
