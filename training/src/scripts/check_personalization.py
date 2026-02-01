from __future__ import annotations

import polars as pl
import itertools
import numpy as np

from training.src.paths import INTERIM_DIR

# ===============================
# CONFIG
# ===============================
PREDICTIONS_PATH = INTERIM_DIR / "predictions.parquet"

N_ANCHORS = 100          # сколько anchor'ов проверяем 10
N_KIOSKS = 100           # сколько киосков на anchor 50
TOP_K = 5               # top-K для оценки 10
RANDOM_SEED = 42
MIN_KIOSKS_PER_ANCHOR = 10   # чтобы anchor был информативным


# ===============================
# UTILS
# ===============================
def weighted_jaccard(a: list[str], b: list[str]) -> float:
    """
    Rank-aware Jaccard:
    weight = 1 / rank
    """
    wa = {item: 1.0 / (i + 1) for i, item in enumerate(a)}
    wb = {item: 1.0 / (i + 1) for i, item in enumerate(b)}

    keys = set(wa) | set(wb)
    num = sum(min(wa.get(k, 0), wb.get(k, 0)) for k in keys)
    den = sum(max(wa.get(k, 0), wb.get(k, 0)) for k in keys)

    return num / den if den > 0 else 1.0


# ===============================
# MAIN
# ===============================
def main():
    print("[INFO] Loading predictions...")
    df = pl.read_parquet(PREDICTIONS_PATH)

    print(f"[INFO] Total rows: {df.shape}")

    rng = np.random.default_rng(RANDOM_SEED)

    # --------------------------------
    # 1. Select anchors with enough kiosks
    # --------------------------------
    anchor_stats = (
        df
        .group_by("anchor_product_id")
        .agg(pl.n_unique("kiosk_id").alias("n_kiosks"))
        .filter(pl.col("n_kiosks") >= MIN_KIOSKS_PER_ANCHOR)
    )

    anchors = anchor_stats["anchor_product_id"].to_list()

    if len(anchors) < N_ANCHORS:
        raise ValueError("Not enough anchors for evaluation")

    rng.shuffle(anchors)
    anchors = anchors[:N_ANCHORS]

    print(f"[INFO] Using {len(anchors)} anchors")

    anchor_results = []

    # --------------------------------
    # 2. Loop over anchors
    # --------------------------------
    for anchor in anchors:
        df_anchor = df.filter(pl.col("anchor_product_id") == anchor)

        kiosks = (
            df_anchor
            .select("kiosk_id")
            .unique()
            .to_series()
            .to_list()
        )

        rng.shuffle(kiosks)
        kiosks = kiosks[:N_KIOSKS]

        if len(kiosks) < 2:
            continue

        # top-K per kiosk (ORDERED)
        topk = {}
        for k in kiosks:
            items = (
                df_anchor
                .filter(pl.col("kiosk_id") == k)
                .sort("score", descending=True)
                .select("candidate_product_id")
                .head(TOP_K)
                .to_series()
                .to_list()
            )
            topk[k] = items

        overlaps = []
        for k1, k2 in itertools.combinations(topk.keys(), 2):
            overlaps.append(weighted_jaccard(topk[k1], topk[k2]))

        if overlaps:
            anchor_results.append(
                {
                    "anchor": anchor,
                    "mean_overlap": float(np.mean(overlaps)),
                    "std_overlap": float(np.std(overlaps)),
                }
            )

    if not anchor_results:
        raise RuntimeError("No valid anchors evaluated")

    # --------------------------------
    # 3. Aggregate results
    # --------------------------------
    mean_overlaps = [r["mean_overlap"] for r in anchor_results]
    std_overlaps = [r["std_overlap"] for r in anchor_results]

    mean_final = float(np.mean(mean_overlaps))
    std_final = float(np.std(mean_overlaps))

    # --------------------------------
    # 4. Report
    # --------------------------------
    print("\n========== PERSONALIZATION CHECK ==========")
    print(f"Anchors evaluated: {len(anchor_results)}")
    print(f"Kiosks per anchor: {N_KIOSKS}")
    print(f"Top-K: {TOP_K}")
    print("------------------------------------------")
    print(f"Mean weighted overlap: {mean_final:.3f}")
    print(f"Std  across anchors:   {std_final:.3f}")

    print("\nInterpretation:")
    if mean_final > 0.9:
        print("Recommendations are almost identical → MBA-like")
    elif mean_final > 0.7:
        print("Weak personalization")
    elif mean_final > 0.5:
        print("Moderate personalization")
    else:
        print("Strong personalization")

    print("===========================================\n")


if __name__ == "__main__":
    main()
