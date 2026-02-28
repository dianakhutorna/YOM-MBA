"""Analyze whether top-K recs are repeat purchases or new-to-kiosk products."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

from training.src.features import add_all_features, lgbm_feature_exprs
from training.src.io import load_orders_csv_sample, load_products_csv, load_commerces_csv
from training.src.steps.build_baskets import build_baskets
from training.src.steps.build_feature_table import build_feature_table
from training.src.steps.generate_candidates import generate_candidates
from training.src.steps.preprocessing import preprocess_orders
from training.src.steps.select_top_k_candidates import select_top_k_candidates
from training.src.steps.split_orders import split_orders_by_time
from training.src.pipelines.training import fill_missing_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-kiosks", type=int, default=200)
    args = parser.parse_args()

    K = args.top_k

    # ── Load data ──────────────────────────────────────────────
    raw = load_orders_csv_sample(
        Path("training/data/raw/2024-20250000_part_00-003.csv"),
        n_rows=1_000_000, sample_position="tail",
    )
    orders = preprocess_orders(raw)
    products = load_products_csv(Path("training/data/external/products_v2.csv"))
    commerces = load_commerces_csv(Path("training/data/external/commerces.csv"))

    active = (
        commerces.filter(pl.col("active") == True)
        .select(pl.col("userid").cast(pl.Utf8).alias("kiosk_id"))
        .unique()
    )
    orders = orders.join(active, on="kiosk_id", how="inner")

    train_orders, _, test_orders = split_orders_by_time(
        orders, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    )

    baskets = build_baskets(train_orders)
    cands = generate_candidates(baskets, min_cooc=2)
    topk = select_top_k_candidates(cands, k=100, min_lift=1.2)

    # ── Build queries from test kiosks ─────────────────────────
    queries = (
        build_baskets(test_orders)
        .select(["kiosk_id", "products"])
        .explode("products")
        .rename({"products": "anchor_product_id"})
        .unique()
    )
    sampled = queries.select("kiosk_id").unique().sample(n=args.sample_kiosks, seed=42)
    queries = queries.join(sampled, on="kiosk_id", how="inner")

    ft = build_feature_table(baskets=baskets, topk_candidates=topk, queries=queries)
    ft = add_all_features(ft, orders=train_orders, products=products, commerces=commerces)

    # ── Score ──────────────────────────────────────────────────
    ranker = lgb.Booster(model_file="training/models/lgbm_ranker.txt")
    feature_cols = json.loads(Path("training/models/lgbm_ranker.features.json").read_text())
    cat_cols = [c for c in ("channel", "region") if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in set(cat_cols)]
    ft = fill_missing_features(ft, num_cols, cat_cols)

    scores = ranker.predict(
        ft.select(lgbm_feature_exprs(feature_cols, cat_cols)).to_numpy()
    )
    scored = ft.select(
        ["kiosk_id", "anchor_product_id", "candidate_product_id",
         "cand_is_new", "pop_store"]
    ).with_columns(pl.Series("score", scores))

    # ── Top-K ──────────────────────────────────────────────────
    top = (
        scored
        .sort(["kiosk_id", "anchor_product_id", "score"], descending=[False, False, True])
        .group_by(["kiosk_id", "anchor_product_id"])
        .head(K)
    )

    total = top.height
    n_new = top.filter(pl.col("cand_is_new") == 1).height
    n_repeat = total - n_new

    print(f"\n{'='*50}")
    print(f"  Top-{K} Recommendations: New vs Repeat")
    print(f"{'='*50}")
    print(f"  Total recs : {total:,}")
    print(f"  Repeat     : {n_repeat:,}  ({100*n_repeat/total:.1f}%)")
    print(f"  New        : {n_new:,}  ({100*n_new/total:.1f}%)")

    # ── By position ────────────────────────────────────────────
    # Add rank within group
    ranked = (
        scored
        .sort(["kiosk_id", "anchor_product_id", "score"], descending=[False, False, True])
        .with_columns(
            pl.col("score")
            .rank(method="ordinal", descending=True)
            .over(["kiosk_id", "anchor_product_id"])
            .alias("rank")
        )
    )

    print(f"\n  By position:")
    for pos in range(1, K + 1):
        pos_df = ranked.filter(pl.col("rank") == pos)
        n = pos_df.height
        if n == 0:
            continue
        n_new_p = pos_df.filter(pl.col("cand_is_new") == 1).height
        print(f"    #{pos}: {100*n_new_p/n:.1f}% new")

    # ── Candidate pool comparison ──────────────────────────────
    total_all = scored.height
    n_new_all = scored.filter(pl.col("cand_is_new") == 1).height
    print(f"\n  Candidate pool (all candidates):")
    print(f"    Repeat : {total_all - n_new_all:,}  ({100*(total_all-n_new_all)/total_all:.1f}%)")
    print(f"    New    : {n_new_all:,}  ({100*n_new_all/total_all:.1f}%)")

    # ── Avg pop_store ──────────────────────────────────────────
    avg_all = scored.select(pl.col("pop_store").mean()).item()
    avg_top = top.select(pl.col("pop_store").mean()).item()
    avg_top_repeat = top.filter(pl.col("cand_is_new") == 0).select(pl.col("pop_store").mean()).item()

    print(f"\n  Avg pop_store:")
    print(f"    All candidates : {avg_all:.1f}")
    print(f"    Top-{K}          : {avg_top:.1f}")
    print(f"    Top-{K} repeat   : {avg_top_repeat:.1f}")

    # ── Example: show a kiosk with mixed new/repeat ────────────
    prod_names = products.select([
        pl.col("productid").cast(pl.Utf8).alias("candidate_product_id"),
        pl.col("name"),
    ])

    # Find kiosks that have both new and repeat in top-K
    mixed = (
        top.group_by(["kiosk_id", "anchor_product_id"])
        .agg([
            pl.col("cand_is_new").sum().alias("n_new"),
            pl.col("cand_is_new").count().alias("n_total"),
        ])
        .filter((pl.col("n_new") > 0) & (pl.col("n_new") < pl.col("n_total")))
    )
    if mixed.height > 0:
        ex = mixed.head(3)
        print(f"\n  Example queries with MIXED new+repeat:")
        for row in ex.iter_rows(named=True):
            kid, aid = row["kiosk_id"], row["anchor_product_id"]
            recs = (
                ranked.filter(
                    (pl.col("kiosk_id") == kid) & (pl.col("anchor_product_id") == aid)
                    & (pl.col("rank") <= K)
                )
                .sort("rank")
                .join(prod_names, on="candidate_product_id", how="left")
            )
            anchor_name = products.filter(
                pl.col("productid").cast(pl.Utf8) == aid
            ).select("name").item()
            print(f"\n    Kiosk {kid} | Anchor: {anchor_name}")
            for r in recs.iter_rows(named=True):
                tag = "NEW" if r["cand_is_new"] == 1 else "   "
                name = r.get("name", "?")
                print(f"      #{int(r['rank'])} [{tag}] pop={int(r['pop_store']):3d}  {name}")

    print()


if __name__ == "__main__":
    main()
