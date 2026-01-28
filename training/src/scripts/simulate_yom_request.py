from __future__ import annotations

from pathlib import Path
import polars as pl


# ===============================
# CONFIG
# ===============================
PREDICTIONS_PATH = Path("training/data/interim/predictions.parquet")
PRODUCTS_PATH = Path("training/data/products_v2.csv")  # чтобы взять category

# ТВОЙ ЗАПРОС
YOM_REQUEST = {
    "kiosk_id": "30037f531441414d92ac845f7f3e1357",
    "anchor_product_id": "004752-001",

    "included_products": [],                 # форсируем в результат
    "excluded_products": ["004747-001"],     # удаляем из результата

    # allowed categories
    "agg_key": None,  # {"Yoghurt", "Jugo", "Queso Maduro"},  # если пусто/None → не фильтруем по категориям
    "n_group_key": 3,   # максимум товаров на категорию
    "N_min": 4,
    "N_max": 10,
}


# ===============================
# CORE LOGIC
# ===============================
def apply_yom_preprocessing(
    df_scored: pl.DataFrame,
    products: pl.DataFrame,
    req: dict,
) -> pl.DataFrame:
    """
    df_scored: строки только для одного (kiosk_id, anchor_product_id) со score
    """
    # join category для candidate
    prod_map = products.select(
        [
            pl.col("productid").cast(pl.Utf8).alias("product_id"),
            pl.col("category").cast(pl.Utf8),
        ]
    )

    df = df_scored.with_columns(
        pl.col("candidate_product_id").cast(pl.Utf8)
    ).join(
        prod_map,
        left_on="candidate_product_id",
        right_on="product_id",
        how="left",
    )


    # 1) excluded
    excluded = req.get("excluded_products") or []
    if excluded:
        df = df.filter(~pl.col("candidate_product_id").is_in(excluded))

    # 2) allowed categories (agg_key)
    allowed_cats = req.get("agg_key")
    if allowed_cats:  # set/list
        df = df.filter(pl.col("category").is_in(list(allowed_cats)))

    # 3) сортировка по score (внутри query)
    df = df.sort("score", descending=True)

    # 4) n_group_key (лимит на категорию)
    n_group_key = req.get("n_group_key")
    if n_group_key is not None and int(n_group_key) > 0:
        df = (
            df
            .with_columns(
                pl.col("category")
                .cum_count()
                .over("category")
                .alias("_cat_rank")
            )
            .filter(pl.col("_cat_rank") <= int(n_group_key))
            .drop("_cat_rank")
        )

    # 5) included_products (форсим в топ, если их нет)
    included = req.get("included_products") or []
    if included:
        included_set = set(included)
        present = set(df.select("candidate_product_id").to_series().to_list())
        missing = list(included_set - present)

        if missing:
            # добавляем отсутствующие included с очень большим score, чтобы гарантированно попали
            # (это чисто бизнес-логика; модель тут ни при чем)
            max_score = df.select(pl.max("score")).item()
            bonus = (max_score if max_score is not None else 0.0) + 1e6

            add_rows = pl.DataFrame(
                {
                    "kiosk_id": [req["kiosk_id"]] * len(missing),
                    "anchor_product_id": [req["anchor_product_id"]] * len(missing),
                    "candidate_product_id": missing,
                    "score": [bonus] * len(missing),
                }
            ).join(
                prod_map,
                left_on="candidate_product_id",
                right_on="product_id",
                how="left",
            )

            df = pl.concat([add_rows, df], how="vertical").sort("score", descending=True)

    # 6) N_max / N_min
    N_max = int(req.get("N_max", 10))
    N_min = int(req.get("N_min", 0))

    df = df.head(N_max)

    if df.height < N_min:
        # не падаем — просто предупреждаем. fallback = отдельная история.
        print(f"[WARN] Returned only {df.height} items (< N_min={N_min}).")

    return df


def main():
    req = YOM_REQUEST

    print("[INFO] Loading predictions.parquet (NO TRAINING, NO INFERENCE PIPELINE)")
    preds = pl.read_parquet(PREDICTIONS_PATH)

    # IMPORTANT: predictions.parquet у тебя уже финальный top-N.
    # Если он был сохранен с FINAL_N=10, ты НЕ сможешь потом получить 50 кандидатов.
    # Для более гибкого pre-processing лучше сохранять top-100 scored.
    # Но для теста бизнес-фильтров и этого хватает.

    print("[INFO] Filtering for kiosk+anchor")
    df_scored = preds.filter(
        (pl.col("kiosk_id") == req["kiosk_id"]) &
        (pl.col("anchor_product_id") == req["anchor_product_id"])
    )

    if df_scored.is_empty():
        print("[WARN] No rows found for this (kiosk_id, anchor_product_id).")
        print("       Check that this pair exists in predictions.parquet.")
        return

    print(f"[INFO] Found rows: {df_scored.shape[0]}")

    print("[INFO] Loading products_v2.csv for category join")
    products = pl.read_csv(PRODUCTS_PATH, separator=";")

    final = apply_yom_preprocessing(df_scored, products, req)

    print("\n===== FINAL RECOMMENDATIONS (AFTER PRE-PROCESSING) =====\n")
    print(
        final.select(
            [
                "kiosk_id",
                "anchor_product_id",
                "candidate_product_id",
                "category",
                pl.col("score").round(6).alias("score"),
            ]
        )
    )


if __name__ == "__main__":
    main()
