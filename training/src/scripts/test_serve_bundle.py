"""End-to-end smoke test for serve_bundle with multi-level fallback.

Tests 5 scenarios:
  A. Known kiosk + known anchor  (LightGBM catalog hit)
  B. Unknown kiosk + known anchor (per-anchor MBA fallback)
  C. Known kiosk + unknown anchor (category + global fallback)
  D. Unknown kiosk + unknown anchor (global fallback only)
  E. Random sample of 200 real queries (latency + completeness)

Run:
  ./venv/bin/python -m training.src.scripts.test_serve_bundle
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import polars as pl

from training.src.io import load_parquet, load_products_csv
from training.src.logging_utils import setup_logging
from training.src.paths import EXTERNAL_DIR, INTERIM_DIR
from training.src.scripts.serve_bundle import build_bundle

LOGGER = logging.getLogger(__name__)

N_MAX = 20
N_MIN = 5


def _load_assets() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
    preds = load_parquet(INTERIM_DIR / "predictions.parquet", label="Predictions")
    anchor_fb = load_parquet(INTERIM_DIR / "popularity_fallback.parquet", label="Anchor fallback")
    products = load_products_csv(EXTERNAL_DIR / "products_v2.csv")

    cat_path = INTERIM_DIR / "category_fallback.parquet"
    glob_path = INTERIM_DIR / "global_fallback.parquet"
    cat_fb = load_parquet(cat_path, label="Category fallback") if cat_path.exists() else None
    glob_fb = load_parquet(glob_path, label="Global fallback") if glob_path.exists() else None

    return preds, anchor_fb, products, cat_fb, glob_fb


def _call_bundle(
    preds, anchor_fb, products, cat_fb, glob_fb,
    kiosk_id: str, anchor_product_id: str,
    *,
    included_products: list[str] | None = None,
    excluded_products: list[str] | None = None,
    allowed_categories: list[str] | None = None,
    n_group_key: int | None = None,
    n_min: int = N_MIN,
    n_max: int = N_MAX,
) -> tuple[pl.DataFrame, float]:
    t0 = time.perf_counter()
    result = build_bundle(
        preds, anchor_fb, products,
        kiosk_id=kiosk_id,
        anchor_product_id=anchor_product_id,
        included_products=included_products or [],
        excluded_products=excluded_products or [],
        allowed_categories=allowed_categories or [],
        n_group_key=n_group_key,
        n_min=n_min,
        n_max=n_max,
        category_fallback=cat_fb,
        global_fallback=glob_fb,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def _check_quality(
    result: pl.DataFrame,
    anchor_product_id: str,
    products: pl.DataFrame,
    label: str,
) -> dict:
    """Check basic quality of the bundle."""
    issues: list[str] = []
    n = result.height

    # 1. Not empty
    if n == 0:
        issues.append("EMPTY bundle")

    # 2. Anchor not in candidates (self-recommendation)
    if n > 0:
        cands = set(result["candidate_product_id"].to_list())
        if anchor_product_id in cands:
            issues.append(f"Anchor {anchor_product_id} appears in its own recommendations")

    # 3. No duplicates
    if n > 0:
        n_unique = result.select(pl.col("candidate_product_id").n_unique()).item()
        if n_unique < n:
            issues.append(f"Duplicates: {n} rows but {n_unique} unique candidates")

    # 4. Category coverage — how many categories represented
    n_categories = 0
    if n > 0 and "category" in result.columns:
        n_categories = result.select(
            pl.col("category").drop_nulls().n_unique()
        ).item()

    return {
        "label": label,
        "n_items": n,
        "n_categories": n_categories,
        "issues": issues,
    }


def _print_result(result: pl.DataFrame, label: str, elapsed_ms: float, quality: dict):
    status = "PASS" if not quality["issues"] and quality["n_items"] >= N_MIN else "FAIL"
    print(f"\n{'='*70}")
    print(f"[{status}] {label}")
    print(f"  Items: {quality['n_items']}/{N_MAX}  |  Categories: {quality['n_categories']}  |  Latency: {elapsed_ms:.1f} ms")
    if quality["issues"]:
        for issue in quality["issues"]:
            print(f"  !! {issue}")
    if result.height > 0:
        display_cols = ["candidate_product_id", "category", "score"]
        if "candidate_name" in result.columns:
            display_cols = ["candidate_product_id", "candidate_name", "category", "score"]
        print(result.select(display_cols).head(5))


def main() -> None:
    setup_logging("test_serve_bundle")
    # Suppress noisy sub-module loggers during testing
    for name in ("training.src.io", "training.src.scripts.serve_bundle"):
        logging.getLogger(name).setLevel(logging.WARNING)

    print("Loading assets...")
    preds, anchor_fb, products, cat_fb, glob_fb = _load_assets()

    # Get sample kiosk + anchor from the catalog
    sample_row = preds.select(["kiosk_id", "anchor_product_id"]).unique().sample(1, seed=42)
    known_kiosk = sample_row["kiosk_id"][0]
    known_anchor = sample_row["anchor_product_id"][0]

    # Known anchors from fallback
    fb_anchors = anchor_fb["anchor_product_id"].unique().to_list() if "anchor_product_id" in anchor_fb.columns else []

    # Pick an anchor NOT in the fallback (truly unknown)
    all_product_ids = products.select(pl.col("productid").cast(pl.Utf8)).to_series().to_list()
    catalog_anchors = set(preds["anchor_product_id"].unique().to_list())
    fb_anchor_set = set(fb_anchors)
    unknown_anchors = [p for p in all_product_ids if p not in catalog_anchors and p not in fb_anchor_set]
    unknown_anchor = unknown_anchors[0] if unknown_anchors else "TOTALLY_FAKE_999"

    # Look up anchor category for relevance check
    anchor_cat_row = products.filter(pl.col("productid").cast(pl.Utf8) == known_anchor).select("category").head(1)
    known_anchor_cat = anchor_cat_row.item() if anchor_cat_row.height > 0 else "?"

    results_summary: list[dict] = []

    # ---- Scenario A: Known kiosk + known anchor (catalog hit) ----
    result_a, ms_a = _call_bundle(preds, anchor_fb, products, cat_fb, glob_fb, known_kiosk, known_anchor)
    qa = _check_quality(result_a, known_anchor, products, f"A: Catalog hit — kiosk={known_kiosk[:12]}… anchor={known_anchor} ({known_anchor_cat})")
    _print_result(result_a, qa["label"], ms_a, qa)
    results_summary.append({**qa, "ms": ms_a, "scenario": "A"})

    # ---- Scenario B: Unknown kiosk + known anchor (per-anchor MBA) ----
    result_b, ms_b = _call_bundle(preds, anchor_fb, products, cat_fb, glob_fb, "UNKNOWN_KIOSK_999", known_anchor)
    qb = _check_quality(result_b, known_anchor, products, f"B: Unknown kiosk + known anchor={known_anchor} ({known_anchor_cat})")
    _print_result(result_b, qb["label"], ms_b, qb)
    results_summary.append({**qb, "ms": ms_b, "scenario": "B"})

    # ---- Scenario C: Known kiosk + unknown anchor (category/global fallback) ----
    result_c, ms_c = _call_bundle(preds, anchor_fb, products, cat_fb, glob_fb, known_kiosk, unknown_anchor)
    qc = _check_quality(result_c, unknown_anchor, products, f"C: Known kiosk + unknown anchor={unknown_anchor}")
    _print_result(result_c, qc["label"], ms_c, qc)
    results_summary.append({**qc, "ms": ms_c, "scenario": "C"})

    # ---- Scenario D: Unknown kiosk + unknown anchor (only global fallback) ----
    result_d, ms_d = _call_bundle(preds, anchor_fb, products, cat_fb, glob_fb, "UNKNOWN_KIOSK_999", "TOTALLY_FAKE_999")
    qd = _check_quality(result_d, "TOTALLY_FAKE_999", products, "D: Unknown kiosk + completely unknown anchor")
    _print_result(result_d, qd["label"], ms_d, qd)
    results_summary.append({**qd, "ms": ms_d, "scenario": "D"})

    # ---- Scenario E: Random sample stress test ----
    print(f"\n{'='*70}")
    print("E: Random sample — 200 real queries + 50 unknown combos")
    print("-" * 70)

    real_queries = (
        preds.select(["kiosk_id", "anchor_product_id"]).unique()
        .sample(n=min(200, preds.select(pl.struct(["kiosk_id", "anchor_product_id"]).n_unique()).item()), seed=123)
    )
    # Add unknown combos
    unknown_combos = pl.DataFrame({
        "kiosk_id": ["UNKNOWN_KIOSK"] * 25 + preds["kiosk_id"].unique().sample(25, seed=77).to_list(),
        "anchor_product_id": [unknown_anchor] * 25 + ["FAKE_ANCHOR"] * 25,
    })
    all_queries = pl.concat([real_queries, unknown_combos], how="vertical")

    latencies: list[float] = []
    empty_count = 0
    under_min_count = 0
    full_count = 0
    self_rec_count = 0
    dup_count = 0

    for row in all_queries.iter_rows(named=True):
        res, ms = _call_bundle(
            preds, anchor_fb, products, cat_fb, glob_fb,
            row["kiosk_id"], row["anchor_product_id"],
        )
        latencies.append(ms)
        n = res.height
        if n == 0:
            empty_count += 1
        if n < N_MIN:
            under_min_count += 1
        if n >= N_MAX:
            full_count += 1
        if n > 0:
            cands = set(res["candidate_product_id"].to_list())
            if row["anchor_product_id"] in cands:
                self_rec_count += 1
            n_unique = res.select(pl.col("candidate_product_id").n_unique()).item()
            if n_unique < n:
                dup_count += 1

    total = all_queries.height
    lat_arr = sorted(latencies)
    p50 = lat_arr[int(total * 0.5)]
    p95 = lat_arr[int(total * 0.95)]
    p99 = lat_arr[min(int(total * 0.99), total - 1)]

    print(f"  Queries tested   : {total}")
    print(f"  Empty bundles    : {empty_count} ({100*empty_count/total:.1f}%)")
    print(f"  Under N_MIN ({N_MIN})  : {under_min_count} ({100*under_min_count/total:.1f}%)")
    print(f"  Full ({N_MAX} items)  : {full_count} ({100*full_count/total:.1f}%)")
    print(f"  Self-recs        : {self_rec_count}")
    print(f"  Duplicates       : {dup_count}")
    print(f"  Latency p50      : {p50:.1f} ms")
    print(f"  Latency p95      : {p95:.1f} ms")
    print(f"  Latency p99      : {p99:.1f} ms")
    print(f"  Latency max      : {max(latencies):.1f} ms")

    # ---- Relevance check: do recommended items match kiosk purchase history? ----
    print(f"\n{'='*70}")
    print("F: Relevance check — do recommendations match kiosk history?")
    print("-" * 70)

    orders = pl.read_parquet(INTERIM_DIR / "orders_sample.parquet")
    sample_kiosks = preds.select("kiosk_id").unique().sample(100, seed=55)

    overlap_ratios: list[float] = []
    same_cat_ratios: list[float] = []

    # Build product→category map
    prod_cat = dict(zip(
        products.select(pl.col("productid").cast(pl.Utf8)).to_series().to_list(),
        products.select(pl.col("category").cast(pl.Utf8)).to_series().to_list(),
    ))

    for kiosk_row in sample_kiosks.iter_rows(named=True):
        kid = kiosk_row["kiosk_id"]

        # Get kiosk purchase history
        kiosk_history = set(
            orders.filter(pl.col("kiosk_id") == kid)
            .select("product_id").unique().to_series().to_list()
        )
        if not kiosk_history:
            continue

        # Categories the kiosk has bought
        kiosk_categories = {prod_cat.get(p, "__NONE__") for p in kiosk_history}

        # Get first anchor for this kiosk from catalog
        anchor_row = preds.filter(pl.col("kiosk_id") == kid).select("anchor_product_id").unique().head(1)
        if anchor_row.height == 0:
            continue
        anchor = anchor_row.item()

        res, _ = _call_bundle(preds, anchor_fb, products, cat_fb, glob_fb, kid, anchor)
        if res.height == 0:
            continue

        rec_products = set(res["candidate_product_id"].to_list())

        # What fraction of recs are in kiosk's purchase history? (repeat items)
        overlap = len(rec_products & kiosk_history) / len(rec_products) if rec_products else 0
        overlap_ratios.append(overlap)

        # What fraction of recs are in categories the kiosk has bought?
        rec_categories = {prod_cat.get(p, "__NONE__") for p in rec_products}
        cat_overlap = len(rec_categories & kiosk_categories) / len(rec_categories) if rec_categories else 0
        same_cat_ratios.append(cat_overlap)

    if overlap_ratios:
        avg_overlap = sum(overlap_ratios) / len(overlap_ratios)
        avg_cat = sum(same_cat_ratios) / len(same_cat_ratios)
        print(f"  Kiosks checked         : {len(overlap_ratios)}")
        print(f"  Avg item overlap       : {100*avg_overlap:.1f}% (recs already purchased by kiosk)")
        print(f"  Avg category overlap   : {100*avg_cat:.1f}% (rec categories already bought by kiosk)")
        print()
        print("  Interpretation:")
        print("    - Item overlap 30-60% = healthy mix of repeat + new items")
        print("    - Category overlap > 80% = recs match kiosk's product profile")
    else:
        print("  No kiosks with history found for relevance check.")

    # ---- Scenario G: Business rules ----
    print(f"\n{'='*70}")
    print("G: Business rules — testing parameter combinations")
    print("-" * 70)

    g_results: list[dict] = []

    def _g_test(label: str, *, check_fn, **kwargs):
        """Run a business-rule sub-test and record pass/fail."""
        res, ms = _call_bundle(
            preds, anchor_fb, products, cat_fb, glob_fb,
            known_kiosk, known_anchor, **kwargs,
        )
        passed, detail = check_fn(res)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}  ({res.height} items, {ms:.1f}ms)  {detail}")
        g_results.append({"label": label, "passed": passed, "detail": detail})

    # --- G1: excluded_products — pick products from baseline top-5, exclude them ---
    baseline, _ = _call_bundle(preds, anchor_fb, products, cat_fb, glob_fb, known_kiosk, known_anchor)
    if baseline.height >= 3:
        exclude_ids = baseline["candidate_product_id"].head(3).to_list()

        def check_excluded(res):
            found = set(res["candidate_product_id"].to_list()) & set(exclude_ids)
            if found:
                return False, f"Excluded items still present: {found}"
            return True, f"Excluded {exclude_ids} — none found in result"

        _g_test("G1: excluded_products", check_fn=check_excluded, excluded_products=exclude_ids)
    else:
        print("  [SKIP] G1: Not enough baseline items to test exclusion")

    # --- G2: included_products — force-include a product NOT already in recs ---
    all_prod_ids = products.select(pl.col("productid").cast(pl.Utf8)).to_series().to_list()
    baseline_set = set(baseline["candidate_product_id"].to_list())
    force_products = [p for p in all_prod_ids if p not in baseline_set and p != known_anchor][:2]

    if force_products:
        def check_included(res):
            result_set = set(res["candidate_product_id"].to_list())
            missing = [p for p in force_products if p not in result_set]
            if missing:
                return False, f"Force-included items missing: {missing}"
            return True, f"Forced {force_products} — all present in result"

        _g_test("G2: included_products", check_fn=check_included, included_products=force_products)
    else:
        print("  [SKIP] G2: Could not find products outside baseline to force-include")

    # --- G3: allowed_categories — restrict to 1 category ---
    if "category" in baseline.columns and baseline.height > 0:
        cats_in_baseline = baseline["category"].drop_nulls().unique().to_list()
        if cats_in_baseline:
            single_cat = [cats_in_baseline[0]]

            def check_cat_filter(res):
                if res.height == 0:
                    return True, f"No items in category {single_cat} — empty is OK"
                res_cats = set(res["category"].drop_nulls().to_list())
                bad = res_cats - set(single_cat)
                if bad:
                    return False, f"Items from forbidden categories: {bad}"
                return True, f"All {res.height} items belong to {single_cat}"

            _g_test("G3: allowed_categories (1 cat)", check_fn=check_cat_filter, allowed_categories=single_cat)

            # Also try 2 categories
            if len(cats_in_baseline) >= 2:
                two_cats = cats_in_baseline[:2]

                def check_two_cats(res):
                    if res.height == 0:
                        return True, f"No items in {two_cats}"
                    res_cats = set(res["category"].drop_nulls().to_list())
                    bad = res_cats - set(two_cats)
                    if bad:
                        return False, f"Items from forbidden categories: {bad}"
                    return True, f"All {res.height} items belong to {two_cats}"

                _g_test("G3b: allowed_categories (2 cats)", check_fn=check_two_cats, allowed_categories=two_cats)

    # --- G4: n_group_key — max items per category ---
    def check_ngroup(n_gk):
        def _check(res):
            if res.height == 0:
                return True, "Empty result"
            if "category" not in res.columns:
                return False, "No category column"
            cat_counts = res.group_by("category").agg(pl.len().alias("cnt"))
            violations = cat_counts.filter(pl.col("cnt") > n_gk)
            if violations.height > 0:
                bad = dict(zip(
                    violations["category"].to_list(),
                    violations["cnt"].to_list(),
                ))
                return False, f"Categories exceeding limit {n_gk}: {bad}"
            max_per_cat = cat_counts["cnt"].max()
            return True, f"Max {max_per_cat} items/category (limit={n_gk}), {cat_counts.height} categories"
        return _check

    _g_test("G4a: n_group_key=1", check_fn=check_ngroup(1), n_group_key=1)
    _g_test("G4b: n_group_key=2", check_fn=check_ngroup(2), n_group_key=2)
    _g_test("G4c: n_group_key=5", check_fn=check_ngroup(5), n_group_key=5)

    # --- G5: n_min / n_max variations ---
    def check_size(expected_max):
        def _check(res):
            if res.height > expected_max:
                return False, f"Got {res.height} items > n_max={expected_max}"
            return True, f"{res.height}/{expected_max} items"
        return _check

    _g_test("G5a: n_max=5 (small)", check_fn=check_size(5), n_max=5)
    _g_test("G5b: n_max=50 (large)", check_fn=check_size(50), n_max=50)
    _g_test("G5c: n_max=1 (minimal)", check_fn=check_size(1), n_max=1)

    # --- G6: Combined rules — exclusion + category filter + n_group_key ---
    if cats_in_baseline and baseline.height >= 3:
        combo_cat = [cats_in_baseline[0]]
        combo_exclude = exclude_ids[:1]  # exclude 1 item

        def check_combo(res):
            issues = []
            # Check exclusion
            found = set(res["candidate_product_id"].to_list()) & set(combo_exclude)
            if found:
                issues.append(f"Excluded items present: {found}")
            # Check category
            if res.height > 0 and "category" in res.columns:
                bad_cats = set(res["category"].drop_nulls().to_list()) - set(combo_cat)
                if bad_cats:
                    issues.append(f"Forbidden categories: {bad_cats}")
            # Check n_group_key
            if res.height > 0 and "category" in res.columns:
                cat_counts = res.group_by("category").agg(pl.len().alias("cnt"))
                violations = cat_counts.filter(pl.col("cnt") > 3)
                if violations.height > 0:
                    issues.append(f"n_group_key=3 violated: {dict(zip(violations['category'].to_list(), violations['cnt'].to_list()))}")
            # Check n_max
            if res.height > 10:
                issues.append(f"Exceeds n_max=10: got {res.height}")
            if issues:
                return False, "; ".join(issues)
            return True, f"{res.height} items, cat={combo_cat}, excl={combo_exclude}, n_gk=3, n_max=10"

        _g_test(
            "G6: Combined (excl + cat + n_group + n_max)",
            check_fn=check_combo,
            excluded_products=combo_exclude,
            allowed_categories=combo_cat,
            n_group_key=3,
            n_max=10,
        )

    # --- G7: included + excluded overlap — included should win ---
    if force_products and baseline.height >= 3:
        overlap_product = force_products[0]

        def check_incl_excl(res):
            result_set = set(res["candidate_product_id"].to_list())
            if overlap_product in result_set:
                return True, f"Included {overlap_product} present despite being in excluded list"
            return False, f"Included {overlap_product} was excluded — included should take priority"

        _g_test(
            "G7: included + excluded overlap",
            check_fn=check_incl_excl,
            included_products=[overlap_product],
            excluded_products=[overlap_product],
        )

    g_pass = sum(1 for r in g_results if r["passed"])
    g_total = len(g_results)
    print(f"\n  G sub-tests: {g_pass}/{g_total} passed")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("-" * 70)
    all_pass = True
    for r in results_summary:
        status = "PASS" if not r["issues"] and r["n_items"] >= N_MIN else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {r['scenario']}: {r['n_items']}/{N_MAX} items, {r['ms']:.1f}ms — {r['label']}")
    e_status = "PASS" if empty_count == 0 else "FAIL"
    if empty_count > 0:
        all_pass = False
    print(f"  [{e_status}] E: {total} queries, {empty_count} empty, p50={p50:.1f}ms, p95={p95:.1f}ms")
    g_status = "PASS" if g_pass == g_total else "FAIL"
    if g_pass < g_total:
        all_pass = False
    print(f"  [{g_status}] G: {g_pass}/{g_total} business-rule sub-tests passed")

    if all_pass:
        print("\n  All scenarios PASSED.")
    else:
        print("\n  Some scenarios FAILED — see details above.")


if __name__ == "__main__":
    main()
