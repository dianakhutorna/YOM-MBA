from __future__ import annotations

from training.src.steps.generate_candidates_hybrid_mba_kiosk import generate_candidates_hybrid_mba_kiosk


def test_hybrid_mba_kiosk_columns(baskets_df):
    result = generate_candidates_hybrid_mba_kiosk(
        baskets_df,
        min_cooc=1,
        min_lift=0.0,
        top_k=6,
        kiosk_share=0.5,
        kiosk_batch_size=20,
    )
    expected = {
        "kiosk_id",
        "anchor_product_id",
        "candidate_product_id",
        "cooc_cosine_sim",
        "source",
    }
    assert expected.issubset(set(result.columns))


def test_hybrid_mba_kiosk_top_k_per_query(baskets_df):
    top_k = 5
    result = generate_candidates_hybrid_mba_kiosk(
        baskets_df,
        min_cooc=1,
        min_lift=0.0,
        top_k=top_k,
        kiosk_share=0.5,
        kiosk_batch_size=20,
    )
    per_query = result.group_by(["kiosk_id", "anchor_product_id"]).len()
    assert per_query["len"].max() <= top_k
    assert set(result["source"].unique().to_list()).issubset({"kiosk_mba", "global_mba"})
