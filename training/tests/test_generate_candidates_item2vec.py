from __future__ import annotations

from training.src.steps.generate_candidates_item2vec import generate_candidates_item2vec


def test_item2vec_candidates_columns(baskets_df):
    result = generate_candidates_item2vec(
        baskets_df,
        min_cooc=1,
        top_k=5,
        embedding_dim=16,
        svd_n_iter=5,
        random_state=42,
    )
    expected = {
        "anchor_product_id",
        "candidate_product_id",
        "cooc_count",
        "anchor_count",
        "candidate_count",
        "support",
        "confidence",
        "lift",
        "cooc_cosine_sim",
        "embedding_cosine_sim",
    }
    assert expected.issubset(set(result.columns))


def test_item2vec_top_k_per_anchor(baskets_df):
    top_k = 4
    result = generate_candidates_item2vec(
        baskets_df,
        min_cooc=1,
        top_k=top_k,
        embedding_dim=16,
        svd_n_iter=5,
        random_state=42,
    )
    per_anchor = result.group_by("anchor_product_id").len()
    assert per_anchor["len"].max() <= top_k
