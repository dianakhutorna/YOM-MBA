from __future__ import annotations


def test_generate_candidates_columns(candidates_df):
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
    }
    assert expected.issubset(set(candidates_df.columns))


def test_generate_candidates_no_self_pairs(candidates_df):
    assert candidates_df.filter(
        candidates_df["anchor_product_id"] == candidates_df["candidate_product_id"]
    ).is_empty()
