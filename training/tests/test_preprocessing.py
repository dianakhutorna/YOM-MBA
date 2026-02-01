from __future__ import annotations

def test_preprocess_orders_columns(cleaned_orders_df):
    expected = {"order_id", "kiosk_id", "product_id", "order_dt", "quantity"}
    assert expected.issubset(set(cleaned_orders_df.columns))


def test_preprocess_orders_non_nulls(cleaned_orders_df):
    for col in ("order_id", "kiosk_id", "product_id", "order_dt"):
        assert cleaned_orders_df.select(col).null_count().item() == 0
