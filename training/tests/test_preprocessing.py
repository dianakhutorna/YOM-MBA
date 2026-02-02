from __future__ import annotations

def test_preprocess_orders_columns(cleaned_orders_df):
    expected = {"order_id", "kiosk_id", "product_id", "order_dt", "quantity"}
    assert expected.issubset(set(cleaned_orders_df.columns))


def test_preprocess_orders_non_nulls(cleaned_orders_df):
    for col in ("order_id", "kiosk_id", "product_id", "order_dt"):
        assert cleaned_orders_df.select(col).null_count().item() == 0


def test_split_orders_by_time(cleaned_orders_df):
    from training.src.steps.split_orders import split_orders_by_time

    train, val, test = split_orders_by_time(
        cleaned_orders_df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    total = train.height + val.height + test.height
    assert total == cleaned_orders_df.height
