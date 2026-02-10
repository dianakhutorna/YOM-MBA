Three stages: Training → Generation of Predictions → Serving
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
│                   training.py (ONCE)                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load raw CSV                                                 │
│ 2. Preprocess → interim orders.parquet                          │
│ 3. Split into train/val/test (by time)                          │
│ 4. Build baskets from train_orders                              │
│ 5. Generate candidates (MBA: cooc, lift, cosine_sim)            │
│ 6. Build feature table (join kiosk×anchor + candidates)         │
│ 7. Add features (product, behavioral, categorical, etc.)        │
│ 8. Build labels (from test_orders)                              │
│ 9. Train LightGBM lambdarank with eval on val/test              │
│ 10. Save model + feature list                                   │
│                                                                 │
│ OUTPUT:                                                         │
│  - lgbm_ranker.txt (model)                                      │
│  - lgbm_ranker.features.json (inference feature list)           │
│  - orders_sample.parquet (cleaned orders)                       │
│  - logs/ (training metrics)                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               INFERENCE / BATCH SCORING                         │
│            generate_predictions.py (OFTEN)                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load saved model (lgbm_ranker.txt)                           │
│ 2. Load feature list (features.json)                            │
│ 3. Load recent orders (DB or interim)                           │
│ 4. Build baskets (from those orders)                            │
│ 5. Generate candidates (same MBA metrics)                       │
│ 6. Build feature table                                          │
│ 7. Add features (SAME FLAGS as training!)                       │
│ 8. Align columns:                                               │
│    - Drop extra features                                        │
│    - Add missing features (zeros)    CAREFUL!                   │
│ 9. Predict scores (LightGBM)                                    │
│ 10. Save predictions.parquet                                    │
│ 11. Build popularity fallback (cold start)                      │
│                                                                 │
│ OUTPUT:                                                         │
│  - predictions.parquet (kiosk, anchor, candidate, score)        │
│  - popularity_fallback.parquet (top products by frequency)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING (production)                         │
│              serve_bundle.py (ALWAYS ONLINE)                    │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load predictions.parquet into memory (index by kiosk_id)     │
│ 2. For each request:                                            │
│    - Get kiosk_id + anchor_product_id (from context)            │
│    - Lookup in predictions → top-K candidates + scores          │
│    - If missing (cold start):                                   │
│      * Fallback to popularity_fallback.parquet                  │
│    - Return JSON recommendations                                │
│                                                                 │
│ OUTPUT:                                                         │
│  - JSON API response (<1ms per request)                         │
└─────────────────────────────────────────────────────────────────┘

What runs once vs. often?

training.py    🔴 Rare (weekly/monthly)
Costly: preprocess + candidates + features + train; needs fresh data

generate_predictions.py    🟡 Often (daily/weekly)
Fast: reuse trained model, re-score candidates on recent data

serve_bundle.py    🟢 Always (prod 24/7)
Simple in-memory lookup, <1ms per request; refreshed when new parquet is generated


⚠️ Critical points

Feature consistency: generate_predictions MUST use the SAME feature flags as training (features_config_path), otherwise:

Missing features → zeros inserted ❌
Zero-only features → model breaks ❌

Candidate alignment: top_k in training vs top_k_candidates in generate_predictions must match, otherwise candidates differ.

Split ratios: if you change train_ratio between training and inference, candidates will differ.
