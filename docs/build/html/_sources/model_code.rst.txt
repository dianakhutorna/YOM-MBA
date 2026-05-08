Model Code Walkthrough
======================

Overview
--------

This page documents technical implementation details from the codebase for model training and scoring.

Training pipeline internals
---------------------------

Primary module: ``training/src/pipelines/training.py``.

The pipeline defines ``TrainingPipelineConfig`` and reads YAML through ``from_yaml``.
Core stages are implemented via step modules in ``training/src/steps``.

Key implementation details:

- Query grouping key is ``(kiosk_id, anchor_product_id)`` via synthetic ``query_id``.
- ``filter_good_queries`` keeps only queries with both positive and negative examples.
- ``sample_negatives`` caps negatives per query using ``max_neg_per_group``.
- ``shuffle_within_query`` randomizes row order inside query groups before LightGBM input.
- ``fill_missing_features`` and ``ensure_feature_columns`` enforce stable feature schema.

LightGBM interface details:

- Objective is LambdaRank.
- Group arrays are generated from query sizes.
- Feature order is persisted to ``lgbm_ranker.features.json`` for inference-time alignment.

Batch scoring internals
-----------------------

Primary module: ``training/src/scripts/generate_predictions.py``.

Key implementation details:

- Loads model and feature list from ``lgbm_ranker.txt`` and sibling ``.features.json``.
- Uses batched prediction in ``_predict_scores_batched`` with ``predict_batch_size``.
- Aligns inference columns to model expectations; missing columns are filled with defaults.
- Applies null handling: numeric to ``0``, categorical to ``__MISSING__``.
- Logs diagnostic warnings for zero-only numeric features.

Prediction catalog generation:

- Sorts by ``score`` descending inside each ``(kiosk_id, anchor_product_id)``.
- Keeps top ``catalog_top_k`` candidates.
- Saves candidate and anchor names using joins with product metadata.

Fallback artifact generation in scoring code
--------------------------------------------

The batch script also writes fallback artifacts:

- Per-anchor fallback from ``cooc_cosine_sim``
- Per-category popularity fallback
- Global popularity fallback

Implementation details include score normalization of fallback signals into model score range before persisting parquet files.

Serving implementation internals
--------------------------------

Primary modules:

- ``training/src/scripts/serve_recommendations_api.py``
- ``training/src/services/recommendation_service.py``
- ``training/src/scripts/lambda_handler.py``

Runtime behavior:

- FastAPI app initializes service during lifespan startup.
- Service downloads predictions parquet from S3 (using ``PREDICTIONS_S3_BUCKET`` and ``PREDICTIONS_S3_KEY``).
- ``RecommendationService.from_parquet`` builds lookup dict keyed by ``(anchor_id, kiosk_id)``.
- ``GET /recommendations`` validates ``limit`` in range ``[1, 100]``.
- Lambda handler is Mangum wrapper around FastAPI app.

API response schema is produced from typed response models in ``training/src/models``.

Operational implication
-----------------------

Model scoring and fallback artifact construction are offline concerns.
Serving remains request-time lookup logic, keeping runtime behavior predictable.
