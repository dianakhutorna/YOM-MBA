System Architecture
===================

Overview
--------

YOM Bundle Recommender is an offline ML system with three stages:

1. Training (monthly)
2. Batch scoring (daily/weekly)
3. Serving (AWS Lambda)

Key principle: there is no online ML inference. Recommendations are pre-computed in batch for ``(kiosk, anchor, candidate)`` combinations, and serving performs dictionary lookup.

Stages
------

1) Training
~~~~~~~~~~~

Main flow:

- Load and preprocess raw orders
- Perform time-based split into train/validation/test
- Build baskets and MBA candidates
- Build feature table and labels
- Train LightGBM LambdaRank
- Save artifacts:

  - ``training/models/lgbm_ranker.txt``
  - ``training/models/lgbm_ranker.features.json``

2) Batch scoring
~~~~~~~~~~~~~~~~

Main flow:

- Load trained model and feature list
- Load recent orders window (90 days)
- Generate candidate products per anchor
- Build feature table and score candidates
- Keep top results per ``(kiosk, anchor)``
- Save parquet artifacts for serving

3) Serving
~~~~~~~~~~

Main flow:

- Load parquet artifacts at startup
- Build in-memory index for lookup
- Serve recommendations through API
- Apply fallback chain when direct model prediction is unavailable

Fallback Strategy
-----------------

Serving uses a 4-level fallback:

1. Model predictions
2. Per-anchor fallback
3. Per-category fallback
4. Global fallback

This guarantees non-empty output coverage for known and unknown request combinations.

Core Files
----------

- ``training/src/pipelines/training.py``
- ``training/src/scripts/generate_predictions.py``
- ``training/src/scripts/serve_recommendations_api.py``
- ``training/src/scripts/lambda_handler.py``
- ``training/src/services/recommendation_service.py``
