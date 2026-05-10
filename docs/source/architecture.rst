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

Design Decisions
-----------------

This page explains the architectural and technology choices behind the YOM Bundle Recommender and the reasoning behind each decision.

Offline first architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~

The system is designed to run without online model inference. Recommendations are pre-computed in batch for ``(kiosk, anchor, candidate)`` combinations, and serving performs fast dictionary lookups. This keeps latency low and makes the system suitable for offline deployment.

Market Basket Analysis for candidate generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project uses Market Basket Analysis as the candidate retrieval stage because it is simple, interpretable, and well aligned with co-purchase data. MBA captures frequent item associations directly from baskets and provides a strong baseline for bundle recommendation without requiring a heavy retrieval model.

LightGBM for ranking
~~~~~~~~~~~~~~~~~~~~

The ranker uses LightGBM with a ``lambdarank`` objective because the task is ranking candidates within each query group. This objective matches the evaluation setup better than classification or regression losses, and the model remains compact and fast at inference time.

Separation of retrieval and ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Candidate generation and final scoring are kept as separate stages. This makes the pipeline easier to debug, allows direct comparison between the MBA baseline and the final ranker, and keeps the scoring space manageable before model prediction.

Technology Choices
------------------

ZenML
~~~~~

ZenML is used as the pipeline orchestration framework. It provides clear step boundaries, reproducible pipeline runs, and caching during development, which helps avoid re-running expensive data preparation and training steps.

Polars instead of Pandas
~~~~~~~~~~~~~~~~~~~~~~~~

Polars is used for most data processing because the training data is stored in large Parquet files and the pipeline benefits from a multi-threaded query engine. Pandas is only used where a downstream library requires it.

Parquet as the storage format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intermediate datasets are stored as Parquet because it is efficient for large tabular data, preserves schema, and works well with columnar reads during training and batch scoring.

Lightweight serving artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The trained model and batch outputs are stored as small local artifacts so the serving layer can load everything at startup without any network dependency.
