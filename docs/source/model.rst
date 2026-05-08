Model Artifacts
===============

Overview
--------

This page documents only model artifacts and behavior described in ``README.md`` and ``documentation.md``.

Main ranking model
------------------

- Artifact: ``training/models/lgbm_ranker.txt``
- Method: LightGBM with ``lambdarank`` objective
- Task: rank candidate products for each query group ``(kiosk, anchor)``

Documented hyperparameters include:

- ``learning_rate: 0.02``
- ``num_leaves: 15``
- ``max_depth: 5``
- ``min_data_in_leaf: 1000``
- ``min_gain_to_split: 0.5``
- ``lambda_l1: 2.0``
- ``lambda_l2: 10.0``
- ``feature_fraction: 0.55``
- ``bagging_fraction: 0.6``
- ``bagging_freq: 1``

Feature metadata
----------------

- Artifact: ``training/models/lgbm_ranker.features.json``
- Purpose: define the exact feature column list and order used for training/prediction alignment

The documented model uses 8 features (see :doc:`features`).

Batch inference artifacts
-------------------------

Batch scoring produces four serving artifacts:

- ``training/data/interim/predictions.parquet``
- ``training/data/interim/popularity_fallback.parquet``
- ``training/data/interim/category_fallback.parquet``
- ``training/data/interim/global_fallback.parquet``

These files are loaded by the serving layer at startup.

Why model design is structured this way
---------------------------------------

- ``lambdarank`` is used because the objective is top-K ranking per query, consistent with rank metrics used in evaluation.
- The model is trained offline and applied in batch to avoid per-request online inference.
- Serving uses precomputed artifacts and fallback levels to keep latency low and maintain recommendation coverage.

References
----------

- :doc:`training`
- :doc:`inference`
- :doc:`serving`
- :doc:`features`
