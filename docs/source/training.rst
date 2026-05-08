Model Training
==============

This page contains only verified training details from ``README.md`` and ``documentation.md``.

Entry points
------------

- Pipeline: ``training/src/pipelines/training.py``
- CLI: ``training/src/scripts/run_training_pipeline.py``
- Config: ``training/configs/training_pipeline.yaml``

Run command
-----------

.. code-block:: bash

   ./venv/bin/python -m training.src.scripts.run_training_pipeline \
     --config training/configs/training_pipeline.yaml

Pipeline stages
---------------

.. code-block:: text

   1) Load & preprocess orders
      - clean columns, cast types, deduplicate
      - filter to active kiosks

   2) Time split
      - train/val/test = 80% / 10% / 10%
      - train is split again:
        * feature_orders (first 70% of train)
        * label_orders   (last 30% of train)

   3) Build baskets
      - group by (kiosk_id, date) into product lists

   4) Generate MBA candidates
      - compute cooc_count, support, lift, cosine_sim
      - filters: min_cooc >= 2, min_lift >= 1.2
      - keep top-100 candidates per anchor

   5) Build feature table + labels
      - cross join kiosks × anchors × top-K candidates
      - add documented 8 features
      - add binary labels from co-purchase pairs
      - sample negatives (max 20 per query)

   6) Train LightGBM LambdaRank
      - objective: lambdarank
      - validation early stopping (100)

   7) Evaluate offline
      - HitRate@K, NDCG@K, Recall@K, MRR@K, Precision@K

   8) Save artifacts
      - lgbm_ranker.txt
      - lgbm_ranker.features.json

Leakage control
---------------

``train_label_ratio`` (default ``0.3``) separates feature construction and labels in time, preventing overlap between periods used for features and periods used for target events.

Why this training design
------------------------

Model method
~~~~~~~~~~~~

- LightGBM with ``lambdarank`` is used because the task is ranking candidates per query ``(kiosk, anchor)``.
- Offline metrics are rank-based (NDCG/HitRate/Recall/MRR/Precision@K), so list ranking objective is aligned with evaluation.

Data split method
~~~~~~~~~~~~~~~~~

- Time-based split is used instead of random split to reflect production chronology.
- ``train_label_ratio`` separates feature period and label period to reduce leakage risk.

Candidate strategy
~~~~~~~~~~~~~~~~~~

- Candidate generation uses MBA statistics with filters ``min_cooc >= 2`` and ``min_lift >= 1.2``.
- Top-K candidate cap (``top_k=100`` in training) keeps ranking space manageable before model scoring.

Regularization profile
~~~~~~~~~~~~~~~~~~~~~~

- Documented hyperparameters are conservative (e.g., limited depth/leaves and non-zero L1/L2).
- This matches the stated goal of controlling overfitting on sparse co-purchase data.

Key hyperparameters
-------------------

.. code-block:: yaml

   lgbm_params:
     learning_rate: 0.02
     num_leaves: 15
     max_depth: 5
     min_data_in_leaf: 1000
     min_gain_to_split: 0.5
     lambda_l1: 2.0
     lambda_l2: 10.0
     feature_fraction: 0.55
     bagging_fraction: 0.6
     bagging_freq: 1

Artifacts
---------

- ``training/models/lgbm_ranker.txt``
- ``training/models/lgbm_ranker.features.json``
- ``training/data/interim/orders_sample.parquet``
- ``logs/training_*.log``
