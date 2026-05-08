Batch Scoring (Inference)
=========================

Overview
--------

Batch scoring generates serving catalogs from a trained model.
The script is ``training/src/scripts/generate_predictions.py``.

Run
---

.. code-block:: bash

   ./venv/bin/python -m training.src.scripts.generate_predictions \
     --config training/configs/generate_predictions.yaml

Input artifacts
---------------

- ``training/data/interim/orders_sample.parquet``
- ``training/data/external/products_v2.csv``
- ``training/data/external/commerces.csv``
- ``training/models/lgbm_ranker.txt``
- ``training/models/lgbm_ranker.features.json``

Main process
------------

1. Load model and canonical feature list.
2. Filter orders to active kiosks.
3. Select recent inference window (``inference_last_n_days``).
4. Build baskets and MBA candidates.
5. Build feature table for ``(kiosk, anchor, candidate)`` triples.
6. Align columns to model feature schema and fill missing defaults.
7. Predict in batches using ``predict_batch_size``.
8. Keep top ``catalog_top_k`` per ``(kiosk, anchor)``.
9. Save prediction catalog parquet.
10. Build and save fallback artifacts.

Configuration (actual YAML defaults)
------------------------------------

+---------------------------+---------+---------------------------------------------------+
| Parameter                 | Value   | Purpose                                           |
+===========================+=========+===================================================+
| ``inference_last_n_days`` | 90      | Recency window                                    |
+---------------------------+---------+---------------------------------------------------+
| ``inference_max_rows``    | 0       | 0 means unlimited rows in selected window         |
+---------------------------+---------+---------------------------------------------------+
| ``min_cooc``              | 2       | MBA candidate minimum co-occurrence               |
+---------------------------+---------+---------------------------------------------------+
| ``min_lift``              | 1.2     | MBA minimum lift                                  |
+---------------------------+---------+---------------------------------------------------+
| ``top_k_candidates``      | 50      | Candidates per anchor before scoring              |
+---------------------------+---------+---------------------------------------------------+
| ``catalog_top_k``         | 30      | Final top-N per ``(kiosk, anchor)``               |
+---------------------------+---------+---------------------------------------------------+
| ``predict_batch_size``    | 200000  | Batch size for model prediction                   |
+---------------------------+---------+---------------------------------------------------+
| ``query_sample_n``        | 0       | 0 means all queries                               |
+---------------------------+---------+---------------------------------------------------+

Outputs
-------

Primary prediction artifact:

- ``training/data/interim/predictions.parquet``

Fallback artifacts:

- ``training/data/interim/popularity_fallback.parquet``
- ``training/data/interim/category_fallback.parquet``
- ``training/data/interim/global_fallback.parquet``

Fallback generation details
---------------------------

The script computes fallback scores and normalizes them to the model score range:

- Per-anchor fallback: based on ``cooc_cosine_sim``.
- Category fallback: based on log-scaled purchase counts per category.
- Global fallback: based on log-scaled global purchase counts.

Why this design
---------------

- Batch mode removes online model-scoring dependency.
- Feature-schema alignment keeps training/inference compatible.
- Separate fallback artifacts improve coverage for sparse or unknown contexts.
