Configuration
=============

Overview
--------

Runtime behavior is controlled by YAML configs in ``training/configs``.

- ``training_pipeline.yaml``: training pipeline
- ``generate_predictions.yaml``: batch scoring pipeline

Training config reference
-------------------------

Main keys from ``training/configs/training_pipeline.yaml``:

.. list-table::
   :header-rows: 1

   * - Key
     - Value / Default
   * - ``raw_paths``
     - list of raw CSV paths
   * - ``n_rows``
     - ``3000000``
   * - ``sample_position``
     - ``tail``
   * - ``train_ratio``
     - ``0.8``
   * - ``val_ratio``
     - ``0.1``
   * - ``test_ratio``
     - ``0.1``
   * - ``train_label_ratio``
     - ``0.3``
   * - ``min_cooc``
     - ``2``
   * - ``min_lift``
     - ``1.2``
   * - ``top_k``
     - ``100``
   * - ``top_k_train``
     - ``100``
   * - ``label_window_days``
     - ``7``
   * - ``min_cooc_label``
     - ``1``
   * - ``label_kiosk_batch_size``
     - ``0`` (auto)
   * - ``max_neg_per_group``
     - ``20``
   * - ``max_eval_queries``
     - ``50000``
   * - ``eval_ks``
     - ``[5,10,20,50]``
   * - ``predict_batch_size``
     - ``200000``
   * - ``num_boost_round``
     - ``2000``
   * - ``early_stopping_rounds``
     - ``100``
   * - ``eval_log_path``
     - ``logs/training_eval_curve.csv``

Documented LightGBM parameter block:

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
     seed: 42

Inference config reference
--------------------------

Main keys from ``training/configs/generate_predictions.yaml``:

.. list-table::
   :header-rows: 1

   * - Key
     - Value / Default
   * - ``orders_path``
     - interim parquet path
   * - ``products_path``
     - products CSV path
   * - ``commerces_path``
     - commerces CSV path
   * - ``model_path``
     - model artifact path
   * - ``predictions_path``
     - predictions parquet path
   * - ``popularity_path``
     - per-anchor fallback path
   * - ``category_fallback_path``
     - category fallback path
   * - ``global_fallback_path``
     - global fallback path
   * - ``inference_last_n_days``
     - ``90``
   * - ``inference_max_rows``
     - ``0`` (unlimited)
   * - ``min_cooc``
     - ``2``
   * - ``min_lift``
     - ``1.2``
   * - ``top_k_candidates``
     - ``50``
   * - ``catalog_top_k``
     - ``30``
   * - ``predict_batch_size``
     - ``200000``
   * - ``query_sample_n``
     - ``0``

Operational notes
-----------------

- Inference columns are aligned against ``lgbm_ranker.features.json``.
- Missing inference features are filled with defaults by code.
- Numeric nulls are filled with ``0``; categorical nulls with ``__MISSING__``.
