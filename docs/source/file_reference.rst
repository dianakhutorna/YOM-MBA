File Reference
==============

Overview
--------

This page summarizes the source modules referenced in ``documentation.md``.

Core pipeline files
-------------------

.. list-table::
	 :header-rows: 1

	 * - File
		 - Purpose
	 * - ``training/src/pipelines/training.py``
		 - End-to-end training pipeline
	 * - ``training/src/scripts/run_training_pipeline.py``
		 - CLI wrapper for training pipeline
	 * - ``training/src/scripts/generate_predictions.py``
		 - Batch scoring and fallback artifact generation

Serving files
-------------

.. list-table::
	 :header-rows: 1

	 * - File
		 - Purpose
	 * - ``training/src/scripts/serve_recommendations_api.py``
		 - FastAPI application and endpoints
	 * - ``training/src/scripts/lambda_handler.py``
		 - AWS Lambda entry point
	 * - ``training/src/services/recommendation_service.py``
		 - Lookup and recommendation logic

Pipeline step modules
---------------------

.. list-table::
	 :header-rows: 1

	 * - Module
		 - Purpose
	 * - ``training/src/steps/preprocessing.py``
		 - Cleaning and normalization
	 * - ``training/src/steps/split_orders.py``
		 - Time-based data split
	 * - ``training/src/steps/build_baskets.py``
		 - Basket construction
	 * - ``training/src/steps/generate_candidates.py``
		 - MBA candidate generation
	 * - ``training/src/steps/select_top_k_candidates.py``
		 - Candidate filtering
	 * - ``training/src/steps/build_feature_table.py``
		 - Triple table generation ``(kiosk, anchor, candidate)``
	 * - ``training/src/steps/add_features.py``
		 - Feature computation
	 * - ``training/src/steps/build_labels.py``
		 - Label generation
	 * - ``training/src/steps/rank_eval_at_k.py``
		 - Ranking metrics

Support modules
---------------

- ``training/src/io/loaders.py``
- ``training/src/features.py``
- ``training/src/config.py``
- ``training/src/paths.py``
- ``training/src/logging_utils.py``

Analysis scripts
----------------

- ``training/src/scripts/check_personalization.py``
- ``training/src/scripts/check_new_vs_repeat.py``
