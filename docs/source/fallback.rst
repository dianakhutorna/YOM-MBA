Fallback System
===============

Overview
--------

The serving layer uses a multi-level fallback chain to guarantee non-empty recommendations.

Fallback levels
---------------

1. Level 1: model predictions for known ``(kiosk, anchor)`` pairs.
2. Level 2: per-anchor fallback from co-purchase signals.
3. Level 3: per-category fallback.
4. Level 4: global fallback.

Artifact mapping
----------------

.. list-table::
	 :header-rows: 1

	 * - Level
		 - Artifact
		 - Typical usage
	 * - Level 1
		 - ``predictions.parquet``
		 - Known kiosk and anchor
	 * - Level 2
		 - ``popularity_fallback.parquet``
		 - Unknown kiosk, known anchor
	 * - Level 3
		 - ``category_fallback.parquet``
		 - Unknown anchor with known category context
	 * - Level 4
		 - ``global_fallback.parquet``
		 - Last resort for unknown contexts

Score normalization
-------------------

``documentation.md`` describes score normalization across fallback levels so ranking remains consistent when fallback outputs are merged with model outputs.

Coverage
--------

The documented behavior is to preserve non-empty output coverage even when kiosk or anchor context is missing.

Why fallback is structured this way
-----------------------------------

- It preserves API reliability for cold-start and sparse-history cases.
- It degrades from personalized to generic recommendations in a controlled sequence.
- It keeps serving logic deterministic and inexpensive at request time.
