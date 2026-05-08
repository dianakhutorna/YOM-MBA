Features
========

This page documents only the features explicitly described in ``README.md`` and ``documentation.md``.

Feature source
--------------

- Feature computation: ``training/src/steps/add_features.py``
- Feature orchestration and alignment: ``training/src/features.py``
- Feature list used by model: ``training/models/lgbm_ranker.features.json``

Model feature set
-----------------

The ranking model uses 8 features:

+----------------------+----------------------+-----------------------------------------------------------+
| Feature              | Type                 | Description                                               |
+======================+======================+===========================================================+
| ``cooc_cosine_sim``  | float                | MBA cosine similarity between anchor and candidate        |
+----------------------+----------------------+-----------------------------------------------------------+
| ``pop_store``        | int                  | Candidate purchase count in this kiosk                    |
+----------------------+----------------------+-----------------------------------------------------------+
| ``pop_global``       | int                  | Candidate purchase count across all kiosks                |
+----------------------+----------------------+-----------------------------------------------------------+
| ``kiosk_product_cnt``| int                  | Total order rows for kiosk (activity proxy)               |
+----------------------+----------------------+-----------------------------------------------------------+
| ``cand_is_new``      | binary (0/1)         | 1 if kiosk has never ordered candidate before             |
+----------------------+----------------------+-----------------------------------------------------------+
| ``same_category``    | binary (0/1)         | 1 if anchor and candidate share category                  |
+----------------------+----------------------+-----------------------------------------------------------+
| ``channel``          | categorical (hashed) | Kiosk channel from commerces metadata                     |
+----------------------+----------------------+-----------------------------------------------------------+
| ``region``           | categorical (hashed) | Kiosk region from commerces metadata                      |
+----------------------+----------------------+-----------------------------------------------------------+

Processing rules
----------------

- ``channel`` and ``region`` are hash-encoded before model use.
- Numeric columns are cast to ``Float64`` and missing values are filled with 0.
- At inference time, feature columns are aligned to ``lgbm_ranker.features.json``.

What is intentionally excluded
------------------------------

No undocumented feature families are listed here.

Why these features
------------------

The documented feature set combines three signal types used by the model:

- **Co-purchase signal:** ``cooc_cosine_sim`` from MBA candidate generation.
- **Popularity and personalization signal:** ``pop_store``, ``pop_global``, ``kiosk_product_cnt``, ``cand_is_new``.
- **Context signal:** ``same_category``, ``channel``, ``region``.

This design follows the training and inference pipeline structure:

- MBA provides candidate-level association strength.
- Store/global counters provide stable priors and kiosk-level personalization.
- Categorical kiosk metadata provides context when behavior-only signals are sparse.

Documented feature-importance summary highlights:

1. ``cooc_cosine_sim`` as strongest signal.
2. ``pop_store`` for kiosk-level personalization.
3. ``pop_global`` as global popularity prior.
4. ``kiosk_product_cnt`` as kiosk activity proxy.
